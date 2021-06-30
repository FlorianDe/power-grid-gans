import os
import numpy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Callable
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
import wget
from pandas import DataFrame

# Base code from: https://gitlab.com/midas-mosaik/midas/-/blob/main/src/midas/tools/weather_data.py
# Horizontal solar radiation is provided as hourly sum in Joule/cm^2
# (i.e., correct would be Joule/s/cm^2 * 3600s), but we want Watt/m^2
# for our PV models. Since 1*W = 1*J/s we first need to get back to
# J/s by dividing by 3600. Next, we want to convert from cm^2 to m^2,
# which is by multiplying with 0.0001, however, since cm^2 is in the
# divisor, we need to divide by that value (or multiply with the
# reciprocal). So the calculation we need to apply is
# 1 / (3.6*1e^3) * 1 / 1e^-4 = 1e^4 / (3.6*1e^3) = 1e^1 / 3.6
# which is equal to:
JOULE_TO_WATT = 10 / 3.6
DATE_COL = 1
DATE_COL_SOL = 8
DATA_START = "2009-01-01 00:00:00"
DATA_END = "2019-12-31 23:00:00"
BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"
PRODUKT_FILE_NAME_BEGINNING = 'produkt'


class WeatherDimension(Enum):
    AIR = 'AIR'
    SOLAR = 'SOLAR'
    WIND = 'WIND'
    SUN = 'SUN'
    CLOUD = 'CLOUD'


@dataclass
class ExtraCalculatedMapping:
    targetColumn: str
    calculate: Callable[[numpy.ndarray, float], numpy.ndarray]


@dataclass
class ColumnMapping:
    sourceColumn: str
    targetColumn: str
    unitFactor: float = 1.0  # unit transformation factor
    extraMappings: List[ExtraCalculatedMapping] = field(default_factory=list)


@dataclass
class WeatherDataSet:
    fileUrlPath: str
    columns: List[ColumnMapping]
    fileSuffix: str = '.zip'
    dateTimeParser: Callable[[str], datetime] = lambda date: datetime.strptime(date, "%Y%m%d%H")


weatherDataSourcesMap: dict[WeatherDimension, WeatherDataSet] = {
    WeatherDimension.AIR: WeatherDataSet(
        fileUrlPath="air_temperature/historical/stundenwerte_TU_00691_19490101_20201231_hist",
        columns=[
            ColumnMapping(
                sourceColumn="TT_TU",
                targetColumn="t_air_degree_celsius",
                extraMappings=[
                    ExtraCalculatedMapping(
                        targetColumn="day_avg_t_air_degree_celsius",
                        calculate=lambda data, fac: (
                                data.reshape(-1, 24).mean(axis=1).repeat(24) * fac
                        )
                    )
                ]
            )
        ]
    ),
    WeatherDimension.SOLAR: WeatherDataSet(
        fileUrlPath="solar/stundenwerte_ST_00691_row",
        dateTimeParser=lambda date: datetime.strptime(date.split(":")[0], "%Y%m%d%H"),
        columns=[
            ColumnMapping("FD_LBERG", "dh_w_per_m2", JOULE_TO_WATT),
            ColumnMapping("FG_LBERG", "gh_w_per_m2", JOULE_TO_WATT)
        ]
    ),
    WeatherDimension.WIND: WeatherDataSet(
        fileUrlPath="wind/historical/stundenwerte_FF_00691_19260101_20201231_hist",
        columns=[
            ColumnMapping("   F", "wind_v_m_per_s"),
            ColumnMapping("   D", "wind_dir_degree", 1)
        ]
    ),
    WeatherDimension.SUN: WeatherDataSet(
        fileUrlPath="sun/historical/stundenwerte_SD_00691_19510101_20201231_hist",
        columns=[
            ColumnMapping("SD_SO", "sun_hours_min_per_h"),
        ]
    ),
    # WeatherDimension.CLOUD: WeatherDataSet(
    #     fileUrlPath="cloudiness/historical/stundenwerte_N_00691_19490101_20201231",
    #     columns=[
    #         ColumnMapping(" V_N", "cloud_percent", 12.5),
    #     ]
    # ),
}


def build_weather_data(path, filename):
    if path is None:
        path = os.path.abspath(
            os.path.join(__file__, "..", "cached")
        )
    if filename is None:
        filename = "WeatherBre2009-2019.hdf5"
    output_path = os.path.abspath(os.path.join(path, filename))
    if os.path.exists(output_path):
        return True

    tmp = os.path.join(path, "tmp")
    os.makedirs(tmp, exist_ok=True)

    for weatherDimension, weatherSource in weatherDataSourcesMap.items():
        file_download_url = f'{BASE_URL}/{weatherSource.fileUrlPath}{weatherSource.fileSuffix}'
        weather_source_file_name = file_download_url.rsplit("/", 1)[-1]
        if not os.path.exists(os.path.join(tmp, weather_source_file_name)):
            print(f'Try downloading {file_download_url}')
            weather_source_file_name = wget.download(file_download_url, out=tmp)
            print(f'Downloaded {file_download_url}')
            unzip(tmp, weather_source_file_name, weatherDimension)
            print(f'Unzipped {weather_source_file_name}')

    data = pd.DataFrame(
        index=pd.date_range(
            start=DATA_START,
            end=DATA_END,
            tz="Europe/Berlin",
            freq="H",
        )
    )
    data = load_data(tmp, WeatherDimension.AIR, data)
    data = load_data(tmp, WeatherDimension.SOLAR, data)
    data = load_data(tmp, WeatherDimension.WIND, data)

    # data = load_data(tmp, "cloud", data)  # Length missmatch
    # data = load_data(tmp, "sun", data)

    # data = data.clip(lower=0) # clip outlier data

    # path = os.path.join(path, "weather_bre2009-2019.hdf5")
    # data.to_hdf(output_path, "weather", "w") # export data to hdf
    # data = data.loc['2009-01-01 00:00:00':'2009-12-4 23:00:00'] # redefine timespan
    # plot([
    #     data['wind_dir_degree'],
    #     data['wind_v_m_per_s'],
    #     data['t_air_degree_celsius'],
    #     data['day_avg_t_air_degree_celsius'],
    #     data['day_avg_t_air_degree_celsius'],
    #     data['dh_w_per_m2'],
    #     data['gh_w_per_m2'],
    # ])


def plot(data: List[DataFrame]):
    temp_plot_options = {
        'figure.figsize': (20, 5),
        'figure.dpi': 300,
        'lines.linewidth': 2
    }
    with plt.rc_context(temp_plot_options):
        for i, d in enumerate(data):
            plt.figure(i)
            plot = d.plot()
            # plt.show()
            fig = plot.get_figure()
            fig.savefig(f'cached/test{i}.eps', format='eps', dpi=1200)


def unzip(path: str, fname: str, target: WeatherDimension):
    with ZipFile(os.path.join(path, fname), "r") as zip_ref:
        zip_ref.extractall(os.path.join(path, target.value.lower()))


def load_data(path: str, target: WeatherDimension, data: DataFrame):
    target_dimension_name = target.value.lower()
    weather_data_set = weatherDataSourcesMap[target]
    target_folder_name = os.path.join(path, target_dimension_name)
    files = os.listdir(target_folder_name)
    target_data_file_name = [f for f in files if f.startswith(PRODUKT_FILE_NAME_BEGINNING)][0]
    target_data_file = os.path.join(target_folder_name, target_data_file_name)

    csv = pd.read_csv(
        target_data_file, sep=";", index_col=1, parse_dates=[1], date_parser=weather_data_set.dateTimeParser
    )
    csv = csv.loc[DATA_START:DATA_END]

    for column in weather_data_set.columns:
        print(f'Data transformation: src_col: {column.sourceColumn}, tar_col: {column.targetColumn}, fac: {column.unitFactor}')
        data[column.targetColumn] = csv[column.sourceColumn].values * column.unitFactor
        for extraMapping in column.extraMappings:
            print(f'Extra mean data transformation: src_col: {column.sourceColumn}, tar_mean_col: {extraMapping.targetColumn}, fac: {column.unitFactor}')
            data[extraMapping.targetColumn] = extraMapping.calculate(csv[column.sourceColumn].values, column.unitFactor)

    return data


if __name__ == "__main__":
    build_weather_data(None, None)
