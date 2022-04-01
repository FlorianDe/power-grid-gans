import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Callable, Optional, Final, Union

import numpy as np
import pandas as pd
import wget
from pandas import DataFrame

from src.data.weather.weather_filters import clip_to_zero, relative_wind_dir_calculation, wind_dir_cleansing
from src.utils.path_utils import unzip, get_root_project_path
from src.utils.plot_utils import plot_dfs
from src.utils.python_ext import FinalClass
from src.utils.pandas_utils import get_datetime_values, get_day_of_year_values

# Base code from: https://gitlab.com/midas-mosaik/midas/-/blob/main/src/midas/tools/weather_data.py
#
# Horizontal solar radiation is provided as hourly sum in Joule/cm^2
# (i.e., correct would be Joule/s/cm^2 * 3600s), but we want Watt/m^2
# for our PV models. Since 1*W = 1*J/s we first need to get back to
# J/s by dividing by 3600. Next, we want to convert from cm^2 to m^2,
# which is by multiplying with 0.0001, however, since cm^2 is in the
# divisor, we need to divide by that value (or multiply with the
# reciprocal). So the calculation we need to apply is
# 1 / (3.6*1e^3) * 1 / 1e^-4 = 1e^4 / (3.6*1e^3) = 1e^1 / 3.6
# which is equal to:


DATE_TIME_FORMAT = "%Y%m%d%H"

JOULE_TO_WATT = 10 / 3.6
DATE_COL = 1
DATE_COL_SOL = 8
DEFAULT_DATA_CACHE_FOLDER = "cached-data"
DEFAULT_DWD_WEATHER_DATA_PATH = "weather/dwd"
DEFAULT_DATA_START_DATE = "2009-01-01 00:00:00"
DEFAULT_DATA_END_DATE = "2019-12-31 23:00:00"
BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/"
PRODUKT_FILE_NAME_BEGINNING = "produkt"


class WeatherDimension(Enum):
    AIR = "AIR"
    SOLAR = "SOLAR"
    WIND = "WIND"
    SUN = "SUN"
    CLOUD = "CLOUD"


@dataclass
class ExtraCalculatedMapping:
    targetColumn: str
    calculate: Callable[[np.ndarray, float], np.ndarray]


@dataclass
class ColumnMapping:
    sourceColumn: str
    targetColumn: str
    unitFactor: float = 1.0  # unit transformation factor
    extraMappings: List[ExtraCalculatedMapping] = field(default_factory=list)
    dataPreprocessing: Optional[Callable[[DataFrame], DataFrame]] = None


@dataclass
class WeatherDataSet:
    fileUrlPath: str
    columns: List[ColumnMapping]
    fileSuffix: str = ".zip"
    dateTimeParser: Callable[[str], datetime] = lambda date: datetime.strptime(date, DATE_TIME_FORMAT)


class WeatherDataColumns(metaclass=FinalClass):
    T_AIR_DEGREE_CELSIUS: Final[str] = "t_air_degree_celsius"  # Temperatur Celsius
    DH_W_PER_M2: Final[str] = "dh_w_per_m2"  # Stundensumme der diffusen solaren Strahlung
    GH_W_PER_M2: Final[str] = "gh_w_per_m2"  # Stundensumme der Globalstrahlung
    WIND_V_M_PER_S: Final[str] = "wind_v_m_per_s"  # Windgeschwindigkeit in m/s
    WIND_DIR_DEGREE: Final[str] = "wind_dir_degree"  # Windrichtung in Grad [0 - 350] in 10er Schritten
    WIND_DIR_DEGREE_DELTA: Final[
        str
    ] = "wind_dir_degree_delta"  # Windrichtung Änderung zum vorherigen Element in Grad [-180, 180]
    CLOUD_PERCENT: Final[str] = "cloud_percent"  # Prozentuale Wolkenbedeckung
    SUN_HOURS_MIN_PER_H: Final[str] = "sun_hours_min_per_h"


WEATHER_DATA_MAPPING: dict[WeatherDimension, WeatherDataSet] = {
    WeatherDimension.AIR: WeatherDataSet(
        fileUrlPath="air_temperature/historical/stundenwerte_TU_00691_19490101_20201231_hist",
        columns=[
            ColumnMapping(
                sourceColumn="TT_TU",
                targetColumn=WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
                # extraMappings=[
                #     ExtraCalculatedMapping(
                #         targetColumn="day_avg_t_air_degree_celsius",
                #         calculate=lambda data, fac: (
                #                 data.reshape(-1, 24).mean(axis=1).repeat(24) * fac
                #         )
                #     )
                # ]
            )
        ],
    ),
    WeatherDimension.SOLAR: WeatherDataSet(
        fileUrlPath="solar/stundenwerte_ST_00691_row",
        dateTimeParser=lambda date: datetime.strptime(date.split(":")[0], "%Y%m%d%H"),
        columns=[
            ColumnMapping("FD_LBERG", WeatherDataColumns.DH_W_PER_M2, JOULE_TO_WATT, dataPreprocessing=clip_to_zero),
            ColumnMapping("FG_LBERG", WeatherDataColumns.GH_W_PER_M2, JOULE_TO_WATT, dataPreprocessing=clip_to_zero),
        ],
    ),
    WeatherDimension.WIND: WeatherDataSet(
        fileUrlPath="wind/historical/stundenwerte_FF_00691_19260101_20201231_hist",
        columns=[
            ColumnMapping("   F", WeatherDataColumns.WIND_V_M_PER_S, dataPreprocessing=clip_to_zero),
            ColumnMapping(
                "   D",
                WeatherDataColumns.WIND_DIR_DEGREE,
                1,
                dataPreprocessing=wind_dir_cleansing,
                extraMappings=[
                    ExtraCalculatedMapping(
                        targetColumn=WeatherDataColumns.WIND_DIR_DEGREE_DELTA,
                        calculate=relative_wind_dir_calculation,
                    )
                ],
            ),
        ],
    ),
    # Length mismatch
    # WeatherDimension.CLOUD: WeatherDataSet(
    #     fileUrlPath="cloudiness/historical/stundenwerte_N_00691_19490101_20201231",
    #     columns=[
    #         ColumnMapping(" V_N", WeatherDataColumns.CLOUD_PERCENT, 12.5),
    #     ]
    # ),
    # WeatherDimension.SUN: WeatherDataSet(
    #     fileUrlPath="sun/historical/stundenwerte_SD_00691_19510101_20201231_hist",
    #     columns=[
    #         ColumnMapping("SD_SO", WeatherDataColumns.SUN_HOURS_MIN_PER_H),
    #     ]
    # ),
}


class DWDWeatherDataImporter:
    def __init__(
        self,
        start_date: Union[str, datetime] = DEFAULT_DATA_START_DATE,
        end_date: Union[str, datetime] = DEFAULT_DATA_END_DATE,
        auto_preprocess=True,
        dimensions: Optional[set[WeatherDataColumns]] = None,
        path: Optional[str] = None,
    ):
        if dimensions is None:
            self.dimensions = WEATHER_DATA_MAPPING.keys()
        else:
            self.dimensions = dimensions

        if path is None:
            root = get_root_project_path()
            self.path = str(root.joinpath(DEFAULT_DATA_CACHE_FOLDER).joinpath(DEFAULT_DWD_WEATHER_DATA_PATH).absolute())
        else:
            self.path = path
        self.start_date = start_date
        self.end_date = end_date
        self.auto_preprocess = auto_preprocess
        self.data: pd.DataFrame = pd.DataFrame(
            index=pd.date_range(
                start=self.start_date,
                end=self.end_date,
                tz="Europe/Berlin",
                freq="H",
            )
        )

    def initialize(self):
        print("1. Download step started.")
        self.__download()
        print("2. Import step started.")
        self.__load()
        if self.auto_preprocess is True:
            print("3. Preprocess step started.")
            self.__preprocess()

    def __load(self):
        for dimension in self.dimensions:
            self.__load_data_dimension(dimension)

        # data.to_hdf(output_path, "weather", "w") # export data to hdf

    def __preprocess(self):
        for dimension in WEATHER_DATA_MAPPING.keys():
            weather_data_set = WEATHER_DATA_MAPPING[dimension]
            for column in weather_data_set.columns:
                target = column.targetColumn
                if column.dataPreprocessing is not None and self.data[target] is not None:
                    processed_data = column.dataPreprocessing(self.data[target])
                    self.data[target] = processed_data

    def __download(self):
        os.makedirs(self.path, exist_ok=True)

        for weatherDimension, weatherSource in WEATHER_DATA_MAPPING.items():
            file_download_url = f"{BASE_URL}/{weatherSource.fileUrlPath}{weatherSource.fileSuffix}"
            weather_source_file_name = file_download_url.rsplit("/", 1)[-1]
            if not os.path.exists(os.path.join(self.path, weather_source_file_name)):
                weather_source_file_name = wget.download(file_download_url, out=self.path)
                print(f"Downloaded {file_download_url}")
                unzip(self.path, weather_source_file_name, weatherDimension.value.lower())
                print(f"Unzipped {weather_source_file_name}")
            else:
                print(f"Using cached values for dimension {weatherDimension}")

    def __load_data_dimension(self, target: WeatherDimension) -> None:
        target_dimension_name = target.value.lower()
        weather_data_set = WEATHER_DATA_MAPPING[target]
        target_folder_name = os.path.join(self.path, target_dimension_name)
        files = os.listdir(target_folder_name)
        target_data_file_name = [f for f in files if f.startswith(PRODUKT_FILE_NAME_BEGINNING)][0]
        target_data_file = os.path.join(target_folder_name, target_data_file_name)

        csv = pd.read_csv(
            target_data_file, sep=";", index_col=1, parse_dates=[1], date_parser=weather_data_set.dateTimeParser
        )
        csv = csv.loc[self.start_date : self.end_date]

        for column in weather_data_set.columns:
            print(f"{column.targetColumn}, src_col: {column.sourceColumn}, fac: {column.unitFactor}")
            self._assign_column_data(column.targetColumn, csv[column.sourceColumn].values * column.unitFactor)
            for extraMapping in column.extraMappings:
                print(
                    f"--> calculated data: tar_mean_col: {extraMapping.targetColumn}, src_col: {column.sourceColumn}, fac: {column.unitFactor}"
                )
                self._assign_column_data(
                    extraMapping.targetColumn,
                    extraMapping.calculate(csv[column.sourceColumn].values, column.unitFactor),
                )

    def _assign_column_data(self, target_column, values):
        self.data[target_column] = values

    def get_datetime_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return get_datetime_values(self.data)

    def get_day_of_year_values(self) -> np.ndarray:
        return get_day_of_year_values(self.data)

    def get_feature_labels(self):
        return self.data.columns

    def get_data_subset(self, columns: Optional[set[WeatherDataColumns]] = None) -> DataFrame:
        if columns is None or len(columns) == 0:
            raise ValueError("You have to specify at least one column to retrieve a real subset!")

        return self.data[list(columns)]


if __name__ == "__main__":
    importer = DWDWeatherDataImporter()
    importer.initialize()
    print(importer.data.size)
    plot_dfs([importer.data])
