import os

import matplotlib.pyplot as plt
import pandas as pd
import wget
from pandas import DatetimeIndex, DataFrame

# Base code idea from: https://gitlab.com/midas-mosaik/midas/-/blob/main/src/midas/tools/commercial_data.py
from playground.pytorch.data.openei.data.building import Building
from playground.pytorch.data.openei.data.city import City
from playground.pytorch.data.openei.data.city_mappings import CITY_MAPPINGS

BASE_URL = "https://openei.org/datasets/files/961/pub/COMMERCIAL_LOAD_DATA_E_PLUS_OUTPUT"

COLUMN_TIME = "Date/Time"
COLUMNS_ELECTRICITY = [
    "Electricity:Facility [kW](Hourly)",
    "Fans:Electricity [kW](Hourly)",
    "Cooling:Electricity [kW](Hourly)",
    "Heating:Electricity [kW](Hourly)",
]
DEFAULT_DATE_RANGE = pd.date_range(
    start="2004-01-01 01:00:00",
    end="2004-12-31 00:00:00",
    freq="H",
    tz="Europe/Berlin",
)


class OpenEIImporter:
    def __init__(self, date_range: DatetimeIndex = DEFAULT_DATE_RANGE, tmp_path: str = ".") -> None:
        super().__init__()
        self.date_range = date_range
        self.tmp_path = tmp_path
        self.data_map: dict[City, dict[Building, DataFrame]] = dict()

    def parse(self, cities: list[City], buildings: list[Building]):
        if len(cities) == 0:
            raise ValueError("You provided zero cities to parse from!")

        for city in cities:
            for building in buildings:
                city_folder = os.path.join(self.tmp_path, city.value)
                local_dataset_path = os.path.join(city_folder, f'{building.value}.csv')
                if not os.path.exists(local_dataset_path):
                    city_mapping = CITY_MAPPINGS[city]
                    if city_mapping is None:
                        raise ValueError("This city has no data set mapping yet, cannot be used.")
                    if not os.path.exists(city_folder):
                        print("Creating city folder", city_folder)
                        os.mkdir(city_folder, 0o755)
                    download_url = f'{BASE_URL}/{city.value}/{city_mapping.prefix}{building.value}{city_mapping.suffix}'
                    downloaded_file = wget.download(download_url, out=local_dataset_path)
                    print(f'Downloaded: {downloaded_file} from {download_url}')
                tsdat = pd.read_csv(local_dataset_path, sep=",", usecols=[COLUMN_TIME, *COLUMNS_ELECTRICITY])
                tsdat.index = self.date_range
                if city not in self.data_map:
                    self.data_map[city] = dict()
                self.data_map[city][building] = tsdat
                # tsdat = tsdat[EL_COLS].sum(axis=1)

    def plot(self, data: DataFrame):
        temp_plot_options = {
            'figure.figsize': (20, 5),
            'figure.dpi': 300,
        }
        with plt.rc_context(temp_plot_options):
            data.plot()
            plt.show()


if __name__ == "__main__":
    cities = [
        City.USA_NY_Rochester_Greater_Rochester_Intl_AP_725290_TMY3
    ]
    buildings = [
        Building.SmallOffice,
        Building.Hospital,
        Building.MediumOffice
    ]
    importer = OpenEIImporter()
    importer.parse(cities, buildings)
    importer.plot(importer.data_map[cities[0]][buildings[0]])
