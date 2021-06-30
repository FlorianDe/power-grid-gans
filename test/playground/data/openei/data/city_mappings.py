from dataclasses import dataclass

from test import City

DEFAULT_PREFIX = 'RefBldg'


@dataclass
class CityDataSet:
    suffix: str
    prefix: str = DEFAULT_PREFIX


CITY_MAPPINGS: dict[City, CityDataSet] = {
    City.USA_NY_Rochester_Greater_Rochester_Intl_AP_725290_TMY3: CityDataSet(
        suffix='New2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv'
    )
}
