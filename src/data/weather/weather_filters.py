from datetime import datetime
from typing import Union

import numpy
from astral import LocationInfo
from astral.sun import sun
from pandas import DataFrame, Series, date_range

from src.data.processor.pandas import PandasPreprocessor
from src.data.weather.sun_position_calculator import DayTime
from src.utils.pandas_utils import iter_timeseries


def clip_to_zero(data: Series) -> Series:
    return PandasPreprocessor(data).clip(0).run()


def exclude_night_time_values(data: Series, location: LocationInfo = LocationInfo("Bremen", "Germany", "Europe/Berlin", 53.0474, 8.786747)) -> Series:
    res_data = data.copy()
    last_py_date_time: Union[datetime, None] = None
    last_sunrise: DayTime = DayTime(1 - 1e-9)
    last_sunset: DayTime = DayTime(0)

    for index, value in iter_timeseries(data):
        cur_py_datetime = index.to_pydatetime()
        if last_py_date_time is None or (
                last_py_date_time.year != cur_py_datetime.year or
                last_py_date_time.month != cur_py_datetime.month or
                last_py_date_time.day != cur_py_datetime.day
        ):
            last_py_date_time = cur_py_datetime
            res_astral = sun(location.observer, date=last_py_date_time)
            last_sunrise = DayTime.from_datetime(res_astral['sunrise'])
            last_sunset = DayTime.from_datetime(res_astral['sunset'])
        if not (last_sunrise <= DayTime.from_datetime(cur_py_datetime) <= last_sunset):
            res_data[index] = None
        # print(f"{index} => {res_data[index]}")

    return res_data


if __name__ == '__main__':
    data = DataFrame(
        index=date_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 3, 23, 59, 59),
            tz="Europe/Berlin",
            freq="H",
        )
    )
    data["A"] = numpy.random.normal(50, 10, 3 * 24)

    exclude_night_time_values(data["A"])
