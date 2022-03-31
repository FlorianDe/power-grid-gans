from datetime import datetime
from typing import Optional, Union

import numpy as np
from astral import LocationInfo
from astral.sun import sun
from pandas import DataFrame, Series, date_range

from src.data.processor.pandas import PandasProcessor
from src.data.weather.sun_position_calculator import DayTime
from src.utils.pandas_utils import iter_timeseries

WIND_DIRECTION_MAX_DEGREE = 360


def clip_to_zero(data: Series) -> Series:
    return PandasProcessor(data).clip(0).run()


def wind_dir_cleansing(data: Series) -> Series:
    return PandasProcessor(data).clip(0).modulo(WIND_DIRECTION_MAX_DEGREE).run()


def relative_wind_dir_calculation(raw_wind_directions: np.ndarray, _: Optional[float] = None) -> np.ndarray:
    wind_direction_deltas = np.ediff1d(raw_wind_directions, to_begin=0)
    delta_overshoots_indices = np.where((np.abs(wind_direction_deltas) > (WIND_DIRECTION_MAX_DEGREE // 2)))
    delta_overshoots_signs = np.sign(wind_direction_deltas[delta_overshoots_indices])
    overshoots_values = wind_direction_deltas[delta_overshoots_indices]
    wind_direction_deltas[delta_overshoots_indices] = (
        overshoots_values - WIND_DIRECTION_MAX_DEGREE * delta_overshoots_signs
    )
    return wind_direction_deltas


def exclude_night_time_values(
    data: Series, location: LocationInfo = LocationInfo("Bremen", "Germany", "Europe/Berlin", 53.0474, 8.786747)
) -> Series:
    res_data = data.copy()
    last_py_date_time: Union[datetime, None] = None
    last_sunrise: DayTime = DayTime(1 - 1e-9)
    last_sunset: DayTime = DayTime(0)

    for index, value in iter_timeseries(data):
        cur_py_datetime = index.to_pydatetime()
        if last_py_date_time is None or (
            last_py_date_time.year != cur_py_datetime.year
            or last_py_date_time.month != cur_py_datetime.month
            or last_py_date_time.day != cur_py_datetime.day
        ):
            last_py_date_time = cur_py_datetime
            res_astral = sun(location.observer, date=last_py_date_time)
            last_sunrise = DayTime.from_datetime(res_astral["sunrise"])
            last_sunset = DayTime.from_datetime(res_astral["sunset"])
        if not (last_sunrise <= DayTime.from_datetime(cur_py_datetime) <= last_sunset):
            res_data[index] = None
        # print(f"{index} => {res_data[index]}")

    return res_data


if __name__ == "__main__":
    data = DataFrame(
        index=date_range(
            start=datetime(2021, 1, 1),
            end=datetime(2021, 1, 3, 23, 59, 59),
            tz="Europe/Berlin",
            freq="H",
        )
    )
    data["A"] = np.random.normal(50, 10, 3 * 24)

    exclude_night_time_values(data["A"])
