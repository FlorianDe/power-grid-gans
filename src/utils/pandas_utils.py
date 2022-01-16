from typing import Iterable, Any, Union

import numpy as nd
from pandas import DatetimeIndex, Series, Timestamp, DataFrame


def iter_timeseries(data: Series) -> Iterable[tuple[Timestamp, Any]]:
    if not isinstance(data.index, DatetimeIndex):
        raise ValueError("This exclusion only works for Series which have a DatetimeIndex!")
    return zip(iter(data.index), iter(data))


def get_datetime_values(data: Union[Series, DataFrame, DatetimeIndex]) -> tuple[nd.ndarray, nd.ndarray, nd.ndarray]:
    index = data
    if not isinstance(data, DatetimeIndex):
        index = data.index
    if not isinstance(index, DatetimeIndex):
        raise ValueError("The index of the importer is no DatetimeIndex therefore it is not enforced to have a month, day and hour column!")
    return index.month.values, index.day.values, index.hour.values
