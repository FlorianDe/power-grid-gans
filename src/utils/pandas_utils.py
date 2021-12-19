from typing import Iterable, Any

from pandas import DatetimeIndex, Series, Timestamp


def iter_timeseries(data: Series) -> Iterable[tuple[Timestamp, Any]]:
    if not isinstance(data.index, DatetimeIndex):
        raise ValueError("This exclusion only works for Series which have a DatetimeIndex!")
    return zip(iter(data.index), iter(data))
