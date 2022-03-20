from datetime import date, datetime, timedelta
from functools import reduce

import numpy

PANDAS_DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_DATE_INPUT_FORMAT = (
    "%Y.%m.%d"  # Format Codes: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
)
DATETIME_CONDITIONAL_VECTOR_SIZE = 14


def interval_generator(start: date, end: date, delta: timedelta = timedelta(days=1)):
    """Creates a generator which yields elements between the start date <= end date with a step size of timedelta.

    Keyword arguments:
    start -- the start date
    date -- the end date
    timedelta -- time step size - DEFAULT: 1 day
    """
    curr = start
    while curr <= end:
        yield curr
        curr += delta


def convert_input_str_to_date(date_time_str: str, format: str = DEFAULT_DATE_INPUT_FORMAT) -> datetime:
    return datetime.strptime(date_time_str, format)


def dates_to_conditional_vectors(months: numpy.ndarray, days: numpy.ndarray, hours: numpy.ndarray):
    converted_dates = (months << 10) + (days << 5) + hours
    return [
        [int(x) for x in list("{0:0{bits}b}".format(d, bits=DATETIME_CONDITIONAL_VECTOR_SIZE))] for d in converted_dates
    ]


def dates_to_days_in_year(months: numpy.ndarray, days: numpy.ndarray, hours: numpy.ndarray):
    days_in_month = numpy.array(
        [
            31,
            29,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ]
    )

    days_in_month_acc = numpy.add.accumulate(days_in_month).astype(int)

    return days_in_month_acc[months] + days


def format_timestamp(timestamp_ns: int, datetime_format: str = "%Y_%m_%d_%H_%M_%S"):
    ns_to_s_factor = 1e9  # unit factor from seconds to nanoseconds
    dt = datetime.fromtimestamp(timestamp_ns // ns_to_s_factor)
    return "{}_{:09.0f}".format(dt.strftime(datetime_format), timestamp_ns % ns_to_s_factor)


def get_day_in_year_from_date(date: date) -> int:
    return int(date.strftime("%j")) - 1


if __name__ == "__main__":
    start_date: str = "2021.01.03"
    end_date: str = "2020.02.29"
    d = convert_input_str_to_date(start_date)
    print(get_day_in_year_from_date(d))

    print(dates_to_days_in_year(numpy.array([1, 1, 1]), numpy.array([1, 2, 3]), numpy.array([1, 2, 3])))

    a = numpy.array([1, 10, 20, 30])
    idxes = numpy.array([1, 2, 3]).astype(int)
    print(a[idxes])
