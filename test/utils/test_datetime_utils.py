import unittest
from datetime import date

from src.utils.datetime_utils import convert_input_str_to_date, interval_generator, format_timestamp


class TestIntervalGenerator(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_convert_input_str_to_date(self):
        year = 2020
        month = 2
        day = 3
        d = convert_input_str_to_date(f"{year}.{month:02}.{day:02}")
        self.assertEqual(year, d.year)
        self.assertEqual(month, d.month)
        self.assertEqual(day, d.day)

    def test_convert_input_str_with_invalid_input(self):
        self.assertRaises(ValueError, convert_input_str_to_date, "01.01.2020")

    def test_interval_generator_with_leap_year(self):
        start = convert_input_str_to_date("2020.01.01")
        end = convert_input_str_to_date("2023.12.31")

        days_count = sum(1 for _ in interval_generator(start, end))

        self.assertEqual(4 * 365 + 1, days_count)

    def test_interval_generator_with_start_older_than_start(self):
        self.assertEqual(0, sum(1 for _ in interval_generator(date(2022, 1, 1), date(2021, 1, 1))))

    def test_format_timestamp(self):
        ts_ns = 1635694638531000000  # corresponds to 2021-10-31 16:37:18.531000064
        ts_ns_as_str = "2021_10_31_15_37_18_531000064"
        formatted_ts = format_timestamp(ts_ns)

        self.assertEqual(formatted_ts, ts_ns_as_str)
