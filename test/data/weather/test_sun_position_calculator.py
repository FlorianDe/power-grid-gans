import datetime
import re
import unittest
from pathlib import Path
from astral import LocationInfo
from astral.sun import sun
import numpy as np

from src.data.weather.sun_position_calculator import DayTime, SunPositionCalculator

class TestSunPositionCalculator(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_instantiation(self):
        start = DayTime(0)
        self.assertEqual(start.hour, 0)
        self.assertEqual(start.minute, 0)
        self.assertEqual(start.second, 0)

        mid = DayTime(0.5)
        self.assertEqual(mid.hour, 12)
        self.assertEqual(mid.minute, 0)
        self.assertEqual(mid.second, 0)

        end = DayTime(1-1e-9)
        self.assertEqual(end.hour, 23)
        self.assertEqual(end.minute, 59)
        self.assertEqual(end.second, 59)

    def test_instantiation_errors(self):
        self.assertRaises(ValueError, DayTime, 0-1e-9)
        self.assertRaises(ValueError, DayTime, 1+1e-9)

    def test_daytime_equality(self):
        for s in np.linspace(0, 1-1e-9, 100):
            self.assertEqual(DayTime(s), DayTime(s))

    def test_sun_position_calculator(self):
        """
        Tests against the following data from https://www.dwd.de/DE/fachnutzer/luftfahrt/teaser/luftsportberichte/eddw/node.html:
        EDDW  53Â°02'50.64" N 08Â°47'12.29" O
            ==> converted to (lat=53.04739, long=8.78672)
        """

        expected_sun_positions_bremen_file = Path(__file__).parent / "eddw_bremen.txt"
        lines = open(expected_sun_positions_bremen_file, "r").readlines()
        regex = r"^(\w{2})\s+(?P<day>[\d-]+)\s+(\d+:\d+)\s+(?P<sunrise_h>\d+):(?P<sunrise_m>\d+)\s+(?P<sunset_h>\d+):(?P<sunset_m>\d+)"
        s = SunPositionCalculator(lat=53.0474, long=8.786747)
        city = LocationInfo("Bremen", "Germany", "Europe/Berlin", 53.0474, 8.786747)  # swap with your location
        max_sunrise_diff = 0
        max_sunset_diff = 0
        for line in lines:
            matches = re.search(regex, line)
            day = matches.group('day')
            sunrise_h = matches.group('sunrise_h')
            sunrise_m = matches.group('sunrise_m')
            sunset_h = matches.group('sunset_h')
            sunset_m = matches.group('sunset_m')

            dt = datetime.datetime.fromisoformat(f'{day}T00:00:00+00:00')  # times are in UTC
            res_custom = s.calc(dt)
            custom_sunrise = res_custom.sunrise
            custom_sunset = res_custom.sunset

            res_astral = sun(city.observer, date=dt)
            astral_sunrise = res_astral['sunrise']
            astral_sunset = res_astral['sunset']

            real_sunrise = DayTime.from_datetime(astral_sunrise)
            real_sunset = DayTime.from_datetime(astral_sunset)

            expected_sunrise = self.__get_daytime(sunrise_h, sunrise_m)
            expected_sunset = self.__get_daytime(sunset_h, sunset_m)

            sunrise_diff = abs(expected_sunrise.raw_value - real_sunrise.raw_value)
            if max_sunrise_diff < sunrise_diff:
                max_sunrise_diff = sunrise_diff
                print(f"{day}: {str(dt)} == {str(expected_sunrise)} => {max_sunrise_diff} => {str(DayTime(max_sunrise_diff))}")
            sunset_diff = abs(expected_sunset.raw_value - real_sunset.raw_value)
            if max_sunset_diff < sunset_diff:
                max_sunset_diff = sunset_diff
                print(f"{day}: {str(dt)} == {str(expected_sunrise)} => {max_sunset_diff} => {str(DayTime(max_sunset_diff))}")

            sunrise_comparision = real_sunrise.compare_to(expected_sunrise, 0.00069444)  # 1min => 0.69444...%
            self.assertTrue(sunrise_comparision)

            sunset_comparision = real_sunset.compare_to(expected_sunset, 0.00069444)  # 1min => 0.69444...%
            self.assertTrue(sunset_comparision)

    def __get_daytime(self, hour_str: str, minute_str: str) -> DayTime:
        hour = int(hour_str)
        minute = int(minute_str)
        return DayTime.from_time(hour, minute)










