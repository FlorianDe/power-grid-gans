import unittest
import numpy as np
from numpy.testing import assert_array_equal

from src.data.weather.weather_filters import relative_wind_dir_calculation


class TestWeatherFilters(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_relative_wind_dir_calculation(self):
        raw_wind_directions = np.array([10, 170, 340, 30, 100, 20, 310])
        expected_wind_direction_deltas = np.array([0, 160, 170, 50, 70, -80, -70])

        assert_array_equal(relative_wind_dir_calculation(raw_wind_directions), expected_wind_direction_deltas)
