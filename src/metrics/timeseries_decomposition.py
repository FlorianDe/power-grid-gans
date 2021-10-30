import numpy as np
import pandas as pd
from matplotlib import pyplot
from numpy.typing import ArrayLike
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult

from src.data.importer.weather.weather_dwd_importer import DWDWeatherDataImporter


def __decompose_sine(seed: int = 0) -> DecomposeResult:
    np.random.seed(seed)
    n = 1500
    dates = np.array('2005-01-01', dtype=np.datetime64) + np.arange(n)
    data = 12 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(12, 2, 1500)
    df = pd.DataFrame({'data': data}, index=dates)

    return seasonal_decompose(df, model='additive', period=365)


def __decompose_weather_data(data: ArrayLike, period: int = int(24 * 365.25)) -> DecomposeResult:
    """
    Apply an additive time series decomposition on

    :param period: Since we are dealing with weather data the periodicity is a year and since we
    are considering multiple years as the input vector so we have to deal with leap days therefore the default period is on avg 365.25 days,
    since the frequency of our data is hourly we have to multiply it with 24 which also recovers for the problem that the periodicity for the time series
    decomposition has to be an integer
    :return: The time series decomposition result including: trend, seasonal, resid, observed
    """

    result = seasonal_decompose(data, model='additive', period=period)
    # print(result.trend)
    # print(result.seasonal)
    # print(result.resid)
    # print(result.observed)

    return result


if __name__ == '__main__':
    importer = DWDWeatherDataImporter()
    importer.initialize()
    data = importer.data['t_air_degree_celsius']

    # difference between leap day and without
    __decompose_weather_data(data, 24 * 365).plot()

    for idx, key in enumerate(importer.data.keys()):
        __decompose_weather_data(importer.data[key]).plot()

    pyplot.show()
