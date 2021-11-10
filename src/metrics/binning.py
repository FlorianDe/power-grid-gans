from collections import namedtuple
from typing import Optional

import sys

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy import stats

from statsmodels.distributions.empirical_distribution import ECDF

from data.importer.weather.weather_dwd_importer import DWDWeatherDataImporter, WeatherDataColumns
from metrics.types import PlotData, PlotOptions

Point = namedtuple('Point', 'x y')


def draw_qq_plot(
        real_pd: PlotData[npt.ArrayLike],
        theo_pd: PlotData[npt.ArrayLike],
        quantile_count: Optional[int] = None,
        plot_options: PlotOptions = PlotOptions('QQ-Plot')
) -> Figure:
    real_sorted = sorted(real_pd.data)
    theo_sorted = sorted(theo_pd.data)
    n = min(len(real_sorted), len(theo_sorted))  # choose the minimum number of elements of one of the inputs
    if quantile_count is not None:
        n = min(quantile_count, n)

    quantiles = []
    # Filliben's estimate
    for i in range(1, n + 1):
        if i == 1:
            quantiles.append(1 - 0.5 ** (1 / n))
        elif i == n:
            quantiles.append(0.5 ** (1 / n))
        else:
            quantiles.append((i - 0.3175) / (n + 0.365))

    real_quant_values = np.quantile(real_sorted, quantiles)
    theo_quant_values = np.quantile(theo_sorted, quantiles)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.margins(x=0, y=0)
    ax.scatter(theo_quant_values, real_quant_values, alpha=0.2)
    # 1. A line that connect the 25th and 75th percentiles of the data and reference distributions
    p1 = Point(np.quantile(theo_sorted, 0.25), np.quantile(real_sorted, 0.25))
    p2 = Point(np.quantile(theo_sorted, 0.75), np.quantile(real_sorted, 0.75))
    slope = (p1.y - p2.y) / (p1.x - p2.x)
    intercept = p1.y - slope * p1.x
    ax.plot(theo_quant_values, slope * theo_quant_values + intercept, 'r-', alpha=0.9)

    # 2. A least squares regression line
    # slope, intercept, r, prob, _ = stats.linregress(theo_quant_values, real_quant_values)
    # ax.plot(theo_quant_values, slope*theo_quant_values + intercept, 'k-', alpha=0.9)

    # 3. A line whose intercept and slope are determined by maximum likelihood estimates of the location and scale parameters of the target distribution.

    # plot everything in one go into a normal plot
    # ax.plot(theo_quant_values, real_quant_values, 'bo', theo_quant_values, slope*theo_quant_values + intercept, 'r-')

    ax.set_title(plot_options.title)
    ax.set_xlabel(theo_pd.label)
    ax.set_ylabel(real_pd.label)

    return fig


def draw_hist_plot(pds: list[PlotData[npt.ArrayLike]], bin_width: float = 1) -> Figure:
    min_data = sys.float_info.max
    max_data = sys.float_info.min

    for pd in pds:
        min_data = min(min(pd.data), min_data)
        max_data = max(max(pd.data), max_data)

    bins = np.arange(min_data, max_data + bin_width, bin_width)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for pd in pds:
        ax.hist(pd.data, bins, label=pd.label, color=pd.color, edgecolor='k', alpha=0.5)

    ax.legend(loc='upper right')
    return fig


def bin_plot_data(data: npt.ArrayLike, binwidth: float = 5):
    min_data = min(data)
    max_data = max(data)
    # pmf, bins = np.histogram(data, bins=np.arange(min_data, max_data + binwidth, binwidth), density=True)
    # res = np.column_stack((bins[:-1], pmf))
    # plt.plot(bins, pmf)
    plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    plt.show()


def visualize_pdf(data, pdf):
    # sb.set_style('whitegrid')
    plt.plot(data, pdf, 'r-', lw=2, alpha=0.6, label='expon pdf', color='k')
    plt.xlabel('intervals')
    plt.ylabel('Probability Density')
    plt.show()


def plot_ecdf(data, dist="norm", sparams=()):
    ecdf = ECDF(data)
    plt.plot(ecdf.x, ecdf.y)
    plt.show()

    # plot Q-Q
    res = stats.probplot(data, dist=dist, sparams=sparams)
    plt.show()


if __name__ == '__main__':
    importer = DWDWeatherDataImporter()
    importer.initialize()
    dh_w_per_m2_without_zero = importer.data[importer.data > 1][WeatherDataColumns.DH_W_PER_M2].dropna()
    print(stats.exponweib.fit(dh_w_per_m2_without_zero))

    bin_plot_data(dh_w_per_m2_without_zero, 5)  # normal distribution
    # bin_plot_data(importer.data[WeatherDataColumns.T_AIR_DEGREE_CELSIUS], 1)  # normal distribution
    # bin_plot_data(importer.data[WeatherDataColumns.DH_W_PER_M2], 5)  # exponential/beta? distribution
    # bin_plot_data(importer.data[WeatherDataColumns.GH_W_PER_M2], 5)  # exponential/beta? distribution
    # bin_plot_data(importer.data[WeatherDataColumns.WIND_V_M_PER_S], 1)  # rayleigh distribution
    # bin_plot_data(importer.data[WeatherDataColumns.WIND_DIR_DEGREE], 1)  # random distribution

    # fit_res = expon.fit(importer.data[WeatherDataColumns.DH_W_PER_M2])
    # exp_pdf = expon.pdf(fit_res)
    # visualize_pdf(fit_res, exp_pdf)

    plot_ecdf(importer.data[WeatherDataColumns.T_AIR_DEGREE_CELSIUS])  # good

    # plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.beta, sparams=(0.3, 2))
    # plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.rayleigh, sparams=(306,))
    plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.weibull_min, sparams=(0.19925197526753652,
                                                                                         2.2125306785365266))  # (0.39911867132844703, -1.0116216408139537e-26, 17.55878804720284), without 0 values (0.19925197526753652, 2.7777777777777772, 2.2125306785365266)
    plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.expon, sparams=(2.7777777777777777, 117.69589143207929))
    plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.exponweib, sparams=(0.7521034487635724, 1.064984731518368, 2.7777777777118837,
                                                                                       108.85637427247518))  # (1.5411043773762327, 0.37977251767629466, -6.933195992806934e-30, 100.00266735788577)

    plot_ecdf(importer.data[WeatherDataColumns.WIND_V_M_PER_S], stats.rayleigh)  # good

    plot_ecdf(importer.data[WeatherDataColumns.WIND_DIR_DEGREE], stats.uniform)  # good
