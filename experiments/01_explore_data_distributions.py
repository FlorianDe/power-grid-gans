import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import stats

from statsmodels.distributions.empirical_distribution import ECDF

from data.importer.weather.weather_dwd_importer import DWDWeatherDataImporter, WeatherDataColumns


def bin_plot_data(data: npt.ArrayLike, title: str, binwidth: float = 5):
    min_data = min(data)
    max_data = max(data)
    # pmf, bins = np.histogram(data, bins=np.arange(min_data, max_data + binwidth, binwidth), density=True)
    # res = np.column_stack((bins[:-1], pmf))
    # plt.plot(bins, pmf)
    plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    plt.title(f"Histogram: {title}")
    plt.show()


def visualize_pdf(data, pdf):
    # sb.set_style('whitegrid')
    plt.plot(data, pdf, 'r-', lw=2, alpha=0.6, label='expon pdf', color='k')
    plt.xlabel('intervals')
    plt.ylabel('Probability Density')
    plt.show()


def plot_ecdf(data, title: str, dist="norm", sparams=()):
    ecdf = ECDF(data)
    plt.plot(ecdf.x, ecdf.y)
    plt.title(f"ECDF-{title}")
    plt.show()

    # plot Q-Q
    fig, ax = plt.subplots(nrows=1, ncols=1)
    res = stats.probplot(data, dist=dist, sparams=sparams, plot=ax)
    print(f"{title=} with {dist} -> {res=}")
    ax.set_title(f"QQ-Plot-{title}")
    fig.show()


if __name__ == '__main__':
    importer = DWDWeatherDataImporter()
    importer.initialize()
    ##dh_w_per_m2_without_zero = importer.data[importer.data > 1][WeatherDataColumns.DH_W_PER_M2].dropna()
    ##print(stats.exponweib.fit(dh_w_per_m2_without_zero))
    # bin_plot_data(dh_w_per_m2_without_zero, 5)  # normal distribution

    bin_plot_data(importer.data[WeatherDataColumns.T_AIR_DEGREE_CELSIUS], WeatherDataColumns.T_AIR_DEGREE_CELSIUS, 1)  # normal distribution
    bin_plot_data(importer.data[WeatherDataColumns.DH_W_PER_M2], WeatherDataColumns.DH_W_PER_M2,  5)  # exponential/beta? distribution
    # bin_plot_data(importer.data[WeatherDataColumns.GH_W_PER_M2], WeatherDataColumns.GH_W_PER_M2, 5)  # exponential/beta? distribution
    bin_plot_data(importer.data[WeatherDataColumns.WIND_V_M_PER_S], WeatherDataColumns.WIND_V_M_PER_S, 1)  # rayleigh distribution
    bin_plot_data(importer.data[WeatherDataColumns.WIND_DIR_DEGREE], WeatherDataColumns.WIND_DIR_DEGREE, 1)  # uniform/random distribution

    # fit_res = expon.fit(importer.data[WeatherDataColumns.DH_W_PER_M2])
    # exp_pdf = expon.pdf(fit_res)
    # visualize_pdf(fit_res, exp_pdf)

    # Check this for fittings: https://stackoverflow.com/a/16651955/11133168
    plot_ecdf(importer.data[WeatherDataColumns.T_AIR_DEGREE_CELSIUS], WeatherDataColumns.T_AIR_DEGREE_CELSIUS)  # good

    plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], WeatherDataColumns.DH_W_PER_M2, stats.beta, sparams=(0.3, 2))
    # plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.rayleigh, sparams=(306,))
    # plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.weibull_min, sparams=(0.19925197526753652, 2.2125306785365266))  # (0.39911867132844703, -1.0116216408139537e-26, 17.55878804720284), without 0 values (0.19925197526753652, 2.7777777777777772, 2.2125306785365266)
    # plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.expon, sparams=(2.7777777777777777, 117.69589143207929))
    # plot_ecdf(importer.data[WeatherDataColumns.DH_W_PER_M2], stats.exponweib, sparams=(0.7521034487635724, 1.064984731518368, 2.7777777777118837, 108.85637427247518))  # (1.5411043773762327, 0.37977251767629466, -6.933195992806934e-30, 100.00266735788577)

    plot_ecdf(importer.data[WeatherDataColumns.WIND_V_M_PER_S], WeatherDataColumns.WIND_V_M_PER_S, stats.rayleigh)  # good

    plot_ecdf(importer.data[WeatherDataColumns.WIND_DIR_DEGREE], WeatherDataColumns.WIND_DIR_DEGREE, stats.uniform)  # good

    # hack
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.show()