from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy import stats

from plots.typing import PlotData, PlotOptions, PlotResult, Point


class QQReferenceLine(Enum):
    FIRST_THIRD_QUARTIL = 'FIRST_THIRD_QUARTIL'  # A line that connect the 25th and 75th percentiles of the data and reference distributions
    LEAST_SQUARES_REGRESSION = 'LEAST_SQUARES_REGRESSION'  # A least squares regression line
    THEORETICAL_LINE = 'THEORETICAL_LINE'  # t(x) = 1*x+0, this would match a linear regression the distribution is equal to itself


def draw_qq_plot(
        real_pd: PlotData[npt.ArrayLike],
        theo_pd: PlotData[npt.ArrayLike],
        quantile_count: Optional[int] = None,
        reference_lines: Optional[set[QQReferenceLine]] = None,
        extra_quantile_points: Optional[list[float]] = None,
        plot_options: PlotOptions = PlotOptions('QQ-Plot')
) -> PlotResult:
    if reference_lines is None:
        reference_lines = [QQReferenceLine.FIRST_THIRD_QUARTIL]
    if plot_options.x_label is None:
        plot_options.x_label = "Theoretical Quantiles"
    if plot_options.y_label is None:
        plot_options.y_label = "Sampled Values"

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
    if QQReferenceLine.FIRST_THIRD_QUARTIL in reference_lines:
        p1 = Point(np.quantile(theo_sorted, 0.25), np.quantile(real_sorted, 0.25))
        p2 = Point(np.quantile(theo_sorted, 0.75), np.quantile(real_sorted, 0.75))
        slope = (p1.y - p2.y) / (p1.x - p2.x)
        intercept = p1.y - slope * p1.x
        ax.plot(theo_quant_values, slope * theo_quant_values + intercept, 'r-', alpha=0.9, label='25th and 75th percentile line')

    # 2. A least squares regression line
    if QQReferenceLine.LEAST_SQUARES_REGRESSION in reference_lines:
        slope, intercept, r, prob, _ = stats.linregress(theo_quant_values, real_quant_values)
        ax.plot(theo_quant_values, slope*theo_quant_values + intercept, 'k-', alpha=0.9, label='Least square regression')

    # 3. 45° degree line
    if QQReferenceLine.THEORETICAL_LINE in reference_lines:
        ax.plot(theo_quant_values, theo_quant_values, 'k-', alpha=0.9, label='Theoretical line')

    # X. A line whose intercept and slope are determined by maximum likelihood estimates of the location and scale parameters of the target distribution.

    # plot everything in one go into a normal plot
    # ax.plot(theo_quant_values, real_quant_values, 'bo', theo_quant_values, slope*theo_quant_values + intercept, 'r-')

    if extra_quantile_points is not None and len(extra_quantile_points) > 0:
        points = [Point(np.quantile(theo_sorted, quantile), np.quantile(real_sorted, quantile)) for quantile in extra_quantile_points]
            # ax.plot(theo_quant_values, slope * theo_quant_values + intercept, 'r-', alpha=0.9)
        ax.scatter(list(map(lambda p: p.x, points)), list(map(lambda p: p.y, points)))
        for idx in range(len(extra_quantile_points)):
            # ax.annotate(f"{extra_quantile_points[idx]}q", points[idx])
            ax.annotate(f"{extra_quantile_points[idx]}q", xy=points[idx],  xycoords='data',
                        xytext=(3, -3), textcoords='offset pixels',
                        # xytext=(0.8, 0.95), textcoords='axes fraction',
                        # arrowprops=dict(facecolor='black', shrink=0.05),
                        horizontalalignment='left', verticalalignment='top',
                        )

    ax.set_title(plot_options.title)
    ax.set_xlabel(theo_pd.label if theo_pd.label is not None else plot_options.x_label)
    ax.set_ylabel(real_pd.label if real_pd.label is not None else plot_options.y_label)

    ax.legend(loc="best")

    return PlotResult(fig, ax)
