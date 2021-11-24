from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from statsmodels.distributions import StepFunction

from plots.typing import PlotData, PlotColor, PlotResult, PlotOptions, PlotDataType


@dataclass
class ECDFPlotData(PlotData[PlotDataType]):
    confidence_band_alpha: Optional[float] = None
    confidence_band_fill_alpha: Optional[float] = None
    confidence_band_color: Optional[PlotColor] = None


def _conf_set(y, alpha: float):
    nobs = len(y)
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2 * nobs))
    lower = np.clip(y - epsilon, 0, 1)
    upper = np.clip(y + epsilon, 0, 1)
    return lower, upper


def draw_ecdf_plot(
        ecdfs: list[ECDFPlotData[StepFunction]],
        plot_options: PlotOptions = PlotOptions('ECDF-Plot')
) -> PlotResult:
    # if confidence_alpha_alphas is None:
    #     confidence_alpha_alphas = np.repeat(0.5, len(ecdfs))
    #
    # if isinstance(confidence_alpha_alphas, float):
    #     confidence_alpha_alphas = np.repeat(confidence_alpha_alphas, len(ecdfs))
    # else:
    #     if len(ecdfs) is not len(confidence_alpha_alphas):
    #         raise ValueError("You have specified a different number of confidence alpha than the passed ecdf count.")

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for idx in range(len(ecdfs)):
        ecdf = ecdfs[idx]
        x = ecdf.data.x
        y = ecdf.data.y
        if ecdf.confidence_band_alpha != 0.0:
            lower, upper = _conf_set(y, ecdf.confidence_band_alpha)
            ax.fill_between(x, lower, upper, alpha=ecdf.confidence_band_fill_alpha, label=f"Confidence band {ecdf.confidence_band_alpha} \u03B1")
        ax.step(x, y, where="post", label=ecdf.label)
        # plt.step(x, lower, "r", where="post")
        # plt.step(x, upper, "r", where="post")

    # ax.set_xlim(0, 1.5)
    # ax.set_ylim(0, 1.05)
    # ax.vlines(x, 0, 0.05)
    ax.set_title(plot_options.title)
    ax.legend(loc="best")
    return PlotResult(fig, ax)