from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from src.plots.typing import PlotData, PlotOptions, PlotResult, PlotColor, PlotDataType
from src.utils.plot_utils import get_min_max


@dataclass
class HistPlotData(PlotData[PlotDataType]):
    edgecolor: PlotColor = "k"
    alpha_fill: float = 0.5


def draw_hist_plot(
    pds: list[HistPlotData[npt.ArrayLike]],
    bin_width: float = 1,
    bins: Optional[float] = None,
    normalized: bool = False,
    plot_options: PlotOptions = PlotOptions("Histogram-Plot"),
    plot: Optional[PlotResult] = None,
) -> PlotResult:
    fig, ax = plt.subplots(nrows=1, ncols=1) if plot is None else plot

    if bins is None:
        min_value, max_value = get_min_max(list(map(lambda pd: pd.data, pds)))
        bins = np.arange(min_value, max_value + bin_width, bin_width)

    for pd in pds:
        ax.hist(
            pd.data,
            bins,
            label=pd.label,
            color=pd.color,
            edgecolor=pd.edgecolor,
            alpha=pd.alpha_fill,
            density=normalized,
        )

    ax.set_title(plot_options.title)
    if plot_options.x_label:
        ax.set_xlabel(plot_options.x_label)
    if plot_options.y_label:
        ax.set_ylabel(plot_options.y_label)
    ax.legend(loc="best")
    return PlotResult(fig, ax)
