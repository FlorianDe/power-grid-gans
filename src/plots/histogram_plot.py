from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from src.plots.typing import PlotData, PlotOptions, PlotResult, PlotColor, PlotDataType
from src.utils.plot_utils import get_min_max


@dataclass
class HistPlotData(PlotData[PlotDataType]):
    edgecolor: PlotColor = 'k'
    alpha_fill: float = 0.5


def draw_hist_plot(
        pds: list[HistPlotData[npt.ArrayLike]],
        bin_width: float = 1,
        plot_options: PlotOptions = PlotOptions('Histogram-Plot'),
) -> PlotResult:
    min_value, max_value = get_min_max(list(map(lambda pd: pd.data, pds)))

    bins = np.arange(min_value, max_value + bin_width, bin_width)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for pd in pds:
        ax.hist(pd.data, bins, label=pd.label, color=pd.color, edgecolor=pd.edgecolor, alpha=pd.alpha_fill)

    ax.set_title(plot_options.title)
    ax.legend(loc="best")
    return PlotResult(fig, ax)
