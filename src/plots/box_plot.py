from typing import Optional, Callable, Any

import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.axes import Axes

from src.plots.typing import PlotResult, PlotData, PlotOptions


def draw_box_like_plot(
        raw_plot_data: list[PlotData[npt.ArrayLike]],
        plot_options: PlotOptions,
        box_plot_fn: Callable[[Axes, list[npt.ArrayLike]], Any],
        plot: Optional[PlotResult] = None
) -> PlotResult:
    fig, ax = plt.subplots(nrows=1, ncols=1) if plot is None else plot

    raw_data = list(map(lambda pd: pd.data, raw_plot_data))
    raw_data_len = range(len(raw_data))
    data_labels = list(map(lambda pd: pd.label, raw_plot_data))

    # call box plot like plot fn
    box_plot_fn(ax, raw_data)
    ax.set_title(plot_options.title)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in raw_data_len])
    ax.set_xlabel(plot_options.x_label)
    ax.set_ylabel(plot_options.y_label)

    # add x-tick labels
    plt.setp(ax, xticks=[y + 1 for y in raw_data_len], xticklabels=data_labels)

    return PlotResult(fig, ax)


def draw_box_plot(
        raw_plot_data: list[PlotData[npt.ArrayLike]],
        plot_options: PlotOptions = PlotOptions('Box plot'),
        plot: Optional[PlotResult] = None
) -> PlotResult:
    return draw_box_like_plot(
        raw_plot_data=raw_plot_data,
        plot_options=plot_options,
        box_plot_fn=lambda ax, data: ax.boxplot(data),
        plot=plot
    )
