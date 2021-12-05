from typing import Optional

import numpy.typing as npt

from plots.box_plot import draw_box_like_plot
from plots.typing import PlotResult, PlotData, PlotOptions


def draw_violin_plot(
        raw_plot_data: list[PlotData[npt.ArrayLike]],
        plot_options: PlotOptions = PlotOptions('Violin plot'),
        plot: Optional[PlotResult] = None
) -> PlotResult:
    return draw_box_like_plot(
        raw_plot_data=raw_plot_data,
        plot_options=plot_options,
        box_plot_fn=lambda ax, data: ax.violinplot(data, showmeans=False, showmedians=True),
        plot=plot
    )
