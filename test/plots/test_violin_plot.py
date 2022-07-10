import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.plots.typing import PlotData, PlotResult, PlotOptions
from src.plots.box_plot import draw_box_plot
from src.plots.violin_plot import draw_violin_plot


def test_draw_violin_plot():
    sns.set_theme()
    sns.set_context("paper")
    all_data = [PlotData(data=np.random.normal(0, std, 100), label=f"{std=}") for std in range(6, 10)]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

    draw_violin_plot(
        all_data, PlotOptions(title="Violin plot opt", x_label="X_lbl", y_label="Y_lbl"), PlotResult(fig, axes[0])
    )
    draw_box_plot(
        all_data, PlotOptions(title="Box plot opt", x_label="X_lbl", y_label="Y_lbl"), PlotResult(fig, axes[1])
    )
    fig.tight_layout()
    fig.suptitle("Box like plots")
    fig.subplots_adjust(wspace=0.4, top=0.8)
    # fig.show()
