import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils.math_utils import LinearIntervalScaler
from src.plots.typing import PlotResult

from experiments.utils import get_experiments_folder


def plot_box_vs_violinplot() -> PlotResult:
    sns.set_theme()
    sns.set_context("paper")

    figsize = (9, 4)
    text_fontsize = 14
    tick_labels_fontsize = 16
    # arrow_linewidth = 2.5

    x_min = -1
    x_max = 2
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    np.random.seed(552)
    data = np.sort(np.random.randn(6, 500).cumsum(axis=1).ravel())

    data_min, data_max = min(data), max(data)
    margin = abs(data_max-data_min) * 0.1
    figure_y_min, figure_y_max = data_min-margin, data_max+margin
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    whisker_low = q1 - (q3 - q1) * 1.5
    whisker_high = q3 + (q3 - q1) * 1.5
    (ax1, ax2) = gs.subplots() # plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharex='row')
    fig.subplots_adjust(wspace = 0)
    sns.boxplot(y=data, color='CornflowerBlue', ax=ax1)
    sns.violinplot(y=data, color='CornflowerBlue', ax=ax2)
    # Outlier plotting from https://stackoverflow.com/a/66920981/11133168
    # outliers = data[(data > whisker_high) | (data < whisker_low)]
    # sns.scatterplot(y=outliers, x=0, marker='D', color='crimson', ax=ax2)
    # plt.setp((ax2), "yticks", [])
    # plt.setp((ax1, ax2), "xticks", [])
    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position("right")
    # sns.despine(ax=ax1, top=True, left=False, right=True)
    # sns.despine(ax=ax2, top=True, left=True, right=False)
    # ax1.tick_params(labelbottom=True)
    # ax2.tick_params(labelbottom=True)

    scaler = LinearIntervalScaler(source=(figure_y_min, figure_y_max), destination=(0, 1))
    def draw_desc(text: str, y_pos: float, x_poses: (float, float)):
        y_cord = scaler(y_pos)

        print(text, y_pos, y_cord)
        plt.figtext(
            x=1,
            y=y_cord,
            s=text,
            fontsize=text_fontsize,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax1.transAxes,
        )
        text_offset = 0.035*len(text)
        ax1.annotate("", xy=(x_poses[0], y_pos), xytext=(x_max-text_offset, y_pos), arrowprops=dict(arrowstyle="->", color="C1"))
        ax2.annotate("", xy=(x_poses[1], y_pos), xytext=(-x_max+text_offset, y_pos), arrowprops=dict(arrowstyle="->", color="C1"))

    draw_desc("Ausreißer", whisker_high+4, (0.05, -0.02))
    draw_desc("oberer Whisker", whisker_high-1.8, (0.2, -0.02))
    draw_desc("3. Quartil", q3, (0.4, -0.02))
    draw_desc(" Median ", q2, (0.4, -0.02))
    draw_desc("1. Quartil", q1, (0.4, -0.02))
    draw_desc("unterer Whisker", whisker_low if data_min < whisker_low else data_min, (0.2, -0.02))

    ax1.set_ylim(figure_y_min, figure_y_max)
    # ax1.set_title('Box-Plot')
    # ax1.set_xlabel("Box-Plot")
    ax1.set_xticklabels(["Box-Plot"], fontsize=tick_labels_fontsize)

    ax2.set_ylim(figure_y_min, figure_y_max)
    # ax2.set_title('Violin-Plot')
    # ax2.set_xlabel("Violin-Plot")
    ax2.set_xticklabels(["Violin-Plot"], fontsize=tick_labels_fontsize)

    ax2.set_yticklabels([], fontsize=0)

    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(-x_max, -x_min)

    gs.tight_layout(fig, pad=0)

    return PlotResult(fig, ax1)


if __name__ == '__main__':
    generated_images_folder = get_experiments_folder().joinpath("generated_images")

    metrics_folder = generated_images_folder.joinpath(f"metrics")
    metrics_folder.mkdir(parents=True, exist_ok=True)

    fig, axes = plot_box_vs_violinplot()
    fig.show()
    fig.savefig(metrics_folder / f"boxplot_vs_violin_plot.pdf", bbox_inches='tight', pad_inches=0)
