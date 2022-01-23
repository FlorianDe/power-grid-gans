import functools
from dataclasses import dataclass

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import distributions

from experiments.image_generation.utils import set_latex_plot_params
from experiments.utils import get_experiments_folder
from src.plots.histogram_plot import draw_hist_plot, HistPlotData
from src.plots.qq_plot import draw_qq_plot, QQReferenceLine
from src.plots.typing import PlotResult, PlotOptions, PlotData
from src.utils.math_utils import LinearIntervalScaler


def normal_dist_str(mean, std):
    return r"\mathcal{N}(" + str(mean) + r"," + str(std) + r")"


def unif_dist_str(unif_start, unif_end):
    return r"\mathcal{U}_{[" + str(unif_start) + r"," + str(unif_end) + "]}"


@dataclass
class NamedPlotResult:
    name: str
    plot: PlotResult


def save_qq_plot_defaults() -> list[NamedPlotResult]:
    np.random.seed(42)
    n = 5000
    unif_start, unif_end = 0, 1
    d1 = np.random.uniform(unif_start, unif_end, int(n / 2))

    norm_mean, norm_std = 0, 0.1
    d2 = np.random.normal(norm_mean, norm_std, n)

    results: list[NamedPlotResult] = []
    for line in [
        QQReferenceLine.THEORETICAL_LINE,
        QQReferenceLine.FIRST_THIRD_QUARTIL,
        QQReferenceLine.LEAST_SQUARES_REGRESSION
    ]:
        qq_res = draw_qq_plot(
            PlotData(data=d1, label=r'$\displaystyle{\text{Stichprobe} \sim ' + unif_dist_str(unif_start, unif_end) + '}$'),
            PlotData(data=d2, label=r"$\displaystyle{\text{Theoretische Verteilung} \sim " + normal_dist_str(norm_mean, norm_std) + "}$"),
            5000,
            {line},
            [0.25, 0.5, 0.75]
        )
        results.append(NamedPlotResult(line.name, qq_res))

    return results


def save_qq_plot_norm_vs_norm() -> PlotResult:
    np.random.seed(42)
    n = 5000

    norm_mean_theo, norm_std_theo = 0, 0.1
    norm_mean_test, norm_std_test = 0, 0.1
    d_theo = np.random.normal(norm_mean_theo, norm_std_theo, n)
    d_test = np.random.normal(norm_mean_test, norm_std_test, n)

    qq_res = draw_qq_plot(
        PlotData(data=d_test, label=r'$\displaystyle{\text{Stichprobe} \sim ' + normal_dist_str(norm_mean_test, norm_std_test) + '}$'),
        PlotData(data=d_theo, label=r"$\displaystyle{\text{Theoretische Verteilung} \sim " + normal_dist_str(norm_mean_theo, norm_std_theo) + "}$"),
        5000,
        {
            QQReferenceLine.THEORETICAL_LINE,
            # QQReferenceLine.FIRST_THIRD_QUARTIL,
            # QQReferenceLine.LEAST_SQUARES_REGRESSION
        },
        [0.25, 0.5, 0.75]
    )

    return qq_res


def save_histogram(normalized: bool = False) -> PlotResult:
    np.random.seed(42)
    n = 5000
    params = [
        (-0.5, 0.3, n, "A"),
        (0, 0.3, 1000, "B"),
        (0.5, 0.2, 2000, "C")
    ]
    pds = [
        HistPlotData(
            data=np.random.normal(p[0], p[1], p[2]),
            label=r"$" + str(p[3]) + r"\sim \mathcal{N}(" + str(p[0]) + r"," + str(p[1]) + r"), \left\lvert " + str(p[3]) + r"\right\rvert =" + str(p[2]) + r"$"
        ) for p in params
    ]
    mean, var = distributions.norm.fit(pds[2].data)
    print(mean, var)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    all_values = functools.reduce(lambda acc, cur: np.concatenate((acc, cur.data)), pds, [])
    bin_width = (max(all_values) - min(all_values)) / 50
    draw_hist_plot(
        pds=pds,
        bin_width=bin_width,
        normalized=normalized,
        plot_options=PlotOptions(
            x_label="$x$",
            y_label="Relative Häufigkeitsdichte" if normalized else "Absolute Häufigkeit"
        ),
        plot=PlotResult(fig, ax)
    )
    return PlotResult(fig, ax)


def save_box_vs_violinplot() -> PlotResult:
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
    margin = abs(data_max - data_min) * 0.1
    figure_y_min, figure_y_max = data_min - margin, data_max + margin
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    whisker_low = q1 - (q3 - q1) * 1.5
    whisker_high = q3 + (q3 - q1) * 1.5
    (ax1, ax2) = gs.subplots()  # plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharex='row')
    fig.subplots_adjust(wspace=0)
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
        text_offset = 0.035 * len(text)
        ax1.annotate("", xy=(x_poses[0], y_pos), xytext=(x_max - text_offset, y_pos), arrowprops=dict(arrowstyle="->", color="C1"))
        ax2.annotate("", xy=(x_poses[1], y_pos), xytext=(-x_max + text_offset, y_pos), arrowprops=dict(arrowstyle="->", color="C1"))

    draw_desc("Ausreißer", whisker_high + 4, (0.05, -0.02))
    draw_desc("oberer Whisker", whisker_high - 1.8, (0.2, -0.02))
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
    sns.set_theme()
    sns.set_context("paper")
    set_latex_plot_params()

    generated_images_folder = get_experiments_folder().joinpath("generated_images")

    metrics_folder = generated_images_folder.joinpath(f"metrics")
    metrics_folder.mkdir(parents=True, exist_ok=True)

    violin_plot_res = save_box_vs_violinplot()
    violin_plot_res.fig.show()
    violin_plot_res.fig.savefig(metrics_folder / f"boxplot_vs_violin_plot.pdf", bbox_inches='tight', pad_inches=0)

    histogram_res_default = save_histogram()
    histogram_res_default.fig.show()
    histogram_res_default.fig.savefig(metrics_folder / f"histogram_default.pdf", bbox_inches='tight', pad_inches=0)

    histogram_res_normed = save_histogram(True)
    histogram_res_normed.fig.show()
    histogram_res_normed.fig.savefig(metrics_folder / f"histogram_normed.pdf", bbox_inches='tight', pad_inches=0)

    qq_plot_defaults_res = save_qq_plot_defaults()
    for res in qq_plot_defaults_res:
        res.plot.show()
        res.plot.fig.savefig(metrics_folder / f"qq_plot_default_{res.name}.pdf", bbox_inches='tight', pad_inches=0)
    qq_plot_norm = save_qq_plot_norm_vs_norm()
    qq_plot_norm.show()
    qq_plot_norm.fig.savefig(metrics_folder / f"qq_plot_norm_vs_norm.pdf", bbox_inches='tight', pad_inches=0)
