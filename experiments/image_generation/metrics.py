import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import distributions
from statsmodels.distributions import ECDF
from statsmodels.tsa.seasonal import STL

from experiments.image_generation.utils import set_latex_plot_params
from experiments.utils import get_experiments_folder
from plots.timeseries_plot import draw_timeseries_plot, DecomposeResultColumns
from src.metrics.kolmogorov_smirnov import ks2_test, ks2_critical_value
from src.plots.histogram_plot import draw_hist_plot, HistPlotData
from src.plots.qq_plot import draw_qq_plot, QQReferenceLine
from src.plots.typing import PlotResult, PlotOptions, PlotData
from src.utils.math_utils import LinearIntervalScaler

DEFAULT_ARROW_COLOR = "C1"


def normal_dist_str(mean, std):
    return r"\mathcal{N}(" + str(mean) + r"," + str(std) + r")"


def unif_dist_str(unif_start, unif_end):
    return r"\mathcal{U}_{[" + str(unif_start) + r"," + str(unif_end) + "]}"


@dataclass
class NamedPlotResult:
    name: str
    plot: PlotResult


def save_ks_test_example_plot() -> PlotResult:
    text_fontsize = 12
    KS_MAX_TEXT_HORIZONTAL_MARGIN = 0.05
    KS_MAX_TEXT_PLACING_FACTOR = 0.75  # set the vertical text position of the KS Test Result in between the arrow [0,1]

    np.random.seed(42)
    n, m = 500, 600
    norm_mean_1, norm_std_1 = 0, 1
    d1 = np.random.normal(norm_mean_1, norm_std_1, n)
    unif_start, unif_end = -3, 4
    d2 = np.random.uniform(unif_start, unif_end, m)

    fig, ax = plt.subplots(ncols=1, nrows=1)

    # ax.set_title('Violin-Plot')
    ax.set_ylabel(r"$\displaystyle{F(x)}$")
    ax.set_xlabel(r"$\displaystyle{x}$")

    ecdf1 = ECDF(d1)
    ecdf2 = ECDF(d2)

    d1.sort()
    d2.sort()
    F1 = ecdf1(d1)
    F2 = ecdf2(d2)

    data_all = sorted(np.concatenate([d1, d2]))

    cdf1 = np.searchsorted(d1, data_all, side='right') / n
    cdf2 = np.searchsorted(d2, data_all, side='right') / m
    max_s, idx = None, None
    for p in range(len(data_all)):
        diff = cdf1[p] - cdf2[p]
        if max_s is None or diff > max_s:
            max_s = diff
            idx = p

    x = data_all[idx]
    y1 = cdf1[idx]
    y2 = cdf2[idx]

    ax.step(d1, F1, where="post", label=r"$\displaystyle{F_{1,n}(x)}\;\text{für}\;X_{1}\sim"+normal_dist_str(norm_mean_1, norm_std_1)+r",\,n=\left\lvert X_{1} \right\rvert = "+str(n)+"$")
    ax.step(d2, F2, where="post", label=r"$\displaystyle{F_{2,m}(x)}\;\text{für}\;X_{2}\sim"+unif_dist_str(unif_start, unif_end)+r",\,n=\left\lvert X_{2} \right\rvert = "+str(m)+"$")
    ax.annotate("",
                xy=(x, y1),
                xytext=(x, y2),
                arrowprops=dict(arrowstyle="<->", color="k")
                )
    ax.text(
        x=x+KS_MAX_TEXT_HORIZONTAL_MARGIN,
        y=(min(y1, y2) + abs(y2 - y1) * KS_MAX_TEXT_PLACING_FACTOR),
        s=r"$\displaystyle{D_{n,m}=" + "{:.2f}".format(max_s)+"}$",
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=text_fontsize,
    )
    ax.legend(loc="best")

    ks_xy = ks2_test(d1, d2)
    crit_xy = ks2_critical_value(d1, d2, 0.05)

    print(f"{ks_xy=}")
    print(f"{crit_xy=}")

    return PlotResult(fig, ax)


def save_timeseries_plot() -> PlotResult:
    np.random.seed(0)
    n = 365 * 4
    dates = np.array('2022-01-01', dtype=np.datetime64) + np.arange(n)
    data = 20 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(5, 2, n)
    df = pd.DataFrame({'data': data}, index=dates)

    decomp_result = STL(df, period=365).fit()

    translations = {
        DecomposeResultColumns.OBSERVED: r"$\displaystyle{\text{Daten}\;Y_t}$",
        DecomposeResultColumns.SEASONAL: r"$\displaystyle{\text{Saisonal}\;S_t}$",
        DecomposeResultColumns.TREND: r"$\displaystyle{\text{Trend}\;T_t}$",
        DecomposeResultColumns.RESID: r"$\displaystyle{\text{Rest}\;R_t}$",
        DecomposeResultColumns.WEIGHTS: r"$\displaystyle{\text{Gewichte}\;W_t}$",
    }
    res = draw_timeseries_plot(
        data=decomp_result,
        translations=translations,
        figsize=(6.4, 6.4)
    )

    return res


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
        ax1.annotate("", xy=(x_poses[0], y_pos), xytext=(x_max - text_offset, y_pos), arrowprops=dict(arrowstyle="->", color=DEFAULT_ARROW_COLOR))
        ax2.annotate("", xy=(x_poses[1], y_pos), xytext=(-x_max + text_offset, y_pos), arrowprops=dict(arrowstyle="->", color=DEFAULT_ARROW_COLOR))

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

    timeseries_decomposition_res = save_timeseries_plot()
    timeseries_decomposition_res.fig.show()
    timeseries_decomposition_res.fig.savefig(metrics_folder / f"timeseries_decomposition_sine_years.pdf", bbox_inches='tight', pad_inches=0)

    ks_test_example_plot_res = save_ks_test_example_plot()
    ks_test_example_plot_res.fig.show()
    ks_test_example_plot_res.fig.savefig(metrics_folder / f"ks_test_example_plot.pdf", bbox_inches='tight', pad_inches=0)
