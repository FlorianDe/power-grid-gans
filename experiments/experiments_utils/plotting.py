from enum import Enum
from typing import Optional

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib import ticker

import torch
from torch import Tensor

from src.plots.typing import Locale

from sine_data import SineGenerationParameters
from train_typing import TrainParameters

PLOT_LANG = Locale.DE


class SimpleGanPlotResultColumns(Enum):
    LOSS = "loss"
    EPOCH = "epoch"
    ITERATION = "iteration"


__PLOT_DICT: dict[SimpleGanPlotResultColumns, dict[Locale, str]] = {
    SimpleGanPlotResultColumns.LOSS: {Locale.EN: "Loss", Locale.DE: "Verlust"},
    SimpleGanPlotResultColumns.EPOCH: {Locale.EN: "Epoch", Locale.DE: "Epoche"},
    SimpleGanPlotResultColumns.ITERATION: {Locale.EN: "Iteration", Locale.DE: "Iteration"},
}


def translate(key: SimpleGanPlotResultColumns) -> str:
    return __PLOT_DICT[key][PLOT_LANG]


def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_sample(
    sample: Tensor, params: TrainParameters, plot: tuple[Figure, Axes], condition: Optional[int] = None
) -> tuple[Figure, Axes]:
    sample = sample.cpu()  # We have to convert it to cpu too, to allow matplot to plot it
    fig, ax = plot if plot is not None else plt.subplots(nrows=1, ncols=1)
    unbind_sample = torch.unbind(sample)
    flattened_sample = torch.concat(unbind_sample)

    def create_y_label() -> str:
        y_lbl = r"$"
        y_lbl += r"G(Z"
        if condition is not None:
            y_lbl += r"\mid C_{"
            y_lbl += str(condition)
            y_lbl += r"}"
        y_lbl += r"), Z \sim \mathcal{N}(0,1), Z \in \mathcal{R}^{"
        y_lbl += str(sample.size(0))
        y_lbl += r" \times "
        y_lbl += str(params.latent_vector_size)
        y_lbl += r"}"
        # y_lbl += str(len(sample))
        y_lbl += r"$"
        return y_lbl

    for i, y in enumerate(torch.transpose(flattened_sample, 0, 1)):
        x = range(len(y))
        ax.plot(x, y, label=r"$f_{" + str(i) + r"}^{t}$")

    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel(
        create_y_label(),
        fontsize=12,
    )
    ax.legend(loc="upper right")
    return fig, ax


def plot_train_data_overlayed(
    samples: list[Tensor],
    samples_parameters: list[SineGenerationParameters],
    params: TrainParameters,
    plot: Optional[tuple[Figure, Axes]] = None,
) -> tuple[Figure, Axes]:
    if len(samples) != len(samples_parameters):
        raise ValueError("The specified samples and sample parameters have to have the same length.")
    samples = [sample.cpu() for sample in samples]

    def create_sample_legend_string(idx: int, samples_parameters: SineGenerationParameters) -> str:
        amplitude_vec = str(sample_params.amplitudes)
        sequence_len = r"s"  # str(params.sequence_len)

        random_var_name = chr(65 + ((idx + 23) % 26))  # 0 => X, 1 => Y ...
        eq = r"$"
        eq += random_var_name
        eq += r"_{[t]_{"
        eq += sequence_len
        eq += r"}} \sim "
        eq += r"\sin(\frac{2\pi t}{"
        eq += sequence_len
        eq += r"})"
        eq += r" \cdot "
        eq += amplitude_vec
        if len(sample_params.amplitudes) > 1:
            eq += r"^{T}"
        if samples_parameters.noise_scale != 0:
            noise_scale = str(sample_params.noise_scale)
            eq += r" + "
            eq += noise_scale
            eq += r" * \mathcal{N}(0,1)"
        eq += r"$"

        return eq

    fig, ax = plot if plot is not None else plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    legends: list[tuple[any, any]] = []
    x = [i for i in range(params.sequence_len)]
    # ax.set_title("Training data")
    alphas = [min(params.sequence_len / sample.size(dim=0), 1) for sample in samples]
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    cmap = prop_cycle.by_key()["color"]
    marker_line_color = "pink"  # cmap[7]
    marker_color = "lightgrey"  # cmap[7]
    for s_idx, (sample, sample_params, alpha) in enumerate(zip(samples, samples_parameters, alphas)):
        legends.append((Line2D([0], [0], color=cmap[s_idx], lw=6), create_sample_legend_string(s_idx, sample_params)))
        for sequence in sample:
            t_seq = torch.transpose(sequence, 0, 1)
            for feature in t_seq:
                # ax.plot(x, feature, color=cmap[s_idx], alpha=alpha, zorder=2)
                ax.step(
                    x, feature, where="mid", color=cmap[s_idx], alpha=alpha, zorder=2, rasterized=True
                )  # use rasterized here, else the generated pdf gets to complex

    for s_idx, sample in enumerate(samples):
        # [n*times, sequence, features] -> [sequence, n*times, features]
        t_samples = torch.transpose(sample, 0, 1)
        # [sequence, n*times, features] -> [sequence, mean(features)]
        t_sample_means = torch.mean(t_samples, dim=1)
        for f_idx, y in enumerate(torch.transpose(t_sample_means, 0, 1)):
            (lin,) = ax.plot(x, y, c=marker_line_color, lw=1.2, alpha=0.5, linestyle="dashed", zorder=3)
            # mark, = ax.plot(x, y, marker='_', alpha=1, markersize=12)
            (mark,) = ax.plot(
                x,
                y,
                linestyle="none",
                marker="_",
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                markersize=10,
                markeredgewidth=1,
                zorder=4,
            )
            # only create a single legend entry
            if s_idx == 0 and f_idx == 0:
                legends.append(((lin, mark), r"$E[X_{[t]_{s}}]$"))

    # ax.set_rasterization_zorder(0)
    x_labels = ["$" + str(i) + "$" for i in x]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("$[t]_{s}$", fontsize=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_ylabel(r"$X_{[t]_s} \sim \sin(\frac{2\pi t}{s})A + \epsilon_{t}$", fontsize=12)
    ax.legend(map(lambda e: e[0], legends), map(lambda e: e[1], legends))

    return fig, ax


def plot_box_plot_per_ts(
    data: Tensor, epoch: int, samples: list[Tensor], params: TrainParameters, condition: Optional[int] = None
) -> tuple[Figure, Axes]:
    data = data.cpu()
    samples = [sample.cpu() for sample in samples]
    sample_size = data.shape[0]
    features_len = data.shape[2]

    t_data = torch.transpose(data, 0, 1)  # [10, 24, 3]  -> [24, 10, 3]
    # [10, 24, 3] -> list_3([24, 10)
    t_data_single_feature = torch.unbind(t_data, 2)

    # list_n([times, sequence, features]) -> [n*times, sequence, features]
    conc_samples = torch.concat(samples, dim=0)
    # [n*times, sequence, features] -> [sequence, n*times, features]
    t_samples = torch.transpose(conc_samples, 0, 1)
    # [sequence, n*times, features] -> [sequence, mean(features)]
    t_sample_means = torch.mean(t_samples, dim=1)

    def create_y_label(f_idx) -> str:
        y_lbl = r"$"
        y_lbl += r"G_{"
        y_lbl += str(f_idx)
        y_lbl += r"}"
        y_lbl += r"(Z"
        if condition is not None:
            y_lbl += r"\mid C_{"
            y_lbl += str(condition)
            y_lbl += r"}"
        y_lbl += r"), Z \sim \mathcal{N}(0,1), Z \in \mathcal{R}^{"
        y_lbl += str(sample_size)
        y_lbl += r" \times "
        y_lbl += str(params.latent_vector_size)
        y_lbl += r"}"
        y_lbl += r"$"
        return y_lbl

    for feature_idx in range(features_len):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        x_labels = ["$" + str(i) + "$" for i in range(params.sequence_len)]
        # labels = ["$t_{" + str(i) + "}$" for i in range(params.sequence_len)]
        ax.plot(
            np.arange(params.sequence_len) + 1,
            torch.transpose(t_sample_means, 0, 1)[feature_idx],
            label=r"$E[X_{[t]_{s}}]$",
        )
        ax.boxplot(
            t_data_single_feature[feature_idx],
            labels=x_labels,
            bootstrap=5000,
            showmeans=True,
            meanline=True,
            notch=True,
        )

        ax.set_xlabel("$[t]_{s}$", fontsize=12)
        ax.set_ylabel(
            create_y_label(feature_idx),
            # r"$G_{" + str(feature_idx) + r"}(Z), Z \sim \mathcal{N}(0,1), \vert Z \vert=" + str(sample_size) + r"$",
            fontsize=12,
        )
        ax.legend()
        yield (fig, ax)
        # save_fig(fig, save_path / f"distribution_result_epoch_{epoch}_feature_{feature_idx}.png")


def plot_model_losses(g_losses: list[any], d_losses: list[any], current_epoch: int) -> tuple[Figure, Axes]:
    fig, ax_iter = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    # ax_iter.set_title("Generator und Diskriminator Loss")
    ax_iter.plot(g_losses, label=r"$L_{G}$")
    ax_iter.plot(d_losses, label=r"$L_{D}$")
    ax_iter.legend()
    ax_iter.set_xlabel(translate(SimpleGanPlotResultColumns.ITERATION))
    ax_iter.set_ylabel(translate(SimpleGanPlotResultColumns.LOSS))

    max_iterations = len(g_losses)
    max_epochs = current_epoch

    def iter2epoch(iter):
        return max_epochs * (iter / max_iterations)

    def epoch2iter(epoch):
        return max_iterations * (epoch / max_epochs)

    ax_epochs = ax_iter.secondary_xaxis("top", functions=(iter2epoch, epoch2iter))
    ax_epochs.set_xlabel(translate(SimpleGanPlotResultColumns.EPOCH))
    # ax_epochs.xaxis.set_major_locator(ticker.Autolocator())
    # ax_epochs.set_xlim(0, params.epochs)
    # ax_epochs.set_xbound(ax_iter.get_xbound())

    return fig, ax_iter
