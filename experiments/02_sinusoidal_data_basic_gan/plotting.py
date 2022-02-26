from enum import Enum
from pathlib import PurePath
from typing import Optional

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


def plot_sample(sample: Tensor, plot: tuple[Figure, Axes]) -> tuple[Figure, Axes]:
    fig, ax = plot if plot is not None else plt.subplots(nrows=1, ncols=1)
    unbind_sample = torch.unbind(sample)
    flattened_sample = torch.concat(unbind_sample)
    for i, y in enumerate(torch.transpose(flattened_sample, 0, 1)):
        ax.plot(range(len(y)), y)
    return fig, ax


def plot_train_data_overlayed(samples: list[Tensor], samples_parameters: list[SineGenerationParameters], params: TrainParameters, plot: Optional[tuple[Figure, Axes]] = None) -> tuple[Figure, Axes]:
    if len(samples) != len(samples_parameters):
        raise ValueError("The specified samples and sample parameters have to have the same length.")

    def create_sample_legend_string(idx: int, samples_parameters: SineGenerationParameters):
        amplitude_vec = str(sample_params.amplitudes)
        sequence_len = str(params.sequence_len)

        random_var_name = chr(65+((idx+23) % 26)) # 0 => X, 1 => Y ...
        eq = r"$"
        eq += random_var_name
        eq += r"_{[t]_{"
        eq += sequence_len
        eq += r"}} \sim "
        eq += amplitude_vec
        if len(sample_params.amplitudes) > 1:
            eq += r"^{T}"
        eq += r"*\sin(\frac{2\pi t}{"
        eq += sequence_len
        eq += r"})"
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
    alphas = [min(params.sequence_len/sample.size(dim=0), 1) for sample in samples]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    cmap = prop_cycle.by_key()['color']
    marker_line_color = "pink" # cmap[7]
    marker_color = "lightgrey" # cmap[7]
    for s_idx, (sample, sample_params, alpha) in enumerate(zip(samples, samples_parameters, alphas)):
        legends.append((Line2D([0], [0], color=cmap[s_idx], lw=6), create_sample_legend_string(s_idx, sample_params)))
        for sequence in sample:
            t_seq = torch.transpose(sequence, 0, 1)
            for feature in t_seq:
                # ax.plot(x, feature, color=cmap[s_idx], alpha=alpha, zorder=2)
                ax.step(x, feature, where='mid', color=cmap[s_idx], alpha=alpha, zorder=2, rasterized=True)  # use rasterized here, else the generated pdf gets to complex

    for s_idx, sample in enumerate(samples):
        # [n*times, sequence, features] -> [sequence, n*times, features]
        t_samples = torch.transpose(sample, 0, 1)
        # [sequence, n*times, features] -> [sequence, mean(features)]
        t_sample_means = torch.mean(t_samples, dim=1)
        for f_idx, y in enumerate(torch.transpose(t_sample_means, 0, 1)):
            lin, = ax.plot(x, y, c=marker_line_color, lw=1.2, alpha=.5, linestyle='dashed', zorder=3)
            # mark, = ax.plot(x, y, marker='_', alpha=1, markersize=12)
            mark, = ax.plot(x, y, linestyle='none', marker='_', markerfacecolor=marker_color, markeredgecolor=marker_color, markersize=10, markeredgewidth=1, zorder=4)
            # only create a single legend entry
            if s_idx == 0 and f_idx == 0:
                legends.append(((lin, mark), r'$E[X_{[t]_{s}}]$'))

    # ax.set_rasterization_zorder(0)
    x_labels = ["$" + str(i) + "$" for i in x]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("$[t]_{s}$", fontsize=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(10))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.set_ylabel(            
        r"$X_{[t]_s} \sim \overrightarrow{a} * \sin(\frac{2\pi t}{s}) + \nu * \mathcal{N}(0,1)$", fontsize=12
    )
    ax.legend(map(lambda e: e[0], legends), map(lambda e: e[1], legends))


    return fig, ax

def save_box_plot_per_ts(data: Tensor, epoch: int, samples: list[Tensor], params: TrainParameters, save_path: PurePath):
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

    for feature_idx in range(features_len):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        x_labels = ["$" + str(i) + "$" for i in range(params.sequence_len)]
        # labels = ["$t_{" + str(i) + "}$" for i in range(params.sequence_len)]
        ax.plot(np.arange(params.sequence_len) + 1, torch.transpose(t_sample_means, 0, 1)[feature_idx], label=r'$E[X_{[t]_{s}}]$')
        ax.boxplot(
            t_data_single_feature[feature_idx], labels=x_labels, bootstrap=5000, showmeans=True, meanline=True, notch=True
        )

        ax.set_xlabel("$[t]_{s}$", fontsize=12)
        ax.set_ylabel(
            r"$G_{t, " + str(feature_idx) + r"}(Z), Z \sim \mathcal{N}(0,1), \vert Z \vert=" + str(sample_size) + r"$",
            fontsize=12,
        )
        ax.legend()

        save_fig(fig, save_path / f"distribution_result_epoch_{epoch}_feature_{feature_idx}.png")

def plot_model_losses(g_losses: list[any], d_losses: list[any], params: TrainParameters) -> tuple[Figure, Axes]:
    fig, ax_iter = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    # ax_iter.set_title("Generator und Diskriminator Loss")
    ax_iter.plot(g_losses, label=r"$L_{G}$")
    ax_iter.plot(d_losses, label=r"$L_{D}$")
    ax_iter.legend()
    ax_iter.set_xlabel(translate(SimpleGanPlotResultColumns.ITERATION))
    ax_iter.set_ylabel(translate(SimpleGanPlotResultColumns.LOSS))

    max_iterations = len(g_losses)
    max_epochs = params.epochs

    def iter2epoch(iter):
        return max_epochs*(iter/max_iterations)

    def epoch2iter(epoch):
        return max_iterations*(epoch/max_epochs)

    ax_epochs = ax_iter.secondary_xaxis("top", functions=(iter2epoch, epoch2iter))
    ax_epochs.set_xlabel(translate(SimpleGanPlotResultColumns.EPOCH))
    # ax_epochs.xaxis.set_major_locator(ticker.Autolocator())
    # ax_epochs.set_xlim(0, params.epochs)
    # ax_epochs.set_xbound(ax_iter.get_xbound())

    return fig, ax_iter