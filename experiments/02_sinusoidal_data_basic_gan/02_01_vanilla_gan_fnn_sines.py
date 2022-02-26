from configparser import NoSectionError
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from pathlib import PurePath
from typing import Callable, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib import ticker
from tqdm import tqdm
import seaborn as sns

import random
import math
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, dropout
from torch.utils.data import DataLoader

from experiments.utils import get_experiments_folder, set_latex_plot_params
from src.plots.typing import Locale

PLOT_LANG = Locale.DE
save_images_path = (
    get_experiments_folder().joinpath("02_sinusoidal_data_basic_gan").joinpath("02_01_vanilla_gan_fnn_sines")
)
save_images_path.mkdir(parents=True, exist_ok=True)


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

@dataclass
class SineGenerationParameters:
    sequence_len: int
    amplitudes: list[float] = field(default_factory=lambda: [1])
    times: int = 1
    noise_scale: float = 0.05

    def __iter__(self):
        return iter((self.sequence_len, self.amplitudes, self.times, self.noise_scale))

@dataclass
class TrainParameters:
    epochs: int
    latent_vector_size: int = 100  # original paper \cite?
    sequence_len: int = 24
    batch_size: int = 8
    device: torch.device = torch.device("cpu")


def fnn_batch_reshaper(data_batch: Tensor, batch_size: int, sequence_len: int, features_len: int) -> Tensor:
    return data_batch.view(batch_size, sequence_len*features_len)

def fnn_noise_generator(current_batch_size: int, params: TrainParameters, features_len: int) -> Tensor:
    return torch.randn(current_batch_size, params.latent_vector_size, device=params.device)


class DiscriminatorFNN(nn.Module):
    def __init__(self, features: int, sequence_len: int, out_features: int = 1, dropout: float = 0.5):
        super(DiscriminatorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        input_size = sequence_len * features
        negative_slope = 1e-2
        self.fnn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, 2*input_size),
            nn.LeakyReLU(negative_slope, inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(2*input_size, 4*input_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            # nn.Linear(4*input_size, 6*input_size),
            # nn.LeakyReLU(negative_slope, inplace=True),
            # nn.Linear(6*input_size, 4*input_size),
            # nn.LeakyReLU(negative_slope, inplace=True),
            # nn.Linear(4*input_size, 2*input_size),
            # nn.LeakyReLU(negative_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4*input_size, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.fnn(x))


class GeneratorFNN(nn.Module):
    def __init__(
        self,
        latent_vector_size: int,
        features: int,
        sequence_len: int,
        dropout: float = 0.5
    ):
        super(GeneratorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        negative_slope = 1e-2
        self.fnn = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(latent_vector_size, 2*latent_vector_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2*latent_vector_size, 4*latent_vector_size),
            nn.LeakyReLU(negative_slope, inplace=True),

            # nn.Linear(4*latent_vector_size, 6*latent_vector_size),
            # nn.LeakyReLU(negative_slope, inplace=True),
            # nn.Linear(6*latent_vector_size, 4*latent_vector_size),
            # nn.LeakyReLU(negative_slope, inplace=True),
            # nn.Linear(4*latent_vector_size, 2*latent_vector_size),
            # nn.LeakyReLU(negative_slope, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4*latent_vector_size, features * sequence_len),
        )
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.tanh(self.fnn(x))
        # return self.fnn(x) # removed tanh


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

def cnn_batch_reshaper(data_batch: Tensor, batch_size: int, sequence_len: int, features_len: int) -> Tensor:
    # data_batch = torch.transpose(data_batch, 1, 2)  # CNN PREPARATION # CNN BROKEN!
    return data_batch.view(batch_size, features_len, sequence_len)

def cnn_noise_generator(current_batch_size: int, params: TrainParameters, features_len: int) -> Tensor:
    return torch.randn(current_batch_size, params.latent_vector_size, 1, device=params.device)


class DiscriminatorCNN(nn.Module):
    def __init__(self, features: int, sequence_len: int, out_features: int = 1):
        super(DiscriminatorCNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len

        nc = self.features
        ndf = self.sequence_len
        self.main = nn.Sequential(
            # input is (nc) x 24
            nn.Conv1d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintSize(),
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintSize(),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintSize(),
            nn.Conv1d(ndf * 4, ndf * 8, 2, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # PrintSize(),
            nn.Conv1d(ndf * 8, out_features, 2, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class GeneratorCNN(nn.Module):
    def __init__(
        self,
        latent_vector_size: int,
        features: int,
        sequence_len: int,
    ):
        super(GeneratorCNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        # L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        nz = latent_vector_size
        nc = self.features
        ngf = self.sequence_len

        self.main = nn.Sequential(
            # PrintSize(),
            nn.ConvTranspose1d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            # PrintSize(),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 2, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # PrintSize(),
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 2, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # PrintSize(),
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 3, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # PrintSize(),
            nn.ConvTranspose1d(ngf, nc, 4, 2, 5, bias=False),
            nn.Tanh(),
            # PrintSize(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)

def rnn_batch_reshaper(data_batch: Tensor, batch_size: int, sequence_len: int, features_len: int) -> Tensor:
    return data_batch.view(batch_size, sequence_len, features_len)

def rnn_noise_generator(current_batch_size: int, params: TrainParameters, features_len: int) -> Tensor:
    return torch.randn(current_batch_size, features_len, params.latent_vector_size, device=params.device)

class DiscriminatorRNN(nn.Module):
    def __init__(
        self,
        features: int, 
        sequence_len: int, 
        out_features: int = 1
    ):
        super(DiscriminatorRNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.out_features = out_features
        self.hidden_size = 2*out_features # hardcoded
        self.num_layers = 1
        self.rnn = nn.GRU( 
            input_size = self.features,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bias = True,
            batch_first = True,
            dropout=0.5
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, self.out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, h = self.rnn(x, h0)
        out = out[:, -1, :]
        # out_last = out[:,-1]
        out = self.fc(self.relu(out))
        out = self.sigmoid(out)
        return out

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()  #.to(device)
    #     return hidden

class GeneratorRNN(nn.Module):
    def __init__(
        self,
        latent_vector_size: int,
        features: int,
        sequence_len: int,
    ):
        super(GeneratorRNN, self).__init__()
        self.latent_vector_size = latent_vector_size
        self.features = features
        self.sequence_len = sequence_len
        self.num_layers = self.features
        self.hidden_size = 2*self.sequence_len# hardcoded

        self.rnn = nn.GRU(
            input_size = self.latent_vector_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bias = True,
            batch_first = True,
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, self.sequence_len)
        self.tanh = nn.Tanh()

    def forward(self, z):
        h0 = torch.zeros(self.num_layers, z.size(0), self.hidden_size).requires_grad_()
        out, h = self.rnn(z, h0)
        out = self.fc(self.relu(out))
        out = self.tanh(out)
        out = out.view(-1, self.sequence_len, self.features)

        return out


def generate_sine_features(
    params: SineGenerationParameters, seed: int = 42
) -> tuple[Tensor, Tensor]:
    """
    Returns a multi dimensional sine wave feature of shape [times, sequence_len, features]
    """
    sequence_len, amplitudes, times, noise_scale = params
    torch.manual_seed(seed)

    features = len(amplitudes)
    a = torch.tensor(amplitudes).view(1, features)
    # x = torch.linspace(0, sequence_len, sequence_len).view(sequence_len, 1).repeat(times, 1)
    x = torch.arange(sequence_len).view(sequence_len, 1).repeat(times, 1)
    sine = torch.sin((2 * math.pi / sequence_len) * x)
    scaled_sines = (sine * a).view(times, sequence_len, features)
    # noises = noise_scale * (2 * torch.rand(scaled_sines.shape) - torch.ones(scaled_sines.shape))   # scaled Uniform dist noise
    noises = noise_scale * torch.randn(scaled_sines.shape)  # scaled Normal dist noise

    return scaled_sines + noises


def train(
    G: nn.Module,
    D: nn.Module,
    batch_reshaper: Callable[[Tensor], Tensor],
    noise_generator: Callable[[int, TrainParameters, int], Tensor],
    samples_parameters: list[SineGenerationParameters],
    params: TrainParameters,
    features_len: int,
    save_path: PurePath,
):
    # Learning rate for optimizers
    lr = 1e-3
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.9

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # generate sample data
    samples = [generate_sine_features(params) for params in samples_parameters]

    # create train test directory
    save_path.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_train_data_overlayed(samples, samples_parameters, params)
    save_fig(fig, save_path / "train_data_plot")

    print(f"Preparing training data for: {save_path.name}")
    print(f"Start training with samples:")
    for i, (sample, sample_params) in enumerate(zip(samples, samples_parameters)):
        print(f"{i}. sample s{sample.shape} with params: {sample_params}")
    flattened_samples = torch.concat(samples, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
    print(f"{flattened_samples.shape=}")
    dataloader = DataLoader(
        flattened_samples,
        batch_size=params.batch_size,
        shuffle=False,
        # num_workers=workers
    )

    G_losses = []
    D_losses = []
    iters = 0

    for epoch in (
        progress := tqdm(range(params.epochs), unit="epochs", bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}")
    ):
        for batch_idx, (data_batch) in enumerate(dataloader):
            current_batch_size = min(params.batch_size, data_batch.shape[0])
            data_batch = batch_reshaper(data_batch, current_batch_size, params.sequence_len, features_len)
            # data_batch = data_batch.view(current_batch_size, -1) # FNN PREPARATION

            # data_batch = data_batch.view(current_batch_size, features_len, params.sequence_len) # CNN PREPARATION
            # data_batch = torch.transpose(data_batch, 1, 2)  # CNN PREPARATION # CNN BROKEN!

            # data_batch = data_batch.view(current_batch_size, params.sequence_len, features_len) # RNN PREPARATION


            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            real_labels = torch.ones(current_batch_size, requires_grad=False, device=params.device)
            fake_labels = torch.zeros(current_batch_size, requires_grad=False, device=params.device)

            ## Train with all-real batch
            D.zero_grad()
            # label = torch.full((current_batch_size), real_label_value, dtype=torch.float, device=params.device)
            d_out_real = D(data_batch).view(-1)
            # print(f"{d_out_real.shape=}")
            d_err_real = criterion(d_out_real, real_labels)
            d_err_real.backward()
            D_x = d_err_real.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = noise_generator(current_batch_size, params, features_len)
            fake_generated = G(noise)
            # print(f"{fake_generated.shape=}")
            d_out_fake = D(fake_generated.detach()).view(-1)
            d_err_fake = criterion(d_out_fake, fake_labels)
            d_err_fake.backward()
            D_G_z1 = d_out_fake.mean().item()
            err_D = d_err_real + d_err_fake
            # Update weights of D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            d_out_fake = D(fake_generated).view(-1)
            err_G = criterion(d_out_fake, real_labels)
            err_G.backward()
            D_G_z2 = d_out_fake.mean().item()
            optimizerG.step()

            if iters % 20 == 0:
                # padded_epoch = str(epoch).ljust(len(str(params.epochs)))
                # padded_batch_idx = str(batch_idx).ljust(len(str(len(dataloader))))
                progress_str = "Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f" % (
                    err_D.item(),
                    err_G.item(),
                    D_x,
                    D_G_z1,
                    D_G_z2,
                )
                progress.set_description(progress_str)

            # Save Losses for plotting later
            G_losses.append(err_G.item())
            D_losses.append(err_D.item())

            iters += 1

        if epoch % 2 == 0:
            with torch.no_grad():
                generated_sample_count = 7
                noise = noise_generator(generated_sample_count, params, features_len)
                generated_sine = G(noise)
                generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
                fig, ax = plot_sample(generated_sine, (fig, ax))
                save_fig(fig, save_path / f"{epoch}.png")

            with torch.no_grad():
                generated_sample_count = 100
                noise = noise_generator(generated_sample_count, params, features_len)
                generated_sine = G(noise)
                generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                save_box_plot_per_ts(
                    data=generated_sine, epoch=epoch, samples=samples, params=params, save_path=save_path
                )

    fig, ax = plot_model_losses(G_losses, D_losses, params)
    save_fig(fig, save_path / f"model_losses_after_{params.epochs}.png")
    plt.close(fig)
    print("End training\n--------------------------------------------")

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


def setup_fnn_models_and_train(params: TrainParameters, samples_parameters: list[SineGenerationParameters], features_len: int, save_path: PurePath, dropout: float = 0.5):
    G = GeneratorFNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
        dropout=dropout
    )
    D = DiscriminatorFNN(features=features_len, sequence_len=params.sequence_len, out_features=1, dropout=dropout)
    train(
        G=G,
        D=D,
        batch_reshaper=fnn_batch_reshaper,
        noise_generator=fnn_noise_generator,
        samples_parameters=samples_parameters,
        params=params,
        features_len=features_len,
        save_path=save_path,
    )
    return D, G


def train_fnn_single_sample_univariate_no_regularization(params: TrainParameters, sample_batches: int):
    """
    This should give us a baseline for the simplest training possible
    """
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0
        ),
    ]

    setup_fnn_models_and_train(params, samples_parameters, features_len, save_images_path / "fnn_single_sample_univariate_no_regularization", 0)


def train_fnn_noisy_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    Check how it affects the training if we add some noise on top of the training data
    """
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.01
        ),
    ]
    setup_fnn_models_and_train(params, samples_parameters, features_len, save_images_path / "fnn_noisy_single_sample_univariate")


def train_fnn_single_sample_multivariate(params: TrainParameters, sample_batches: int):
    """
    This should show us that the convergence is not as fast for multiple features!
    """
    amplitudes = [0.5, 1.0]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.01)
    ]
    setup_fnn_models_and_train(params, samples_parameters, features_len, save_images_path / "fnn_single_sample_multivariate")


def train_fnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should result in a mode collapse
    """
    features_len = 1
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[1], times=sample_batches, noise_scale=0.01),
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.75], times=sample_batches, noise_scale=0.01
        ),
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0.01),
    ]
    setup_fnn_models_and_train(params, samples_parameters, features_len, save_images_path / "fnn_multiple_sample_univariate")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def setup_cnn_models_and_train(params: TrainParameters, samples_parameters: list[SineGenerationParameters], features_len: int, save_path: PurePath):
    G = GeneratorCNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
    )
    G.apply(weights_init)

    D = DiscriminatorCNN(features=features_len, sequence_len=params.sequence_len, out_features=1)
    D.apply(weights_init)

    train(
        G=G,
        D=D,
        batch_reshaper=cnn_batch_reshaper,
        noise_generator=cnn_noise_generator,
        samples_parameters=samples_parameters,
        params=params,
        features_len=features_len,
        save_path=save_path,
    )
    return G


def train_cnn_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    XXXX
    """
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.01)
    ]
    return setup_cnn_models_and_train(params, samples_parameters, features_len, save_images_path / "cnn_single_sample_univariate")


def train_cnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    XXXX
    """
    features_len = 1
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[1], times=sample_batches, noise_scale=0.01),
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0.01),
    ]
    return setup_cnn_models_and_train(params, samples_parameters, features_len, save_images_path / "cnn_multiple_sample_univariate")

def train_cnn_single_sample_multivariate(params: TrainParameters, sample_batches: int):
    """
    XXXX
    """
    features_len = 2
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[1, 0.5], times=sample_batches, noise_scale=0.01),
    ]
    return setup_cnn_models_and_train(params, samples_parameters, features_len, save_images_path / "cnn_single_sample_multivariate")

def setup_rnn_models_and_train(params: TrainParameters, samples_parameters: list[SineGenerationParameters], features_len: int, save_path: PurePath):
    G = GeneratorRNN(
        latent_vector_size=train_params.latent_vector_size,
        features=features_len,
        sequence_len=train_params.sequence_len
    )

    D = DiscriminatorRNN(
        features=features_len,
        out_features=1,
        sequence_len=train_params.sequence_len
    )

    train(
        G=G,
        D=D,
        batch_reshaper=rnn_batch_reshaper,
        noise_generator=rnn_noise_generator,
        samples_parameters=samples_parameters,
        params=params,
        features_len=features_len,
        save_path=save_path,
    )
    return G


def train_rnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    features_len = 1
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0.01),
        # SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[1, 0.5, 0.25], times=sample_batches, noise_scale=0.01)
    ]

    return setup_rnn_models_and_train(params, samples_parameters, features_len, save_images_path / "rnn_multiple_sample_univariate")

def save_multi_sample_multivariate_training_data_sample_overview(params: TrainParameters):
    times = 500
    save_path = save_images_path
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[1], times=times, noise_scale=0.05),
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[2, 0.5], times=times, noise_scale=0.01)
    ]
    # generate sample data
    samples = [generate_sine_features(params) for params in samples_parameters]

    # create train test directory
    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_train_data_overlayed(samples, samples_parameters, params)
    save_fig(fig, save_path / "train_data_plot.pdf")
    

if __name__ == "__main__":
    sns.set_theme()
    sns.set_context("paper")
    # sns.set_palette("colorblind")
    set_latex_plot_params()

    manualSeed = 1337
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    # rng = np.random.default_rng(seed=0) # use for numpy
    torch.manual_seed(manualSeed)

    train_params = TrainParameters(epochs=50)
    sample_batches = train_params.batch_size * 100

    # save sample image for synthetic test data
    save_multi_sample_multivariate_training_data_sample_overview(train_params)
    
    # FNN trainings
    train_fnn_single_sample_univariate_no_regularization(TrainParameters(epochs=200), sample_batches)
    # train_fnn_noisy_single_sample_univariate(train_params, sample_batches)
    # train_fnn_single_sample_multivariate(train_params, sample_batches)
    # train_fnn_multiple_sample_univariate(train_params, sample_batches)

    # CNN trainings
    # train_cnn_single_sample_univariate(train_params, sample_batches)
    # train_cnn_single_sample_multivariate(train_params, sample_batches)
    # train_cnn_multiple_sample_univariate(train_params, sample_batches)

    # RNN
    # train_rnn_multiple_sample_univariate(train_params, sample_batches)
