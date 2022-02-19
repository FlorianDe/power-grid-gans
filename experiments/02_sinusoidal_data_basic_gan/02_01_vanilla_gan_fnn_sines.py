from dataclasses import dataclass
from enum import Enum
from functools import reduce
from pathlib import PurePath
from typing import Callable, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from tqdm import tqdm

import random
import math
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
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
    EPOCHS = "epochs"
    ITERATIONS = "iterations"


__PLOT_DICT: dict[SimpleGanPlotResultColumns, dict[Locale, str]] = {
    SimpleGanPlotResultColumns.LOSS: {Locale.EN: "Loss", Locale.DE: "Verlust"},
    SimpleGanPlotResultColumns.EPOCHS: {Locale.EN: "Epochs", Locale.DE: "Epochen"},
    SimpleGanPlotResultColumns.ITERATIONS: {Locale.EN: "Iterations", Locale.DE: "Iteration"},
}


def translate(key: SimpleGanPlotResultColumns) -> str:
    return __PLOT_DICT[key][PLOT_LANG]


@dataclass
class TrainParameters:
    epochs: int
    latent_vector_size: int = 100  # original paper \cite?
    sequence_len: int = 24
    batch_size: int = 8
    device: torch.device = torch.device("cpu")


def fnn_noise_generator(current_batch_size: int, params: TrainParameters) -> Tensor:
    return torch.randn(current_batch_size, params.latent_vector_size, device=params.device)


class DiscriminatorFNN(nn.Module):
    def __init__(self, features: int, sequence_len: int, out_features: int = 1):
        super(DiscriminatorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.dense1 = nn.Linear(
            in_features=sequence_len * features,
            out_features=out_features,
        )
        self.activation1 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation1(self.dense1(x))


class GeneratorFNN(nn.Module):
    def __init__(
        self,
        latent_vector_size: int,
        features: int,
        sequence_len: int,
    ):
        super(GeneratorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.dense1 = nn.Linear(
            in_features=latent_vector_size,
            out_features=features * sequence_len,
        )
        self.activation1 = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation1(self.dense1(x))
        # return self.dense1(x) # removed tanh


class PrintSize(nn.Module):
    def __init__(self):
        super(PrintSize, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def cnn_noise_generator(current_batch_size: int, params: TrainParameters) -> Tensor:
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


def generate_sine_features(
    sequence_len: int, amplitudes: list[float] = [1], times: int = 1, noise_scale: float = 0.05, seed: int = 42
) -> tuple[Tensor, Tensor]:
    """
    Returns a multi dimensional sine wave feature of shape [times, sequence_len, features]
    """
    torch.manual_seed(seed)

    features = len(amplitudes)
    a = torch.tensor(amplitudes).view(1, features)
    x = torch.linspace(0, sequence_len, sequence_len).view(sequence_len, 1).repeat(times, 1)
    sine = torch.sin((2 * math.pi / sequence_len) * x)
    scaled_sines = (sine * a).view(times, sequence_len, features)
    # noises = noise_scale * (2 * torch.rand(scaled_sines.shape) - torch.ones(scaled_sines.shape))
    noises = noise_scale * torch.randn(scaled_sines.shape)

    return scaled_sines + noises


def train(
    G: nn.Module,
    D: nn.Module,
    noise_generator: Callable[[TrainParameters], Tensor],
    samples: list[npt.ArrayLike],
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

    for i, sample in enumerate(samples):
        print(f"{i}. {sample.shape}")
    # # TODO FOR NOW JUST CONCAT THEM!
    flattened_samples = torch.concat(samples, dim=0)
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
            # TODO PREPARE DATA FOR SPECIFIC NETS
            # data_batch = data_batch.view(current_batch_size, -1) # FNN PREPARATION
            # CNN PREPARATION # TODO CHECK THIS
            data_batch = data_batch.view(current_batch_size, features_len, params.sequence_len)
            # data_batch = torch.transpose(data_batch, 1, 2)  # CNN PREPARATION # TODO CHECK THIS
            # print(f"{data_batch.shape=}")

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
            noise = noise_generator(current_batch_size, params)
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
                noise = noise_generator(generated_sample_count, params)
                generated_sine = G(noise)
                generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                fig, ax = plot_sample(generated_sine)
                save_fig(fig, save_path / f"{epoch}.png")

            with torch.no_grad():
                generated_sample_count = 100
                noise = noise_generator(generated_sample_count, params)
                generated_sine = G(noise)
                generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                save_box_plot_per_ts(
                    data=generated_sine, epoch=epoch, samples=samples, params=params, save_path=save_path
                )

    fig, ax = plot_model_losses(G_losses, D_losses, params)
    save_fig(fig, save_path / f"model_losses_after_{params.epochs}.png")
    plt.close(fig)


def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def plot_sample(sample: Tensor) -> tuple[Axes, Figure]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    unbind_sample = torch.unbind(sample)
    flattened_sample = torch.concat(unbind_sample)
    for i, y in enumerate(torch.transpose(flattened_sample, 0, 1)):
        ax.plot(range(len(y)), y)
    return fig, ax


def plot_train_data_overlayed(samples: list[Tensor], params: TrainParameters, features_len: int) -> tuple[Axes, Figure]:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    legends: list[tuple[any, any]] = []
    x = [i for i in range(params.sequence_len)]
    ax.set_title("Training data")
    sequence_count = sum(sample.size(dim=1) for sample in samples)
    alpha = max(1/sequence_count, 1/255)
    cmap = plt.cm.viridis
    for i, sample in enumerate(samples):
        legends.append(
            (
                Line2D([0], [0], color=cmap(i), lw=6), 
                r"$0.5*\sin(\frac{2\pi t}{" + str(params.sequence_len) + r"}) + 0.01 * \mathcal{N}(0,1)$" # TODO
            )
        )
        for sequence in sample:
            t_seq = torch.transpose(sequence, 0, 1)
            for feature in t_seq:
                # ax.plot(x, feature, color='b', alpha=alpha)
                ax.step(x, feature, where='mid', color=cmap(i), alpha=alpha, zorder=2)
    # ax.legend()

    # list_n([times, sequence, features]) -> [n*times, sequence, features]
    conc_samples = torch.concat(samples, dim=0)
    # [n*times, sequence, features] -> [sequence, n*times, features]
    t_samples = torch.transpose(conc_samples, 0, 1)
    # [sequence, n*times, features] -> [sequence, mean(features)]
    t_sample_means = torch.mean(t_samples, dim=1)
    for i, y in enumerate(torch.transpose(t_sample_means, 0, 1)):
        lin, = ax.plot(x, y, c="grey", lw=2, alpha=.7, linestyle='dashed', zorder=3)
        # mark, = ax.plot(x, y, marker='_', alpha=1, markersize=12)
        # plt.scatter(x, y, s=500, c="grey", alpha=1, marker='_', zorder=4)
        mark, = ax.plot(x, y, color='orange', linestyle='none', marker='_', markerfacecolor='orange', markersize=10, markeredgewidth=2, zorder=4)
        legends.append(((lin, mark), f'mean feature {i}'))

    x_labels = ["$t_{" + str(i) + "}$" for i in x]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(            
        r"$\sin(\frac{2\pi t}{" + str(params.sequence_len) + r"}) + \alpha * \mathcal{N}(0,1)$", fontsize=12
    )

    ax.grid(axis='x', color='0.95')

    ax.legend(map(lambda e: e[0], legends), map(lambda e: e[1], legends))
    

    return fig, ax


def plot_model_losses(g_losses: list[any], d_losses: list[any], params: TrainParameters) -> tuple[Axes, Figure]:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    ax.set_title("Generator and Discriminator Loss During Training")
    ax.plot(g_losses, label=r"$L_{G}$")
    ax.plot(d_losses, label=r"$L_{D}$")
    ax.legend()
    ax.set_xlabel(translate(SimpleGanPlotResultColumns.ITERATIONS))
    ax.set_ylabel(translate(SimpleGanPlotResultColumns.LOSS))
    # exp = lambda x: 10**(x)
    # log = lambda x: np.log(x)

    # # Set x scale to exponential
    # ax.set_xscale('function', functions=(exp, log))
    # ax.set_xscale('log')

    # ax2 = ax.twiny()
    # ax2 = ax.secondary_xaxis('top')
    # ax2.set_xlabel(translate(SimpleGanPlotResultColumns.EPOCHS))

    # ax2.set_xlim(0, params.epochs)

    # epoch_ticks =
    # ax2.set_xticks([10, 30, 40])
    # ax2.set_xticklabels(['7','8','99'])
    # ax2.set_xscale('log')
    # ax2.set_xticks([10, 30, 40])
    # ax2.set_xticklabels(['7','8','99'])

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
        labels = ["$t_{" + str(i) + "}$" for i in range(params.sequence_len)]
        ax.plot(np.arange(params.sequence_len) + 1, torch.transpose(t_sample_means, 0, 1)[feature_idx])
        ax.boxplot(
            t_data_single_feature[feature_idx], labels=labels, bootstrap=5000, showmeans=True, meanline=True, notch=True
        )

        ax.set_xlabel("t", fontsize=12)
        ax.set_ylabel(
            r"$G_{t, " + str(feature_idx) + r"}(Z), Z \sim \mathcal{N}(0,1), \vert Z \vert=" + str(sample_size) + r"$",
            fontsize=12,
        )

        save_fig(fig, save_path / f"distribution_result_epoch_{epoch}_feature_{feature_idx}.png")


def setup_fnn_models_and_train(params: TrainParameters, samples: list[Tensor], features_len: int, save_path: PurePath):

    G = GeneratorFNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
    )
    D = DiscriminatorFNN(features=features_len, sequence_len=params.sequence_len, out_features=1)

    train(
        G=G,
        D=D,
        noise_generator=fnn_noise_generator,
        samples=samples,
        params=params,
        features_len=features_len,
        save_path=save_path,
    )
    return D, G


def train_fnn_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should give us a baseline for the simplest training possible
    """
    fnn_single_sample_univariate = save_images_path / "fnn_single_sample_univariate"
    fnn_single_sample_univariate.mkdir(parents=True, exist_ok=True)
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples: list[Tensor] = [
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0
        ),
    ]

    setup_fnn_models_and_train(params, samples, features_len, fnn_single_sample_univariate)


def train_fnn_noisy_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    Check how it affects the training if we add some noise on top of the training data
    """
    fnn_noisy_single_sample_univariate = save_images_path / "noisy_single_sample_univariate"
    fnn_noisy_single_sample_univariate.mkdir(parents=True, exist_ok=True)
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples: list[Tensor] = [
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.05
        ),
    ]

    setup_fnn_models_and_train(params, samples, features_len, fnn_noisy_single_sample_univariate)


def train_fnn_single_sample_multivariate(params: TrainParameters, sample_batches: int):
    """
    This should show us that the convergence is not as fast for multiple features!
    """
    fnn_single_sample_multivariate = save_images_path / "fnn_single_sample_multivariate"
    fnn_single_sample_multivariate.mkdir(parents=True, exist_ok=True)
    amplitudes = [0.5, 1.0]
    features_len = len(amplitudes)
    samples: list[Tensor] = [
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0
        ),
    ]

    setup_fnn_models_and_train(params, samples, features_len, fnn_single_sample_multivariate)


def train_fnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should result in a mode collapse
    """
    fnn_multiple_sample_univariate = save_images_path / "fnn_multiple_sample_univariate"
    fnn_multiple_sample_univariate.mkdir(parents=True, exist_ok=True)
    features_len = 1
    samples: list[Tensor] = [
        generate_sine_features(sequence_len=params.sequence_len, amplitudes=[1], times=sample_batches, noise_scale=0),
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=[0.75], times=sample_batches, noise_scale=0
        ),
        generate_sine_features(sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0),
    ]

    setup_fnn_models_and_train(params, samples, features_len, fnn_multiple_sample_univariate)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def setup_cnn_models_and_train(params: TrainParameters, samples: list[Tensor], features_len: int, save_path: PurePath):
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
        noise_generator=cnn_noise_generator,
        samples=samples,
        params=params,
        features_len=features_len,
        save_path=save_path,
    )
    return G


def train_cnn_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should give us a baseline for the simplest training possible
    """
    cnn_single_sample_univariate = save_images_path / "cnn_single_sample_univariate"
    cnn_single_sample_univariate.mkdir(parents=True, exist_ok=True)
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples: list[Tensor] = [
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.01
        )
    ]
    fig, ax = plot_train_data_overlayed(samples, params, features_len)
    save_fig(fig, cnn_single_sample_univariate / "train_data_plot")

    return setup_cnn_models_and_train(params, samples, features_len, cnn_single_sample_univariate)


def train_cnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should give us a baseline for the simplest training possible
    """
    cnn_multiple_sample_univariate = save_images_path / "cnn_multiple_sample_univariate"
    cnn_multiple_sample_univariate.mkdir(parents=True, exist_ok=True)
    # amplitudes = [1.0]
    features_len = 1
    samples: list[Tensor] = [
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=[1], times=sample_batches, noise_scale=0.01
        ),
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=[0.75], times=sample_batches, noise_scale=0.01
        ),
        generate_sine_features(
            sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0.01
        ),
    ]
    fig, ax = plot_train_data_overlayed(samples, params, features_len)
    save_fig(fig, cnn_multiple_sample_univariate / "train_data_plot")

    return setup_cnn_models_and_train(params, samples, features_len, cnn_multiple_sample_univariate)


if __name__ == "__main__":
    manualSeed = 1337
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    # rng = np.random.default_rng(seed=0) # use for numpy
    torch.manual_seed(manualSeed)
    set_latex_plot_params()
    train_params = TrainParameters(epochs=50)
    sample_batches = train_params.batch_size * 500

    # train_fnn_single_sample_univariate(train_params, sample_batches)
    # train_fnn_noisy_single_sample_univariate(train_params, sample_batches)
    # train_fnn_single_sample_multivariate(train_params, sample_batches)
    # train_fnn_multiple_sample_univariate(train_params, sample_batches)

    train_cnn_single_sample_univariate(train_params, sample_batches)
    train_cnn_multiple_sample_univariate(train_params, sample_batches)
