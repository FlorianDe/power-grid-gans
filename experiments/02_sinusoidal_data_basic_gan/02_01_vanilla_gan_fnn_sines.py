from dataclasses import dataclass
from enum import Enum
from pathlib import PurePath
from tqdm import tqdm

from functools import reduce

import math
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from experiments.utils import get_experiments_folder, set_latex_plot_params
from src.plots.typing import Locale

PLOT_LANG = Locale.DE
save_images_path = get_experiments_folder().joinpath(
    "02_sinusoidal_data_basic_gan").joinpath("02_01_vanilla_gan_fnn_sines")
save_images_path.mkdir(parents=True, exist_ok=True)

class SimpleGanPlotResultColumns(Enum):
    LOSS = "loss"
    EPOCHS = "epochs"
    ITERATIONS = "iterations"

__PLOT_DICT: dict[SimpleGanPlotResultColumns, dict[Locale, str]] = {
    SimpleGanPlotResultColumns.LOSS: {
        Locale.EN: "Loss",
        Locale.DE: "Verlust"
    },    
    SimpleGanPlotResultColumns.EPOCHS: {
        Locale.EN: "Epochs",
        Locale.DE: "Epochen"
    },
    SimpleGanPlotResultColumns.ITERATIONS: {
        Locale.EN: "Iterations",
        Locale.DE: "Iteration"
    },
}
def translate(key: SimpleGanPlotResultColumns) -> str:
    return __PLOT_DICT[key][PLOT_LANG]


@dataclass
class TrainParameters:
    epochs: int
    latent_vector_size: int = 128  # original paper \cite?
    sequence_len: int = 24
    batch_size: int = 8
    device: torch.device = torch.device("cpu")


class DiscriminatorFNN(nn.Module):
    def __init__(
        self,
        features: int,
        sequence_len: int,
        out_features: int = 1
    ):
        super(DiscriminatorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.dense1 = nn.Linear(
            in_features=sequence_len*features,
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
            out_features=features*sequence_len,
        )
        self.activation1 = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation1(self.dense1(x))
        # return self.dense1(x) # removed tanh


def generate_sine_features(sequence_len: int, amplitudes: list[float] = [1], times: int = 1, noise_scale: float = 0.05, seed: int = 42) -> tuple[Tensor, Tensor]:
    """
        Returns a multi dimensional sine wave feature of shape [times, sequence_len, features]
    """
    torch.manual_seed(seed)

    size = times*sequence_len
    features = len(amplitudes)
    a = torch.tensor(amplitudes).view(1, features)
    x = torch.linspace(0, size, size).view(size, 1)
    sine = torch.sin((2*math.pi/sequence_len)*x)
    scaled_sines = (sine*a).view(times, sequence_len, features)
    noises = noise_scale*torch.rand(scaled_sines.shape)

    return (scaled_sines+noises)


def train(
    G: nn.Module,
    D: nn.Module,
    samples: list[npt.ArrayLike],
    params: TrainParameters,
    features_len: int,
    save_path: PurePath
):
    # Learning rate for optimizers
    lr = 5e-4
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.9

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # # TODO FOR NOW JUST CONCAT THEM!
    flattened_samples = torch.concat(samples, dim=0)
    dataloader = DataLoader(
        flattened_samples,
        batch_size=params.batch_size,
        shuffle=False,
        # num_workers=workers
    )

    G_losses = []
    D_losses = []
    iters = 0

    for epoch in (progress := tqdm(range(params.epochs), unit="epochs", bar_format='{desc}{percentage:3.0f}%|{bar:10}{r_bar}')):
        for batch_idx, (data_batch) in enumerate(dataloader):
            current_batch_size = min(params.batch_size, data_batch.shape[0])
            data_batch = data_batch.view(current_batch_size, -1)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            real_labels = torch.ones(
                current_batch_size, requires_grad=False, device=params.device)
            fake_labels = torch.zeros(
                current_batch_size, requires_grad=False, device=params.device)

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
            noise = torch.randn(current_batch_size,
                                params.latent_vector_size, device=params.device)
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

            if iters % 50 == 0:
                # padded_epoch = str(epoch).ljust(len(str(params.epochs)))
                # padded_batch_idx = str(batch_idx).ljust(len(str(len(dataloader))))
                progress_str = 'Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (err_D.item(), err_G.item(), D_x, D_G_z1, D_G_z2)
                progress.set_description(progress_str)

            # Save Losses for plotting later
            G_losses.append(err_G.item())
            D_losses.append(err_D.item())

            iters += 1

        if epoch % 50 == 0:
            with torch.no_grad():
                generated_sample_count = 7
                noise = torch.randn(
                    generated_sample_count, params.latent_vector_size, device=params.device)
                generated_sine = G(noise)
                generated_sine = generated_sine.view(
                    generated_sample_count, params.sequence_len, features_len)
                fig, ax = plot_sample(generated_sine)
                save_fig(fig, save_path / f"{epoch}.png")
                    
            with torch.no_grad():
                generated_sample_count = 100
                noise = torch.randn(generated_sample_count, params.latent_vector_size, device=params.device)
                generated_sine = G(noise)
                generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                save_box_plot_per_ts(data=generated_sine, epoch=epoch, samples=samples, params=params, save_path=save_path)

    fig, ax = plot_model_losses(G_losses, D_losses, params)
    save_fig(fig, save_path / f"model_losses_after_{params.epochs}.png")
    plt.close(fig)

def save_fig(fig, path):
    fig.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_model_losses(g_losses: list[any], d_losses: list[any], params: TrainParameters):
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
    ax2 = ax.secondary_xaxis()
    ax2.set_xlabel(translate(SimpleGanPlotResultColumns.EPOCHS))

    # ax2.set_xlim(0, params.epochs)

    # epoch_ticks = 
    # ax2.set_xticks([10, 30, 40])
    # ax2.set_xticklabels(['7','8','99'])
    # ax2.set_xscale('log')
    # ax2.set_xticks([10, 30, 40])
    # ax2.set_xticklabels(['7','8','99'])

    return fig, ax


def plot_sample(sample: Tensor):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    unbind_sample = torch.unbind(sample)
    flattened_sample = torch.concat(unbind_sample)
    for i, y in enumerate(torch.transpose(flattened_sample, 0, 1)):
        ax.plot(range(len(y)), y)
    return fig, ax


def save_box_plot_per_ts(data: Tensor, epoch: int, samples: list[Tensor], params: TrainParameters, save_path: PurePath):
    sample_size = data.shape[0]
    features_len = data.shape[2]

    t_data = torch.transpose(data, 0, 1)  # [10, 24, 3]  -> [24, 10, 3]
    t_data_single_feature = torch.unbind(t_data, 2)  # [10, 24, 3] -> list_3([24, 10)

    conc_samples = torch.concat(samples, dim=0)  # list_n([times, sequence, features]) -> [n*times, sequence, features]
    t_samples = torch.transpose(conc_samples, 0, 1)  # [n*times, sequence, features] -> [sequence, n*times, features]
    t_sample_means = torch.mean(t_samples, dim=1)  # [sequence, n*times, features] -> [sequence, mean(features)]

    for feature_idx in range(features_len):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        labels = ["$t_{"+str(i)+"}$" for i in range(params.sequence_len)]
        ax.plot(np.arange(params.sequence_len) + 1,  torch.transpose(t_sample_means, 0, 1)[feature_idx])
        ax.boxplot(t_data_single_feature[feature_idx], labels=labels, bootstrap=5000, showmeans=True, meanline=True, notch=True)

        ax.set_xlabel("t", fontsize=12)
        ax.set_ylabel(r"$G_{t, "+str(feature_idx)+r"}(Z), Z \sim \mathcal{N}(0,1), \vert Z \vert="+str(sample_size)+r"$", fontsize=12)

        save_fig(fig, save_path / f"distribution_result_epoch_{epoch}_feature_{feature_idx}.png")


def setup_models_and_train(params: TrainParameters, samples: list[Tensor], features_len: int, save_path: PurePath):

    G = GeneratorFNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
    )
    D =DiscriminatorFNN(
        features=features_len,
        sequence_len=params.sequence_len,
        out_features=1
    )

    train(
        G=G,
        D=D,
        samples=samples,
        params=params,
        features_len=features_len,
        save_path=save_path
    )
    return D, G

def train_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should give us a baseline for the simplest training possible
    """
    single_sample_univariate = save_images_path / "single_sample_univariate"
    single_sample_univariate.mkdir(parents=True, exist_ok=True)
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples: list[Tensor] = [
        generate_sine_features(sequence_len=params.sequence_len,
                               amplitudes=amplitudes,
                               times=sample_batches,
                               noise_scale=0
                               ),
    ]

    setup_models_and_train(params, samples, features_len, single_sample_univariate)

def train_noisy_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    Check how it affects the training if we add some noise on top of the training data
    """
    noisy_single_sample_univariate = save_images_path / "noisy_single_sample_univariate"
    noisy_single_sample_univariate.mkdir(parents=True, exist_ok=True)
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples: list[Tensor] = [
        generate_sine_features(sequence_len=params.sequence_len,
                               amplitudes=amplitudes,
                               times=sample_batches,
                               noise_scale=0.05
                               ),
    ]

    setup_models_and_train(params, samples, features_len, noisy_single_sample_univariate)


def train_single_sample_multivariate(params: TrainParameters, sample_batches: int):
    """
    This should show us that the convergence is not as fast for multiple features!
    """
    single_sample_multivariate = save_images_path / "single_sample_multivariate"
    single_sample_multivariate.mkdir(parents=True, exist_ok=True)
    amplitudes = [0.5, 1.0]
    features_len = len(amplitudes)
    samples: list[Tensor] = [
        generate_sine_features(sequence_len=params.sequence_len,
                               amplitudes=amplitudes,
                               times=sample_batches,
                               noise_scale=0
                               ),
    ]

    setup_models_and_train(params, samples, features_len, single_sample_multivariate)


def train_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should result in a mode collapse
    """
    multiple_sample_univariate = save_images_path / "multiple_sample_univariate"
    multiple_sample_univariate.mkdir(parents=True, exist_ok=True)
    features_len = 1
    samples: list[Tensor] = [
        generate_sine_features(sequence_len=params.sequence_len,
                               amplitudes=[1],
                               times=sample_batches,
                               noise_scale=0
                               ),
        generate_sine_features(sequence_len=params.sequence_len,
                               amplitudes=[0.75],
                               times=sample_batches,
                               noise_scale=0
                               ),
        generate_sine_features(sequence_len=params.sequence_len,
                               amplitudes=[0.5],
                               times=sample_batches,
                               noise_scale=0
                               ),
    ]

    setup_models_and_train(params, samples, features_len, multiple_sample_univariate)


if __name__ == '__main__':
    set_latex_plot_params()
    train_params = TrainParameters(epochs=600)
    sample_batches = train_params.batch_size*100
    train_single_sample_univariate(train_params, sample_batches)
    train_noisy_single_sample_univariate(train_params, sample_batches)
    train_single_sample_multivariate(train_params, sample_batches)
    train_multiple_sample_univariate(train_params, sample_batches)
