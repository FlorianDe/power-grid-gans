from pathlib import PurePath
from typing import Optional
from tqdm import tqdm
import seaborn as sns

import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params
from experiments.experiments_utils.plotting import (
    plot_model_losses,
    plot_train_data_overlayed,
    plot_box_plot_per_ts,
    plot_sample,
    save_fig,
)
from src.net.summary.net_summary import LatexTableOptions, LatexTableStyle
from experiments.experiments_utils.train_typing import TrainParameters, AdamParameters, BatchReshaper, NoiseGenerator
from experiments.experiments_utils.sine_data import SineGenerationParameters, generate_sine_features
from src.net.summary.net_parsing import print_net_summary

save_images_path = get_experiments_folder().joinpath("02_sinusoidal_data_gans").joinpath("02_01_vanilla_gan_fnn_sines")
save_images_path.mkdir(parents=True, exist_ok=True)


def fnn_batch_reshaper(data_batch: Tensor, batch_size: int, sequence_len: int, features_len: int) -> Tensor:
    return data_batch.view(batch_size, sequence_len * features_len)


def fnn_noise_generator(current_batch_size: int, params: TrainParameters, features_len: int) -> Tensor:
    return torch.randn(current_batch_size, params.latent_vector_size, device=params.device)


def dense_block(input: int, output: int, dropout: float, activation_slope: Optional[float]):
    layers: list[nn.Module] = []
    if dropout != 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(input, output))
    if activation_slope:
        layers.append(nn.LeakyReLU(activation_slope, inplace=True))
    return layers


class DiscriminatorFNN(nn.Module):
    def __init__(
        self, features: int, sequence_len: int, out_features: int = 1, dropout: float = 0.5, negative_slope=1e-2
    ):
        super(DiscriminatorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        input_size = sequence_len * features
        self.fnn = nn.Sequential(
            *dense_block(input_size, 2 * input_size, dropout, negative_slope),
            *dense_block(2 * input_size, 4 * input_size, dropout, negative_slope),
            *dense_block(4 * input_size, out_features, dropout, None),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.fnn(x))


class GeneratorFNN(nn.Module):
    def __init__(
        self, latent_vector_size: int, features: int, sequence_len: int, dropout: float = 0.5, negative_slope=1e-2
    ):
        super(GeneratorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.fnn = nn.Sequential(
            *dense_block(latent_vector_size, 2 * latent_vector_size, dropout, negative_slope),
            *dense_block(2 * latent_vector_size, 4 * latent_vector_size, dropout, negative_slope),
            *dense_block(4 * latent_vector_size, features * sequence_len, dropout, None),
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
            # nn.LeakyReLU(0.2, inplace=True),
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
    def __init__(self, features: int, sequence_len: int, out_features: int = 1):
        super(DiscriminatorRNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.out_features = out_features
        self.hidden_size = 2 * out_features  # hardcoded
        self.num_layers = 1
        self.rnn = nn.GRU(
            input_size=self.features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=0.5,
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size, self.out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()  # TODO ADD DEVICE!
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
        self.hidden_size = 2 * self.sequence_len  # hardcoded

        self.rnn = nn.GRU(
            input_size=self.latent_vector_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
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


def train(
    G: nn.Module,
    D: nn.Module,
    batch_reshaper: BatchReshaper,
    noise_generator: NoiseGenerator,
    samples_parameters: list[SineGenerationParameters],
    params: TrainParameters,
    features_len: int,
    save_path: PurePath,
    latex_options: Optional[LatexTableOptions] = None,
    plots_file_ending: str = "pdf",
):
    beta1 = params.optimizer_parameters.beta1
    beta2 = params.optimizer_parameters.beta2

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(D.parameters(), lr=params.optimizer_parameters.lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(G.parameters(), lr=params.optimizer_parameters.lr, betas=(beta1, beta2))

    # generate sample data
    samples = [
        generate_sine_features(params=sample_params, device=params.device) for sample_params in samples_parameters
    ]

    # create train test directory
    save_path.mkdir(parents=True, exist_ok=True)
    loss_save_path = save_path / "losses"
    loss_save_path.mkdir(parents=True, exist_ok=True)
    distributions_save_path = save_path / "distributions"
    distributions_save_path.mkdir(parents=True, exist_ok=True)
    sample_save_path = save_path / "sample"
    sample_save_path.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_train_data_overlayed(samples, samples_parameters, params)
    save_fig(fig, save_path / f"train_data_plot.{plots_file_ending}")

    print(f"Preparing training data for: {save_path.name}")
    print(f"Start training with samples:")
    for i, (sample, sample_params) in enumerate(zip(samples, samples_parameters)):
        print(f"{i}. sample {sample.shape} with params: {sample_params}")
    flattened_samples = torch.concat(samples, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
    print(f"{flattened_samples.shape=}")
    dataloader = DataLoader(
        flattened_samples,
        batch_size=params.batch_size,
        shuffle=False,
        # num_workers=workers
        # pin_memory=True,
    )
    if latex_options:
        data_batch = next(iter(dataloader))
        discriminator_input_size = batch_reshaper(data_batch, params.batch_size, params.sequence_len, features_len)[
            0
        ].size()
        generator_input_size = noise_generator(params.batch_size, params, features_len)[0].size()
        print_net_summary(
            G=G,
            D=D,
            generator_input_size=generator_input_size,
            discriminator_input_size=discriminator_input_size,
            latex_options=latex_options,
        )

    G_losses = []
    D_losses = []
    iters = 0

    for epoch in (
        progress := tqdm(
            range(1, params.epochs + 1), unit="epochs", bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}"
        )
    ):
        for batch_idx, (data_batch) in enumerate(dataloader):
            current_batch_size = min(params.batch_size, data_batch.shape[0])
            data_batch = batch_reshaper(data_batch, current_batch_size, params.sequence_len, features_len).to(
                params.device
            )

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            real_labels = torch.ones(current_batch_size, requires_grad=False, device=params.device)
            fake_labels = torch.zeros(current_batch_size, requires_grad=False, device=params.device)

            ## Train with all-real batch
            D.zero_grad()
            # label = torch.full((current_batch_size), real_label_value, dtype=torch.float, device=params.device)
            d_out_real = D(data_batch).view(-1)
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

            if iters % 40 == 0:
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

        if epoch % 10 == 0:
            with torch.no_grad():
                generated_sample_count = 7
                noise = noise_generator(generated_sample_count, params, features_len)

                generated_sine = G(noise)
                generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
                fig, ax = plot_sample(sample=generated_sine, params=params, plot=(fig, ax))
                save_fig(fig, sample_save_path / f"sample_after_{epoch}.{plots_file_ending}")

            with torch.no_grad():
                generated_sample_count = 100
                noise = noise_generator(generated_sample_count, params, features_len)
                generated_sine = G(noise)
                generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                for feature_idx, (fig, ax) in enumerate(
                    plot_box_plot_per_ts(data=generated_sine, epoch=epoch, samples=samples, params=params)
                ):
                    save_fig(
                        fig,
                        distributions_save_path
                        / f"distribution_result_epoch_{epoch}_feature_{feature_idx}.{plots_file_ending}",
                    )

            fig, ax = plot_model_losses(G_losses, D_losses, epoch)
            save_fig(fig, loss_save_path / f"losses_after_{epoch}.{plots_file_ending}")
    print("End training\n--------------------------------------------")


def setup_fnn_models_and_train(
    params: TrainParameters,
    samples_parameters: list[SineGenerationParameters],
    features_len: int,
    save_path: PurePath,
    latex_options: Optional[LatexTableOptions],
    dropout: float = 0.5,
):
    G = GeneratorFNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
        dropout=dropout,
    ).to(params.device)
    D = DiscriminatorFNN(features=features_len, sequence_len=params.sequence_len, out_features=1, dropout=dropout).to(
        params.device
    )

    train(
        G=G,
        D=D,
        batch_reshaper=fnn_batch_reshaper,
        noise_generator=fnn_noise_generator,
        samples_parameters=samples_parameters,
        params=params,
        features_len=features_len,
        save_path=save_path,
        latex_options=latex_options,
    )
    return D, G


def train_fnn_single_sample_univariate_no_regularization(params: TrainParameters, sample_batches: int):
    """
    This should give us a baseline for the simplest training possible
    """
    amplitudes = [0.75]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0
        ),
    ]
    latex_options = LatexTableOptions(
        caption="Vanilla GAN {net_type}-Netz in Form eines mehrschichtigen Perzeptrons für univariate sinusoidale Daten ohne Regularisierung",
        label="fnn_sines_net_single_univariate_no_regularization_{net_type}",
    )
    setup_fnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "fnn_single_sample_univariate_no_regularization",
        dropout=0,
        latex_options=latex_options,
    )


def train_fnn_noisy_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    Check how it affects the training if we add some noise on top of the training data
    """
    amplitudes = [0.75]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.05
        ),
    ]
    latex_options = LatexTableOptions(
        caption="Vanilla GAN {net_type}-Netz in Form eines mehrschichtigen Perzeptrons für univariate sinusoidale Daten",
        label="fnn_sines_net_single_univariate_{net_type}",
    )
    setup_fnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "fnn_noisy_single_sample_univariate",
        latex_options=latex_options,
    )


def train_fnn_single_sample_multivariate(params: TrainParameters, sample_batches: int):
    """
    This should show us that the convergence is not as fast for multiple features!
    """
    amplitudes = [0.5, 1.0]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.01
        )
    ]
    latex_options = LatexTableOptions(
        caption="Vanilla GAN {net_type}-Netz in Form eines mehrschichtigen Perzeptrons für multivariate sinusoidale Daten",
        label="fnn_sines_net_single_multivariate_{net_type}",
    )
    setup_fnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "fnn_single_sample_multivariate",
        latex_options=latex_options,
    )


def train_fnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    This should result in a mode collapse
    """
    features_len = 1
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[1], times=sample_batches, noise_scale=0.01
        ),
        # SineGenerationParameters(
        #     sequence_len=params.sequence_len, amplitudes=[0.75], times=sample_batches, noise_scale=0.01
        # ),
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0.01
        ),
    ]
    setup_fnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "fnn_multiple_sample_univariate",
        latex_options=None,
    )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def setup_cnn_models_and_train(
    params: TrainParameters,
    samples_parameters: list[SineGenerationParameters],
    features_len: int,
    save_path: PurePath,
    latex_options: LatexTableOptions,
):
    G = GeneratorCNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
    ).to(params.device)
    G.apply(weights_init)

    D = DiscriminatorCNN(features=features_len, sequence_len=params.sequence_len, out_features=1).to(params.device)
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
        latex_options=latex_options,
    )
    return G


def train_cnn_single_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    XXXX
    """
    amplitudes = [1.0]
    features_len = len(amplitudes)
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=amplitudes, times=sample_batches, noise_scale=0.01
        )
    ]
    latex_options = LatexTableOptions(
        caption="DCGAN {net_type}-Netz in Form eines Convolutional Neuronal Networks für univariate sinusoidale Daten. Die Abkürzungen der Optionen entsprechen: in=in\_channels, out=out\_channels, k=kernel\_size, s=stride, p=padding, op=output\_padding, pm=padding\_mode, d=dilation, g=groups, e=epsilon.",
        label="cnn_sines_net_single_univariate_{net_type}",
        style=LatexTableStyle(useTabularx=True, scaleWithAdjustbox=1.4),
    )

    return setup_cnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "cnn_single_sample_univariate",
        latex_options=latex_options,
    )


def train_cnn_single_sample_multivariate(params: TrainParameters, sample_batches: int):
    """
    XXXX
    """
    features_len = 2
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[1, 0.5], times=sample_batches, noise_scale=0.01
        ),
    ]
    latex_options = LatexTableOptions(
        caption="DCGAN {net_type}-Netz in Form eines Convolutional Neuronal Networks für multivariate sinusoidale Daten. Die Abkürzungen der Optionen entsprechen: in=in\_channels, out=out\_channels, k=kernel\_size, s=stride, p=padding, op=output\_padding, pm=padding\_mode, d=dilation, g=groups, e=epsilon.",
        label="cnn_sines_net_single_multivariate_{net_type}",
        style=LatexTableStyle(useTabularx=True, scaleWithAdjustbox=1.4),
    )
    return setup_cnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "cnn_single_sample_multivariate",
        latex_options=latex_options,
    )


def train_cnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    XXXX
    """
    features_len = 1
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[1], times=sample_batches, noise_scale=0.01
        ),
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0.01
        ),
    ]

    return setup_cnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "cnn_multiple_sample_univariate",
        latex_options=None,
    )


def setup_rnn_models_and_train(
    params: TrainParameters, samples_parameters: list[SineGenerationParameters], features_len: int, save_path: PurePath
):
    G = GeneratorRNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
    ).to(params.device)

    D = DiscriminatorRNN(features=features_len, out_features=1, sequence_len=params.sequence_len).to(params.device)

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
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.5], times=sample_batches, noise_scale=0.01
        ),
        # SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[1, 0.5, 0.25], times=sample_batches, noise_scale=0.01)
    ]

    return setup_rnn_models_and_train(
        params, samples_parameters, features_len, save_images_path / "rnn_multiple_sample_univariate"
    )


def save_multi_sample_multivariate_training_data_sample_overview(params: TrainParameters):
    times = 500
    save_path = save_images_path
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[1], times=times, noise_scale=0.05),
        SineGenerationParameters(sequence_len=params.sequence_len, amplitudes=[2, 0.5], times=times, noise_scale=0.01),
    ]
    # generate sample data
    samples = [generate_sine_features(params=sine_params, device=params.device) for sine_params in samples_parameters]

    # create train test directory
    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_train_data_overlayed(samples, samples_parameters, params)
    save_fig(fig, save_path / "train_data_plot.pdf")


def main():
    sns.set_theme()
    sns.set_context("paper")
    # sns.set_palette("colorblind")
    set_latex_plot_params()

    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    manualSeed = 1337
    print("Using device: ", device)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    # rng = np.random.default_rng(seed=0) # use for numpy
    torch.manual_seed(manualSeed)

    batch_size = 16
    sample_batches = batch_size * 128

    fnn_train_params = TrainParameters(batch_size=16, epochs=4000, device=device)
    # save sample image for synthetic test data
    save_multi_sample_multivariate_training_data_sample_overview(fnn_train_params)

    # train_fnn_single_sample_univariate_no_regularization(train_params, sample_batches)
    # train_fnn_noisy_single_sample_univariate(train_params, sample_batches)
    # train_fnn_single_sample_multivariate(train_params, sample_batches)
    # train_fnn_multiple_sample_univariate(train_params, sample_batches)

    # CNN trainings
    cnn_optimizer_parameters = AdamParameters(lr=0.0002, beta1=0.5)
    cnn_train_params = TrainParameters(
        batch_size=128, epochs=4000, optimizer_parameters=cnn_optimizer_parameters, device=device
    )
    # train_cnn_single_sample_univariate(cnn_train_params, sample_batches)
    train_cnn_single_sample_multivariate(cnn_train_params, sample_batches)
    # train_cnn_multiple_sample_univariate(cnn_train_params, sample_batches)

    # RNN
    # train_rnn_multiple_sample_univariate(train_params, sample_batches)


if __name__ == "__main__":
    main()
