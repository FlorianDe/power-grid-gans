from pathlib import PurePath
from typing import Callable, Optional
from tqdm import tqdm
import seaborn as sns

import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch import Size, Tensor
from torch.utils.data import DataLoader, TensorDataset

from experiments.utils import get_experiments_folder, set_latex_plot_params

from plotting import plot_model_losses, plot_train_data_overlayed, plot_box_plot_per_ts, plot_sample, save_fig
from sine_data import SineGenerationParameters, generate_sine_features
from src.net.net_summary import LatexTableOptions, LatexTableStyle
from train_typing import TrainParameters, ConditionalTrainParameters, BatchReshaper, NoiseGenerator
from net_parsing import print_net_summary

save_images_path = (
    get_experiments_folder().joinpath("02_sinusoidal_data_basic_gan").joinpath("02_02_conditional_gan_sines")
)
save_images_path.mkdir(parents=True, exist_ok=True)


def fnn_batch_reshaper(data_batch: Tensor, batch_size: int, sequence_len: int, features_len: int) -> Tensor:
    return data_batch.view(batch_size, sequence_len * features_len)


def fnn_noise_generator(current_batch_size: int, params: TrainParameters, features_len: int) -> Tensor:
    return torch.randn(current_batch_size, params.latent_vector_size, device=params.device)


class DiscriminatorFNN(nn.Module):
    def __init__(
        self,
        features: int,
        sequence_len: int,
        conditions: int,
        embeddings: int,
        out_features: int = 1,
        dropout: float = 0.5,
    ):
        super(DiscriminatorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        input_size = sequence_len * features

        def dense_block(input: int, output: int, normalize=True):
            negative_slope = 1e-2
            layers: list[nn.Module] = []
            if normalize:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(input, output))
            # layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            return layers

        self.embedding = nn.Embedding(conditions, embeddings)
        self.fnn = nn.Sequential(
            *dense_block(input_size + embeddings, 2 * input_size, False),
            *dense_block(2 * input_size, 4 * input_size),
            # *dense_block(4 * input_size, 4 * input_size),
            nn.Linear(4 * input_size, out_features),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        embedded_conditions = self.embedding(condition)
        x = torch.cat((x, embedded_conditions), dim=1)
        x = self.fnn(x)
        x = self.sigmoid(x)
        return x


class GeneratorFNN(nn.Module):
    def __init__(
        self,
        latent_vector_size: int,
        features: int,
        sequence_len: int,
        conditions: int,
        embeddings: int,
        dropout: float = 0.5,
    ):
        super(GeneratorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.embedding = nn.Embedding(conditions, embeddings)

        def dense_block(input: int, output: int, normalize=True):
            negative_slope = 1e-2
            layers: list[nn.Module] = []
            if normalize:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(input, output))
            # layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            return layers

        self.fnn = nn.Sequential(
            *dense_block(latent_vector_size + embeddings, 2 * latent_vector_size),
            *dense_block(2 * latent_vector_size, 4 * latent_vector_size),
            # *dense_block(4 * latent_vector_size, 4 * latent_vector_size),
            nn.Linear(4 * latent_vector_size, features * sequence_len),
        )
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        embedded_conditions = self.embedding(condition)
        x = torch.cat((x, embedded_conditions), dim=1)
        x = self.fnn(x)
        x = self.tanh(x)
        return x
        # return self.fnn(x) # removed tanh


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
    # Learning rate for optimizers
    lr = 1e-3
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.9

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # generate sample data
    samples = [generate_sine_features(params) for params in samples_parameters]
    conditions = [torch.squeeze(torch.full((params.times, 1), idx)) for idx, params in enumerate(samples_parameters)]

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
        print(f"{i}. sample {sample.shape} with params: {sample_params} and condition: {i}")
    flattened_samples = torch.concat(samples, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
    flattened_conditions = torch.concat(conditions, dim=0)  # TODO FOR NOW JUST CONCAT THEM!

    print(f"{flattened_samples.shape=}")
    print(f"{flattened_conditions.shape=}")
    dataset = TensorDataset(flattened_samples, flattened_conditions)
    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        # num_workers=workers
    )

    if latex_options:
        (data_batch, conditions_batch) = next(iter(dataloader))
        discriminator_input_size = batch_reshaper(data_batch, params.batch_size, params.sequence_len, features_len)[
            0
        ].size()
        generator_input_size = noise_generator(params.batch_size, params, features_len)[0].size()
        conditions_size = 1
        print_net_summary(
            G=G,
            D=D,
            generator_input_size=[generator_input_size, conditions_size],
            discriminator_input_size=[discriminator_input_size, conditions_size],
            latex_options=latex_options,
            dtypes=[torch.FloatTensor, torch.IntTensor],
        )

    G_losses = []
    D_losses = []
    iters = 0

    for epoch in (
        progress := tqdm(
            range(1, params.epochs + 1), unit="epochs", bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}"
        )
    ):
        for batch_idx, (data_batch, conditions_batch) in enumerate(dataloader):
            current_batch_size = min(params.batch_size, data_batch.shape[0])
            data_batch = batch_reshaper(data_batch, current_batch_size, params.sequence_len, features_len)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # real_labels = torch.ones(current_batch_size, requires_grad=False, device=params.device)  # try with 0.9
            real_labels = torch.squeeze(
                torch.full((current_batch_size, 1), 0.9, requires_grad=False, device=params.device)
            )
            fake_labels = torch.zeros(current_batch_size, requires_grad=False, device=params.device)

            ## Train with all-real batch
            D.zero_grad()
            # label = torch.full((current_batch_size), real_label_value, dtype=torch.float, device=params.device)
            d_out_real = D(data_batch, conditions_batch).view(-1)
            # print(f"{d_out_real.shape=}")
            d_err_real = criterion(d_out_real, real_labels)
            d_err_real.backward()
            D_x = d_err_real.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = noise_generator(current_batch_size, params, features_len)
            fake_generated = G(noise, conditions_batch)
            # print(f"{fake_generated.shape=}")
            d_out_fake = D(fake_generated.detach(), conditions_batch).view(-1)  # TODO OTHER CONDITIONS?
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
            d_out_fake = D(fake_generated, conditions_batch).view(-1)
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

        if epoch % 10 == 0:
            for cond_idx, sample in enumerate(samples):
                with torch.no_grad():
                    generated_sample_count = 7
                    batch_noise = noise_generator(generated_sample_count, params, features_len)
                    batch_conditions = torch.squeeze(torch.full((generated_sample_count, 1), cond_idx))
                    generated_sine = G(batch_noise, batch_conditions)
                    generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
                    fig, ax = plot_sample(sample=generated_sine, params=params, plot=(fig, ax), condition=cond_idx)
                    save_fig(fig, sample_save_path / f"{epoch}_cond_{cond_idx}.{plots_file_ending}")

                with torch.no_grad():
                    generated_sample_count = 100
                    batch_noise = noise_generator(generated_sample_count, params, features_len)
                    batch_conditions = torch.squeeze(torch.full((generated_sample_count, 1), cond_idx))
                    generated_sine = G(batch_noise, batch_conditions)
                    generated_sine = generated_sine.view(generated_sample_count, params.sequence_len, features_len)
                    for feature_idx, (fig, ax) in enumerate(
                        plot_box_plot_per_ts(
                            data=generated_sine, epoch=epoch, samples=[sample], params=params, condition=cond_idx
                        )
                    ):
                        save_fig(
                            fig,
                            distributions_save_path
                            / f"distribution_epoch_{epoch}_cond_{cond_idx}_feature_{feature_idx}.{plots_file_ending}",
                        )

            fig, ax = plot_model_losses(g_losses=G_losses, d_losses=D_losses, current_epoch=epoch)
            save_fig(fig, loss_save_path / f"losses_after_{epoch}.{plots_file_ending}")
            plt.close(fig)
    print("End training\n--------------------------------------------")


# # weight initialization
# def init_weights(m: nn.Module):
#     if type(m) == nn.Linear:
#         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
#         # torch.nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain("relu"))
#         m.bias.data.fill_(0.01)


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.01)
        elif classname.find("BatchNorm1d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def setup_fnn_models_and_train(
    params: ConditionalTrainParameters,
    samples_parameters: list[SineGenerationParameters],
    features_len: int,
    save_path: PurePath,
    latex_options: Optional[LatexTableOptions],
    dropout: float = 0.5,
):
    conditions = len(samples_parameters)
    G = GeneratorFNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
        dropout=dropout,
        conditions=conditions,
        embeddings=params.embedding_dim,
    )

    D = DiscriminatorFNN(
        features=features_len,
        sequence_len=params.sequence_len,
        conditions=conditions,
        embeddings=params.embedding_dim,
        out_features=1,
        dropout=dropout,
    )
    init_weights(D, "xavier", init_gain=nn.init.calculate_gain("relu"))
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


def train_fnn_multiple_sample_univariate(params: TrainParameters, sample_batches: int):
    """
    Baseline for conditional GAN with FNN
    """
    features_len = 1
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.95], times=sample_batches, noise_scale=0.01
        ),
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.25], times=sample_batches, noise_scale=0.01
        ),
    ]
    setup_fnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "fnn_multiple_sample_univariate",
        latex_options=None,
    )


def train_fnn_multiple_sample_multivariate(params: TrainParameters, sample_batches: int):
    """
    Multivariate for conditional GAN with FNN
    """
    features_len = 2
    samples_parameters: list[SineGenerationParameters] = [
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.95, 0.5], times=sample_batches, noise_scale=0.01
        ),
        SineGenerationParameters(
            sequence_len=params.sequence_len, amplitudes=[0.5, 0.2], times=sample_batches, noise_scale=0.01
        ),
    ]
    latex_options = LatexTableOptions(
        caption="Conditional GAN {net_type}-Netz in Form eines mehrschichtigen Perzeptrons für multivariate sinusoidale Daten",
        label="conditional_gan_fnn_sines_net_multiple_multivariate_{net_type}",
        style=LatexTableStyle(scaleWithAdjustbox=1.4),
    )
    setup_fnn_models_and_train(
        params=params,
        samples_parameters=samples_parameters,
        features_len=features_len,
        save_path=save_images_path / "fnn_multiple_sample_multivariate",
        latex_options=latex_options,
    )


def main():
    sns.set_theme()
    sns.set_context("paper")
    # sns.set_palette("colorblind")
    set_latex_plot_params()

    manualSeed = 1337
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    # rng = np.random.default_rng(seed=0) # use for numpy
    torch.manual_seed(manualSeed)

    train_params = ConditionalTrainParameters(batch_size=16, epochs=4000, embedding_dim=32)
    sample_batches = train_params.batch_size * 128

    # FNN trainings
    # train_fnn_multiple_sample_univariate(train_params, sample_batches)
    train_fnn_multiple_sample_multivariate(train_params, sample_batches)


if __name__ == "__main__":
    main()
