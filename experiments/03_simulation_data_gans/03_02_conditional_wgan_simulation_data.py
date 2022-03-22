from pathlib import PurePath
from typing import Optional
import numpy as np
from tqdm import tqdm
import seaborn as sns

import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad as torch_grad

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from statsmodels.tsa.seasonal import STL

from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params

from experiments.experiments_utils.plotting import (
    plot_model_losses,
    plot_train_data_overlayed,
    plot_box_plot_per_ts,
    plot_sample,
    save_fig,
)
from experiments.experiments_utils.sine_data import SineGenerationParameters, generate_sine_features
from experiments.experiments_utils.train_typing import (
    TrainParameters,
    ConditionalTrainParameters,
    BatchReshaper,
    NoiseGenerator,
)
from experiments.experiments_utils.net_parsing import print_net_summary
from experiments.experiments_utils.weight_init import init_weights
from src.data.data_holder import DataHolder
from src.data.normalization.np.minmax_normalizer import MinMaxNumpyNormalizer
from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer
from src.data.typing import Feature
from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter

from src.net.net_summary import LatexTableOptions, LatexTableStyle
from src.plots.histogram_plot import HistPlotData, draw_hist_plot
from src.plots.timeseries_plot import DecomposeResultColumns, draw_timeseries_plot
from src.utils.datetime_utils import (
    PANDAS_DEFAULT_DATETIME_FORMAT,
    convert_input_str_to_date,
    dates_to_conditional_vectors,
    get_day_in_year_from_date,
    interval_generator,
)

save_images_path = (
    get_experiments_folder().joinpath("03_simulation_data_gans").joinpath("03_02_conditional_wgan_simulation_data")
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
        # x = self.sigmoid(x)
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
        # x = self.tanh(x)
        return x
        # return self.fnn(x) # removed tanh


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


class CWGANGPTrainer:
    def __init__(
        self,
        data_holder: DataHolder,
        G,
        D,
        G_opt,
        D_opt,
        batch_reshaper: BatchReshaper,
        noise_generator: NoiseGenerator,
        params: TrainParameters,
        features_len: int,
        save_path: PurePath,
        gp_weight=10,
        critic_iterations=5,
        print_every=50,
        use_cuda=False,
        plots_file_ending: str = "pdf",
    ):
        self.data_holder = data_holder
        self.G = G
        self.G_opt = G_opt
        self.D = D
        self.D_opt = D_opt
        self.batch_reshaper = batch_reshaper
        self.noise_generator = noise_generator
        self.params = params
        self.features_len = features_len
        self.save_path = save_path
        self.losses = {"G": [], "D": [], "GP": [], "gradient_norm": []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.plots_file_ending = plots_file_ending

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _gradient_penalty(self, real_data, generated_data, real_conditions):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand((batch_size, 1), requires_grad=True)
        # alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        # interpolated = Variable(interpolated, requires_grad=True)
        # if self.use_cuda:
        # interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, real_conditions)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).cuda()
            if self.use_cuda
            else torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses["gradient_norm"].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        epsilon = 1e-12
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + epsilon)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _generator_train_iteration(self, data_batch, conditions_batch):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data_batch.size()[0]
        noise = self.noise_generator(batch_size, self.params, self.features_len)
        generated_data = self.G(noise, conditions_batch)

        # Calculate loss and optimize
        d_generated = self.D(generated_data, conditions_batch)
        g_loss = -d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses["G"].append(g_loss.item())

    def _critic_train_iteration(self, data_batch, conditions_batch):
        """ """
        # Get generated data
        batch_size = data_batch.size()[0]
        noise = self.noise_generator(batch_size, self.params, self.features_len)
        generated_data = self.G(noise, conditions_batch)

        d_real = self.D(data_batch, conditions_batch)
        d_generated = self.D(generated_data, conditions_batch)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data_batch, generated_data, conditions_batch)
        self.losses["GP"].append(gradient_penalty.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses["D"].append(d_loss.item())

    def _train_epoch(self, data_loader):
        for i, (data_batch, conditions_batch) in enumerate(data_loader):
            current_batch_size = min(self.params.batch_size, data_batch.shape[0])
            data_batch = data_batch = self.batch_reshaper(
                data_batch, current_batch_size, self.params.sequence_len, self.features_len
            )
            self.num_steps += 1
            self._critic_train_iteration(data_batch, conditions_batch)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data_batch, conditions_batch)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses["D"][-1]))
                print("GP: {}".format(self.losses["GP"][-1]))
                print("Gradient norm: {}".format(self.losses["gradient_norm"][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses["G"][-1]))

    def train(self, data_loader, epochs):
        loss_save_path = self.save_path / "losses"
        loss_save_path.mkdir(parents=True, exist_ok=True)
        sample_save_path = self.save_path / "sample"
        sample_save_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            print("\nEpoch {}".format(epoch))
            self._train_epoch(data_loader)

            if (epoch == 1) or (epoch % 50 == 0):
                fig, ax = plot_model_losses(g_losses=self.losses["G"], d_losses=self.losses["D"], current_epoch=epoch)
                save_fig(fig, loss_save_path / f"losses_after_{epoch}.{self.plots_file_ending}")

                with torch.no_grad():
                    start_date: str = "2009.01.01"
                    end_date: str = "2009.01.08"
                    start = convert_input_str_to_date(start_date)
                    end = convert_input_str_to_date(end_date)
                    batch_conditions = torch.from_numpy(
                        np.array([get_day_in_year_from_date(d) for d in interval_generator(start, end)])
                    )
                    generated_sample_count = batch_conditions.size(0)
                    batch_noise = self.noise_generator(generated_sample_count, self.params, self.features_len)
                    generated_data = self.G(batch_noise, batch_conditions)
                    generated_data = generated_data.view(
                        generated_sample_count, self.params.sequence_len, self.features_len
                    )
                    if self.data_holder.normalizer is not None:
                        generated_data = self.data_holder.normalizer.renormalize(generated_data)
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 1))
                    fig, ax = plot_sample(sample=generated_data, params=self.params, plot=(fig, ax), condition="ALL")
                    save_fig(fig, sample_save_path / f"{epoch}_cond_ALL.{self.plots_file_ending}")


def setup_fnn_models_and_train(
    data_holder: DataHolder,
    params: ConditionalTrainParameters,
    save_path: PurePath,
    latex_options: Optional[LatexTableOptions],
    dropout: float = 0.5,
):
    conditions = 366  #
    features_len = data_holder.get_feature_size()

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

    # init_weights(G, "xavier", init_gain=nn.init.calculate_gain("leaky_relu"))
    # init_weights(D, "xavier", init_gain=nn.init.calculate_gain("leaky_relu"))

    # Learning rate for optimizers
    lr = 1e-4
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.9
    beta2 = 0.99

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    # optimizerD = optim.RMSprop(D.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    # optimizerG = optim.RMSprop(G.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)

    data = torch.from_numpy(data_holder.data).view(-1, 24, data_holder.get_feature_size())
    samples = [data]
    data_conditions = torch.from_numpy(data_holder.conditions).view(-1, 24)[..., 0]
    conditions = [data_conditions]
    flattened_samples = torch.concat(samples, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
    flattened_conditions = torch.concat(conditions, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
    dataset = TensorDataset(flattened_samples, flattened_conditions)
    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
    )

    trainer = CWGANGPTrainer(
        data_holder=data_holder,
        G=G,
        G_opt=optimizerG,
        D=D,
        D_opt=optimizerD,
        batch_reshaper=fnn_batch_reshaper,
        noise_generator=fnn_noise_generator,
        params=params,
        features_len=features_len,
        save_path=save_path,
        # latex_options=latex_options,
    )
    trainer.train(dataloader, params.epochs)
    return D, G


def setup_cnn_models_and_train(
    data_holder: DataHolder,
    params: ConditionalTrainParameters,
    save_path: PurePath,
    latex_options: Optional[LatexTableOptions],
    dropout: float = 0.5,
):
    conditions = 366  #
    features_len = data_holder.get_feature_size()

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

    # init_weights(G, "xavier", init_gain=nn.init.calculate_gain("leaky_relu"))
    # init_weights(D, "xavier", init_gain=nn.init.calculate_gain("leaky_relu"))

    # Learning rate for optimizers
    lr = 1e-4
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.9
    beta2 = 0.99

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    # optimizerD = optim.RMSprop(D.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    # optimizerG = optim.RMSprop(G.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)

    data = torch.from_numpy(data_holder.data).view(-1, 24, data_holder.get_feature_size())
    samples = [data]
    data_conditions = torch.from_numpy(data_holder.conditions).view(-1, 24)[..., 0]
    conditions = [data_conditions]
    flattened_samples = torch.concat(samples, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
    flattened_conditions = torch.concat(conditions, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
    dataset = TensorDataset(flattened_samples, flattened_conditions)
    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
    )

    trainer = CWGANGPTrainer(
        data_holder=data_holder,
        G=G,
        G_opt=optimizerG,
        D=D,
        D_opt=optimizerD,
        batch_reshaper=fnn_batch_reshaper,
        noise_generator=fnn_noise_generator,
        params=params,
        features_len=features_len,
        save_path=save_path,
        # latex_options=latex_options,
    )
    trainer.train(dataloader, params.epochs)
    return D, G


def train_all_features(data_holder: DataHolder, params: TrainParameters):
    """
    Multivariate for conditional GAN with FNN
    """
    latex_options = LatexTableOptions(
        caption="Conditional GAN {net_type}-Netz in Form eines mehrschichtigen Perzeptrons für Simulationsdaten",
        label="conditional_gan_fnn_sines_net_simulation_data_{net_type}",
        # style=LatexTableStyle(scaleWithAdjustbox=1.0),
    )
    setup_fnn_models_and_train(
        data_holder=data_holder,
        params=params,
        save_path=save_images_path / "fnn_all_features",
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
    start_date_str = DEFAULT_DATA_START_DATE
    end_date_str = "2010-12-31 23:00:00"  # "2019-12-31 23:00:00"
    start_date = convert_input_str_to_date(start_date_str, format=PANDAS_DEFAULT_DATETIME_FORMAT)
    end_date = convert_input_str_to_date(end_date_str, format=PANDAS_DEFAULT_DATETIME_FORMAT)
    data_importer = DWDWeatherDataImporter(start_date=start_date_str, end_date=end_date_str)
    data_importer.initialize()
    # conditions = np.array([get_day_in_year_from_date(d) for d in interval_generator(start_date, end_date)])
    data_holder = DataHolder(
        data=data_importer.data.values.astype(np.float32),
        data_labels=data_importer.get_feature_labels(),
        dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
        conditions=np.array(data_importer.get_day_of_year_values()),
        normalizer_constructor=StandardNumpyNormalizer,
    )
    train_params = ConditionalTrainParameters(batch_size=64, epochs=4000, embedding_dim=32)

    train_all_features(data_holder=data_holder, params=train_params)


if __name__ == "__main__":
    main()
