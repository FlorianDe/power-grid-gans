from datetime import timedelta, datetime
from pathlib import PurePath
from typing import Optional
import numpy as np
import pandas as pd
import seaborn as sns

import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torch import Tensor

from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params

from experiments.experiments_utils.plotting import (
    plot_model_losses,
    plot_sample,
    save_fig,
)
from src.data.weather.weather_dwd_postprocessor import DWDWeatherPostProcessor
from src.evaluator.evaluator import Evaluator
from src.gan.trainer.cgan_trainer import CGANTrainer
from src.gan.trainer.typing import ConditionalTrainParameters, TrainModel, TrainingEpoch
from src.data.data_holder import DataHolder
from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer
from src.data.typing import Feature
from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns

from src.net.summary.net_summary import LatexTableOptions
from src.net.weight_init import init_weights
from src.plots.histogram_plot import HistPlotData, draw_hist_plot
from src.plots.typing import PlotData, PlotOptions
from src.plots.violin_plot import draw_violin_plot
from src.plots.zoom_line_plot import ConnectorBoxOptions, ZoomBoxEffectOptions, ZoomPlotOptions, draw_zoom_line_plot
from src.utils.datetime_utils import (
    convert_input_str_to_date,
    dates_to_conditional_vectors,
    get_day_in_year_from_date,
    interval_generator,
)

save_images_path = (
    get_experiments_folder().joinpath("03_simulation_data_gans").joinpath("03_01_conditional_gan_simulation_data")
)
save_images_path.mkdir(parents=True, exist_ok=True)
# create train test directory

# def fnn_batch_reshaper(data_batch: Tensor, batch_size: int, sequence_len: int, features_len: int) -> Tensor:
#     return data_batch.view(batch_size, sequence_len * features_len)


# def fnn_noise_generator(current_batch_size: int, params: TrainParameters, features_len: int) -> Tensor:
#     return torch.randn(current_batch_size, params.latent_vector_size, device=params.device)


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
            *dense_block(4 * input_size, 8 * input_size),
            # *dense_block(8 * input_size, 16 * input_size),
            # *dense_block(16 * input_size, 8 * input_size),
            *dense_block(8 * input_size, 4 * input_size),
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
            *dense_block(4 * latent_vector_size, 8 * latent_vector_size),
            # *dense_block(8 * latent_vector_size, 16 * latent_vector_size),
            # *dense_block(16 * latent_vector_size, 8 * latent_vector_size),
            *dense_block(8 * latent_vector_size, 4 * latent_vector_size),
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


# def train(
#     data_holder: DataHolder,
#     G: nn.Module,
#     D: nn.Module,
#     batch_reshaper: BatchReshaper,
#     noise_generator: NoiseGenerator,
#     params: TrainParameters,
#     features_len: int,
#     save_path: PurePath,
#     latex_options: Optional[LatexTableOptions] = None,
#     plots_file_ending: str = "pdf",
# ):
#     # Learning rate for optimizers
#     lr = 1e-3
#     # Beta1 hyperparam for Adam optimizers
#     beta1 = 0.9
#     beta2 = 0.99

#     criterion = nn.BCELoss()  # nn.BCEWithLogitsLoss()

#     # optimizerD = optim.RMSprop(D.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
#     # optimizerG = optim.RMSprop(G.parameters(), lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
#     optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
#     optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

#     # reshape data
#     print(f"{data_holder.data.shape=}")
#     data = torch.from_numpy(data_holder.data).view(-1, 24, data_holder.get_feature_size())
#     samples = [data]
#     data_conditions = torch.from_numpy(data_holder.conditions).view(-1, 24)[
#         ..., 0
#     ]  # torch.from_numpy(np.array([idx % 366 for idx in range(data.size(0))])) #NOT CORRECT for multiple years etc
#     print(data_conditions[data_conditions == 0])
#     conditions = [data_conditions]
#     print(f"{data.shape=}")
#     print(f"{data_conditions.shape=}")

#     # create train test directory
#     save_path.mkdir(parents=True, exist_ok=True)
#     loss_save_path = save_path / "losses"
#     loss_save_path.mkdir(parents=True, exist_ok=True)
#     distributions_save_path = save_path / "distributions"
#     distributions_save_path.mkdir(parents=True, exist_ok=True)
#     sample_save_path = save_path / "sample"
#     sample_save_path.mkdir(parents=True, exist_ok=True)

#     # TODO CREATE PLOTS
#     # fig, _ = plot_train_data_overlayed(samples, samples_parameters, params)
#     # save_fig(fig, save_path / f"train_data_plot.{plots_file_ending}")

#     print(f"Preparing training data for: {save_path.name}")
#     # print(f"Start training with samples:")
#     # for i, (sample, sample_params) in enumerate(zip(samples, samples_parameters)):
#     #     print(f"{i}. sample {sample.shape} with params: {sample_params} and condition: {i}")
#     flattened_samples = torch.concat(samples, dim=0)  # TODO FOR NOW JUST CONCAT THEM!
#     flattened_conditions = torch.concat(conditions, dim=0)  # TODO FOR NOW JUST CONCAT THEM!

#     print(f"{flattened_samples.shape=}")
#     print(f"{flattened_conditions.shape=}")
#     dataset = TensorDataset(flattened_samples, flattened_conditions)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=params.batch_size,
#         shuffle=False,
#         # num_workers=workers
#     )

#     if latex_options:
#         (data_batch, conditions_batch) = next(iter(dataloader))
#         discriminator_input_size = batch_reshaper(data_batch, params.batch_size, params.sequence_len, features_len)[
#             0
#         ].size()
#         generator_input_size = noise_generator(params.batch_size, params, features_len)[0].size()
#         conditions_size = 1
#         print_net_summary(
#             G=G,
#             D=D,
#             generator_input_size=[generator_input_size, conditions_size],
#             discriminator_input_size=[discriminator_input_size, conditions_size],
#             latex_options=latex_options,
#             dtypes=[torch.FloatTensor, torch.IntTensor],
#         )

#     G_losses = []
#     D_losses = []
#     iters = 0

#     for epoch in (
#         progress := tqdm(
#             range(1, params.epochs + 1), unit="epochs", bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}"
#         )
#     ):
#         for batch_idx, (data_batch, conditions_batch) in enumerate(dataloader):
#             current_batch_size = min(params.batch_size, data_batch.shape[0])
#             data_batch = batch_reshaper(data_batch, current_batch_size, params.sequence_len, features_len)
#             # ADD NOISE
#             data_batch = data_batch + 0.01 * torch.randn(data_batch.shape, device=params.device)

#             ############################
#             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#             ###########################
#             real_labels = torch.ones(current_batch_size, requires_grad=False, device=params.device)  # try with 0.9
#             # real_labels = torch.squeeze(
#             #     torch.full((current_batch_size, 1), 0.9, requires_grad=False, device=params.device)
#             # )
#             fake_labels = torch.zeros(current_batch_size, requires_grad=False, device=params.device)

#             ## Train with all-real batch
#             D.zero_grad()
#             # label = torch.full((current_batch_size), real_label_value, dtype=torch.float, device=params.device)
#             d_out_real = D(data_batch, conditions_batch).view(-1)
#             # print(f"{d_out_real.shape=}")
#             d_err_real = criterion(d_out_real, real_labels)
#             d_err_real.backward()
#             D_x = d_err_real.mean().item()

#             ## Train with all-fake batch
#             # Generate batch of latent vectors
#             noise = noise_generator(current_batch_size, params, features_len)
#             fake_generated = G(noise, conditions_batch)
#             # print(f"{fake_generated.shape=}")
#             d_out_fake = D(fake_generated.detach(), conditions_batch).view(-1)  # TODO OTHER CONDITIONS?
#             d_err_fake = criterion(d_out_fake, fake_labels)
#             d_err_fake.backward()
#             D_G_z1 = d_out_fake.mean().item()
#             err_D = d_err_real + d_err_fake
#             # Update weights of D
#             optimizerD.step()

#             ############################
#             # (2) Update G network: maximize log(D(G(z)))
#             ###########################
#             G.zero_grad()
#             d_out_fake = D(fake_generated, conditions_batch).view(-1)
#             err_G = criterion(d_out_fake, real_labels)
#             err_G.backward()
#             D_G_z2 = d_out_fake.mean().item()
#             optimizerG.step()

#             if iters % 100 == 0:
#                 # padded_epoch = str(epoch).ljust(len(str(params.epochs)))
#                 # padded_batch_idx = str(batch_idx).ljust(len(str(len(dataloader))))
#                 progress_str = "Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f" % (
#                     err_D.item(),
#                     err_G.item(),
#                     D_x,
#                     D_G_z1,
#                     D_G_z2,
#                 )
#                 progress.set_description(progress_str)

#             # Save Losses for plotting later
#             G_losses.append(err_G.item())
#             D_losses.append(err_D.item())

#             iters += 1

#         if (epoch == 1) or (epoch % 10 == 0):
#             # cond_idx = 128
#             with torch.no_grad():
#                 # start_date: str = "2009.01.01"
#                 # end_date: str = "2109.12.31"
#                 # start = convert_input_str_to_date(start_date)
#                 # end = convert_input_str_to_date(end_date)
#                 # batch_conditions = torch.from_numpy(
#                 #     np.array([get_day_in_year_from_date(d) for d in interval_generator(start, end)])
#                 # )
#                 times = 100
#                 batch_conditions = torch.from_numpy(
#                     np.tile(np.concatenate((np.tile(np.arange(365), 3), np.arange(366))), times)
#                 )
#                 generated_sample_count = batch_conditions.size(0)
#                 batch_noise = noise_generator(generated_sample_count, params, features_len)
#                 # print(f"{batch_noise.shape=}")
#                 # print(f"{batch_conditions.shape=}")
#                 generated_data = G(batch_noise, batch_conditions)
#                 generated_data = generated_data.view(generated_sample_count, params.sequence_len, features_len)
#                 if data_holder.normalizer is not None:
#                     generated_data = data_holder.normalizer.renormalize(generated_data)
#                 generated_data = generated_data.cpu()  # We have to convert it to cpu too, to allow matplot to plot it
#                 # fig, ax = plt.subplots(nrows=1, ncols=1)
#                 # generated_data_seperated = torch.unbind(generated_data)
#                 generated_data_seperated = generated_data.view(
#                     features_len, generated_sample_count, params.sequence_len
#                 )
#                 # flattened_sample = torch.concat(unbind_sample)
#                 for feature_idx, (feature_data, feature_label) in enumerate(
#                     zip(generated_data_seperated, data_holder.get_feature_labels())
#                 ):
#                     label_str = feature_label.label if isinstance(feature_label, Feature) else feature_label
#                     label_str = label_str.replace("_", r"\_")
#                     # res = draw_hist_plot(
#                     #     pds=[HistPlotData(data=feature_data.view(-1).numpy(), label=label_str)],
#                     #     # bin_width=0.5,
#                     #     # normalized=True,
#                     # )

#                     # fig, ax = plt.subplots(nrows=1, ncols=1)
#                     # ax.hist(feature_data.view(-1).numpy(), label=label_str, density=False)
#                     data = PlotData(data=feature_data.view(-1).numpy(), label=label_str)
#                     res = draw_violin_plot(
#                         [data], PlotOptions(title="Violin plot opt", x_label=r"X\_lbl", y_label=r"Y\_lbl")
#                     )

#                     save_fig(
#                         res.fig,
#                         distributions_save_path
#                         / f"distribution_hist_epoch_{epoch}_cond_ALL_feature_{feature_idx}.{plots_file_ending}",
#                     )
#             with torch.no_grad():
#                 winter_start = convert_input_str_to_date("2009.01.01")
#                 winter_end = convert_input_str_to_date("2009.01.08")
#                 summer_start = convert_input_str_to_date("2009.08.01")
#                 summer_end = convert_input_str_to_date("2009.08.08")
#                 winter_batch_conditions = torch.from_numpy(
#                     np.array([get_day_in_year_from_date(d) for d in interval_generator(winter_start, winter_end)])
#                 )
#                 summer_batch_conditions = torch.from_numpy(
#                     np.array([get_day_in_year_from_date(d) for d in interval_generator(summer_start, summer_end)])
#                 )
#                 batch_conditions = torch.cat((winter_batch_conditions, summer_batch_conditions), 0)
#                 generated_sample_count = batch_conditions.size(0)
#                 batch_noise = noise_generator(generated_sample_count, params, features_len)
#                 generated_data = G(batch_noise, batch_conditions)
#                 generated_data = generated_data.view(generated_sample_count, params.sequence_len, features_len)
#                 if data_holder.normalizer is not None:
#                     generated_data = data_holder.normalizer.renormalize(generated_data)
#                 fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 1))
#                 for idx, (fig, ax) in enumerate(
#                     plot_sample(sample=generated_data, params=params, plot=(fig, ax), condition="ALL")
#                 ):
#                     save_fig(fig, sample_save_path / f"{epoch}_feat_{idx}_cond_ALL.{plots_file_ending}")

#                 # TODO PRINT SINGLE FEATURES
#                 # for i, y in enumerate(torch.transpose(flattened_sample, 0, 1)):
#                 #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 1))
#                 #     x = range(len(y))
#                 #     ax.plot(x, y, label=r"$f_{" + str(i) + r"}^{t}$")
#                 #     save_fig(fig, sample_save_path / f"{epoch}_cond_ALL.{plots_file_ending}")

#             # with torch.no_grad():
#             #     generated_sample_count = 100
#             #     batch_noise = noise_generator(generated_sample_count, params, features_len)
#             #     batch_conditions = torch.squeeze(torch.full((generated_sample_count, 1), cond_idx))
#             #     generated_data = G(batch_noise, batch_conditions)
#             #     generated_data = generated_data.view(generated_sample_count, params.sequence_len, features_len)
#             #     if data_holder.normalizer is not None:
#             #         generated_data = data_holder.normalizer.renormalize(generated_data)
#             #     for feature_idx, (fig, ax) in enumerate(
#             #         plot_box_plot_per_ts(
#             #             data=generated_data, epoch=epoch, samples=samples, params=params, condition=cond_idx
#             #         )
#             #     ):
#             #         save_fig(
#             #             fig,
#             #             distributions_save_path
#             #             / f"distribution_epoch_{epoch}_cond_{cond_idx}_feature_{feature_idx}.{plots_file_ending}",
#             #         )

#             fig, ax = plot_model_losses(g_losses=G_losses, d_losses=D_losses, current_epoch=epoch)
#             save_fig(fig, loss_save_path / f"losses_after_{epoch}.{plots_file_ending}")
#             plt.close(fig)
#     print("End training\n--------------------------------------------")


def create_plot_sample_fn(sample_save_path: PurePath, plots_file_ending: str = "pdf"):
    def plot_sample_range(epoch: TrainingEpoch, trainer: CGANTrainer):
        winter_start = convert_input_str_to_date("2009.01.01")
        winter_end = convert_input_str_to_date("2009.01.08")
        summer_start = convert_input_str_to_date("2009.08.01")
        summer_end = convert_input_str_to_date("2009.08.08")
        winter_batch_conditions = torch.from_numpy(
            np.array([get_day_in_year_from_date(d) for d in interval_generator(winter_start, winter_end)])
        )
        summer_batch_conditions = torch.from_numpy(
            np.array([get_day_in_year_from_date(d) for d in interval_generator(summer_start, summer_end)])
        )
        batch_conditions = torch.cat((winter_batch_conditions, summer_batch_conditions), 0)
        generated_sample_count = batch_conditions.size(0)
        batch_noise = trainer.noise_generator(generated_sample_count, trainer.params)
        generated_data = trainer.generator.model(batch_noise, batch_conditions)
        generated_data = generated_data.view(
            generated_sample_count, trainer.params.sequence_len, trainer.params.features_len
        )
        if trainer.data_holder.normalizer is not None:
            generated_data = trainer.data_holder.normalizer.renormalize(generated_data)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 1))
        sample_generator = plot_sample(
            sample=generated_data, params=trainer.params, plot=(fig, ax), condition="ALL", generate_single_features=True
        )
        for idx, (fig, ax) in enumerate(sample_generator):
            save_fig(fig, sample_save_path / f"{epoch}_feat_{idx}_cond_ALL.{plots_file_ending}")

    return plot_sample_range


def create_plot_loss_fn(loss_save_path: PurePath, plots_file_ending: str = "pdf"):
    def plot_loss(epoch: TrainingEpoch, trainer: CGANTrainer):
        fig, ax = plot_model_losses(g_losses=trainer.gen_losses, d_losses=trainer.dis_losses, current_epoch=epoch)
        save_fig(fig, loss_save_path / f"losses_after_{epoch}.{plots_file_ending}")

    return plot_loss


def create_save_model_fn(model_root_save_path: PurePath):
    def save_model(epoch: TrainingEpoch, trainer: CGANTrainer):
        model_save_path = model_root_save_path / str(epoch)
        model_save_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_save_path)

    return save_model


def setup_fnn_models_and_train(
    data_holder: DataHolder,
    save_path: PurePath,
    latex_options: Optional[LatexTableOptions],
    epochs: int,
):
    # Learning rate for optimizers
    lr = 1e-3
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.9
    beta2 = 0.99
    conditions = 366  #
    dropout = 0.3
    features_len = data_holder.get_feature_size()
    params = ConditionalTrainParameters(
        batch_size=32, embedding_dim=int(24 * 1.5 * features_len), features_len=features_len
    )

    model_G = GeneratorFNN(
        latent_vector_size=params.latent_vector_size,
        features=features_len,
        sequence_len=params.sequence_len,
        dropout=dropout,
        conditions=conditions,
        embeddings=params.embedding_dim,
    )
    optimizer_G = optim.Adam(model_G.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.5)
    generator = TrainModel(model=model_G, optimizer=optimizer_G, scheduler=scheduler_G)

    model_D = DiscriminatorFNN(
        features=features_len,
        sequence_len=params.sequence_len,
        conditions=conditions,
        embeddings=params.embedding_dim,
        out_features=1,
        dropout=dropout,
    )
    optimizer_D = optim.Adam(model_D.parameters(), lr=lr, betas=(beta1, beta2))
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.5)
    discriminator = TrainModel(model=model_D, optimizer=optimizer_D, scheduler=scheduler_D)
    init_weights(model_D, "xavier", init_gain=nn.init.calculate_gain("leaky_relu", 1e-2))

    # init_weights(D, "xavier", init_gain=nn.init.calculate_gain("leaky_relu"))

    save_path.mkdir(parents=True, exist_ok=True)
    loss_save_path = save_path / "losses"
    loss_save_path.mkdir(parents=True, exist_ok=True)
    distributions_save_path = save_path / "distributions"
    distributions_save_path.mkdir(parents=True, exist_ok=True)
    sample_save_path = save_path / "sample"
    sample_save_path.mkdir(parents=True, exist_ok=True)
    model_root_save_path = save_path / "models"
    model_root_save_path.mkdir(parents=True, exist_ok=True)

    trainer = CGANTrainer(
        data_holder=data_holder,
        generator=generator,
        discriminator=discriminator,
        # batch_reshaper=fnn_batch_reshaper,
        # noise_generator=fnn_noise_generator,
        params=params,
        callback_options=[
            (
                lambda epoch, _: (epoch == 1) or (epoch % 10 == 0),
                [
                    create_plot_loss_fn(loss_save_path),
                    create_plot_sample_fn(sample_save_path),
                    create_save_model_fn(model_root_save_path),
                ],
            )
        ],
        latex_options=latex_options,
    )
    trainer.train(epochs)
    return trainer


def train_all_features(data_importer: DWDWeatherDataImporter, epochs: int):
    """
    Multivariate for conditional GAN with FNN
    """
    conditions = np.array(data_importer.get_day_of_year_values()) - 1  # days to conditions from 0 - 365
    columns_to_use = set(
        [
            WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
            WeatherDataColumns.DH_W_PER_M2,
            WeatherDataColumns.GH_W_PER_M2,
            # WeatherDataColumns.WIND_DIR_DEGREE,
            # WeatherDataColumns.WIND_DIR_DEGREE_DELTA,
        ]
    )
    data_holder = DataHolder(
        data=data_importer.get_data_subset(columns_to_use).values.astype(np.float32),
        # data_labels=data_importer.get_feature_labels(),
        dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
        conditions=conditions,
        normalizer_constructor=StandardNumpyNormalizer,
    )
    latex_options = LatexTableOptions(
        caption="Conditional GAN {net_type}-Netz in Form eines mehrschichtigen Perzeptrons für Simulationsdaten",
        label="conditional_gan_fnn_sines_net_simulation_data_{net_type}",
        # style=LatexTableStyle(scaleWithAdjustbox=1.0),
    )

    setup_fnn_models_and_train(
        data_holder=data_holder,
        save_path=save_images_path / "fnn_all_features",
        latex_options=latex_options,
        epochs=epochs,
    )


def train():
    manualSeed = 1337
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    # rng = np.random.default_rng(seed=0) # use for numpy
    torch.manual_seed(manualSeed)

    sns.set_theme()
    sns.set_context("paper")
    # sns.set_palette("colorblind")
    set_latex_plot_params()

    start_date_str = DEFAULT_DATA_START_DATE
    end_date_str = "2019-12-31 23:00:00"  # "2019-12-31 23:00:00"
    data_importer = DWDWeatherDataImporter(start_date=start_date_str, end_date=end_date_str)
    data_importer.initialize()
    epochs = 4000

    train_all_features(data_importer=data_importer, epochs=epochs)


def eval(epoch):
    manualSeed = 1337
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    set_latex_plot_params()

    model_path = save_images_path / "fnn_all_features" / "models" / str(epoch)
    evaluator = Evaluator.load(str(model_path))
    start = datetime.fromisoformat("2023-01-01T00:00:00")
    end = datetime.fromisoformat("2023-12-31T23:00:00")
    generated_data = evaluator.generate(start, end)
    # train_params = ConditionalTrainParameters(batch_size=32, embedding_dim=32)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 1))
    # (fig, ax) = plot_sample(
    #     sample=generated_data, params=train_params, plot=(fig, ax), condition="ALL", generate_single_features=False
    # )

    seperated_generated_data = torch.unbind(generated_data, -1)
    dataframe = pd.DataFrame(
        index=pd.date_range(
            start=start,
            end=end,
            tz="Europe/Berlin",
            freq="h",
        )
    )

    for idx, data in enumerate(seperated_generated_data):
        dataframe[evaluator.feature_labels[idx].label] = data.view(-1).numpy()

    weather_post_processor = DWDWeatherPostProcessor()
    dataframe = weather_post_processor(dataframe)
    plot_options: dict[WeatherDataColumns, any] = {
        WeatherDataColumns.GH_W_PER_M2: {"zoom_plot_label": r"Globalstrahlung $\frac{W}{m^{2}}$"},
        WeatherDataColumns.DH_W_PER_M2: {"zoom_plot_label": r"Diffusstrahlung $\frac{W}{m^{2}}$"},
        WeatherDataColumns.WIND_DIR_DEGREE: {"zoom_plot_label": r"Windrichtung $^{\circ}$"},
        WeatherDataColumns.WIND_DIR_DEGREE_DELTA: {"zoom_plot_label": r"Windrichtungsänderung $^{\circ}$"},
        WeatherDataColumns.T_AIR_DEGREE_CELSIUS: {"zoom_plot_label": r"Temperatur $^{\circ}C$"},
    }
    dates = np.array([d for d in interval_generator(start, end, delta=timedelta(hours=1))])
    generated_labels = list(map(lambda feature: feature.label, evaluator.feature_labels))
    print(f"{generated_labels=}")
    plot_data = [
        PlotData(data=dataframe[col].values, label=plot_option["zoom_plot_label"])
        for col, plot_option in plot_options.items()
        if col in generated_labels
    ]
    print(f"{plot_data=}")
    fig, axes = draw_zoom_line_plot(
        raw_plot_data=plot_data,
        x=dates,
        zoom_boxes_options=[
            ZoomPlotOptions(
                x_start=datetime.fromisoformat("2023-01-01T00:00:00"),
                x_end=datetime.fromisoformat("2023-01-07T23:00:00"),
                effect_options=ZoomBoxEffectOptions(source_connector_box_options=ConnectorBoxOptions()),
            ),
            ZoomPlotOptions(
                x_start=datetime.fromisoformat("2023-08-01T00:00:00"),
                x_end=datetime.fromisoformat("2023-08-07T23:00:00"),
            ),
        ],
    )

    plt.show()


if __name__ == "__main__":
    # train()
    eval(epoch=450)
