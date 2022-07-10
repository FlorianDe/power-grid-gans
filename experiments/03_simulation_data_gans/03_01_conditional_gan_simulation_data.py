from pathlib import PurePath
from typing import Optional
import numpy as np
import seaborn as sns

import random
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params

from experiments.experiments_utils.plotting import (
    plot_model_losses,
    plot_sample,
    save_fig,
)
from src.gan.trainer.cgan_trainer import CGANTrainer, DiscriminatorFNN, GeneratorFNN
from src.gan.trainer.typing import ConditionalTrainParameters, TrainModel, TrainingEpoch
from src.data.data_holder import DataHolder
from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer
from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns

from src.net.summary.net_summary import LatexTableOptions, LatexTableStyle
from src.net.weight_init import init_weights
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


def train_features(data_importer: DWDWeatherDataImporter, columns: set[WeatherDataColumns], epochs: int, path: str):
    """
    Multivariate for conditional GAN with FNN
    """
    conditions = np.array(data_importer.get_day_of_year_values()) - 1  # days to conditions from 0 - 365
    data_subset = data_importer.get_data_subset(columns)
    data_holder = DataHolder(
        data=data_subset.values.astype(np.float32),
        data_labels=data_subset.columns.to_list(),
        dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
        conditions=conditions,
        normalizer_constructor=StandardNumpyNormalizer,
    )
    latex_options = LatexTableOptions(
        caption="Conditional GAN {net_type}-Netz in Form eines mehrschichtigen Perzeptrons für Simulationsdaten",
        label="conditional_gan_fnn_net_simulation_data_{net_type}",
        style=LatexTableStyle(scaleWithAdjustbox=1.0),
    )

    setup_fnn_models_and_train(
        data_holder=data_holder,
        save_path=save_images_path / path,
        latex_options=latex_options,
        epochs=epochs,
    )


def train_sun_temp_features(data_importer: DWDWeatherDataImporter, epochs: int):
    columns_to_use = set(
        [
            WeatherDataColumns.GH_W_PER_M2,
            WeatherDataColumns.DH_W_PER_M2,
            WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
        ]
    )

    train_features(data_importer=data_importer, columns=columns_to_use, epochs=epochs, path="fnn_features_sun_temp")


def train_all_features(data_importer: DWDWeatherDataImporter, epochs: int):
    columns_to_use = set(
        [
            WeatherDataColumns.GH_W_PER_M2,
            WeatherDataColumns.DH_W_PER_M2,
            WeatherDataColumns.WIND_DIR_DEGREE,
            WeatherDataColumns.WIND_V_M_PER_S,
            WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
        ]
    )
    train_features(data_importer=data_importer, columns=columns_to_use, epochs=epochs, path="fnn_features_all")


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
    # train_sun_temp_features(data_importer=data_importer, epochs=epochs)


if __name__ == "__main__":
    train()
