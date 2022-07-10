from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR

from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer
from src.data.weather.weather_dwd_importer import (
    DWDWeatherDataImporter,
    WeatherDataColumns,
    DEFAULT_DATA_START_DATE,
    DEFAULT_DATA_END_DATE,
)
from src.data.weather.weather_dwd_postprocessor import DWDWeatherPostProcessor
from src.data.data_holder import DataHolder
from src.evaluator.evaluator import Evaluator
from src.gan.trainer.cgan_trainer import CGANTrainer, DiscriminatorFNN, GeneratorFNN
from src.gan.trainer.typing import ConditionalTrainParameters, TrainModel
from src.net.weight_init import init_weights
from src.utils.args_utils import DataclassArgumentParser, Choice, Int, Arg, Str, Float
from src.utils.path_utils import get_root_project_path


DEFAULT_SAVE_LOAD_DIRECTORY = get_root_project_path().joinpath("runs").joinpath("model-test").absolute()


class Datasets(Enum):
    SINE = "sine"
    WEATHER = "weather"


class Nets(Enum):
    CGAN = "cgan"
    VANILLA = "vanilla"


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass(frozen=True)
class PGGANArgs:
    """Arguments for the GAN framework"""

    mode: Choice[Mode] = Mode.TRAIN
    batch_size: Int(help="Number of batches") = 24
    noise_vector_size: Int(help="Size of the noise vector") = 50
    epochs: Int = 500
    lr: Float(help="Learning rate for optimizers") = 0.003
    beta1: Float(help="Beta1 hyperparameter for the Adam optimizers") = 0.9
    beta2: Float(help="Beta2 hyperparameter for the Adam optimizers") = 0.99
    device: Arg(type=torch.device) = torch.device("cpu")
    dataset: Choice[Datasets] = Datasets.WEATHER
    model: Choice[Nets] = Nets.CGAN
    save_path: Str = DEFAULT_SAVE_LOAD_DIRECTORY
    start_date: Str = DEFAULT_DATA_START_DATE
    end_date: Str = DEFAULT_DATA_END_DATE
    load_path: Str = DEFAULT_SAVE_LOAD_DIRECTORY
    results_path: Str = DEFAULT_SAVE_LOAD_DIRECTORY / "generated_data"
    seed: Int = 1337


def get_data_holder(args: PGGANArgs) -> DataHolder:
    if args.dataset is Datasets.WEATHER:
        data_importer = DWDWeatherDataImporter(start_date=args.start_date, end_date=args.end_date)
        data_importer.initialize()
        columns_to_use = set(
            [
                WeatherDataColumns.GH_W_PER_M2,
                WeatherDataColumns.DH_W_PER_M2,
                WeatherDataColumns.WIND_DIR_DEGREE,
                WeatherDataColumns.WIND_V_M_PER_S,
                WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
            ]
        )
        conditions = np.array(data_importer.get_day_of_year_values()) - 1  # days to conditions from 0 - 365
        data_subset = data_importer.get_data_subset(columns_to_use)
        return DataHolder(
            data=data_subset.values.astype(np.float32),
            data_labels=data_subset.columns.to_list(),
            dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
            conditions=conditions,
            normalizer_constructor=StandardNumpyNormalizer,
        )
    else:
        raise ValueError("Not supported yet!")


def get_cgan_trainer(data_holder: DataHolder, args: PGGANArgs):
    conditions = 366
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
    optimizer_G = optim.Adam(model_G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
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
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.5)
    discriminator = TrainModel(model=model_D, optimizer=optimizer_D, scheduler=scheduler_D)
    init_weights(model_D, "xavier", init_gain=nn.init.calculate_gain("leaky_relu", 1e-2))

    return CGANTrainer(
        data_holder=data_holder,
        generator=generator,
        discriminator=discriminator,
        # batch_reshaper=fnn_batch_reshaper,
        # noise_generator=fnn_noise_generator,
        params=params,
        # callback_options=[],
        # latex_options=latex_options,
    )


if __name__ == "__main__":
    args = dataclassArgsParser = DataclassArgumentParser(container_class=PGGANArgs).parse_args()
    print("args:", args)

    if args.mode is Mode.TRAIN:
        data_holder = get_data_holder(args)
        trainer = None
        if args.model is Nets.CGAN:
            trainer = get_cgan_trainer(data_holder, args)
        else:
            raise ValueError("Not supported yet!")

        trainer.train(args.epochs)
        trainer.save_model(str(args.save_path))
    elif args.mode is Mode.EVAL:
        evaluator = Evaluator.load(str(args.load_path))
        start = datetime.fromisoformat("2020-01-01T00:00:00")
        end = datetime.fromisoformat("2020-12-31T23:00:00")
        weather_post_processor = DWDWeatherPostProcessor()
        dataframe = evaluator.generate_dataframe(start, end)
        if args.dataset is Datasets.WEATHER:
            dataframe = weather_post_processor(dataframe)
        Path(args.results_path).mkdir(parents=True, exist_ok=True)
        dataframe.to_hdf(args.results_path / "result_generated_data.hdf", "weather", "w")

    else:
        raise ValueError(f"Mode: {args.mode} is not supported.")
