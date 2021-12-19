from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import torch

from torch.optim.lr_scheduler import StepLR

from evaluator.evaluator import Evaluator
from metrics.timeseries_decomposition import decompose_weather_data
from src.data.weather.weather_dwd_importer import DWDWeatherDataImporter
from src.data.data_holder import DataHolder
from src.gan.trainer.cgan_trainer import CGANBasicGenerator, CGANBasicDiscriminator, CGANTrainer
from src.gan.trainer.typing import TrainModel
from src.utils.args_utils import DataclassArgumentParser, Choice, Int, Arg, Str, Float
from src.utils.datetime_utils import dates_to_conditional_vectors, convert_input_str_to_date
from src.utils.path_utils import get_root_project_path
from utils.plot_utils import plot_dfs

DEFAULT_SAVE_LOAD_DIRECTORY = get_root_project_path().joinpath('runs').joinpath('model-test').absolute()


class Datasets(Enum):
    SINE = 'sine'
    WEATHER = 'weather'


class Nets(Enum):
    CGAN = 'cgan'
    VANILLA = 'vanilla'


class Mode(Enum):
    TRAIN = 'train'
    EVAL = 'eval'


@dataclass(frozen=True)
class PGGANArgs:
    """Arguments for the GAN framework"""
    mode: Choice[Mode] = Mode.TRAIN
    batch_size: Int(help="Number of batches") = 24
    noise_vector_size: Int(help="Size of the noise vector") = 50
    epochs: Int = 100
    lr: Float(help="Learning rate for optimizers") = 0.003
    beta1: Float(help="Beta1 hyperparameter for the Adam optimizers") = 0.9
    device: Arg(type=torch.device) = torch.device('cpu')
    dataset: Choice[Datasets] = Datasets.WEATHER
    model: Choice[Nets] = Nets.CGAN
    save_path: Str = DEFAULT_SAVE_LOAD_DIRECTORY
    start_date: Str = '2009.01.01'
    end_date: Str = '2019.12.31'
    load_path: Str = DEFAULT_SAVE_LOAD_DIRECTORY
    seed: Int = 1337


def get_data_holder(dataset: Choice[Datasets]) -> DataHolder:
    if dataset is Datasets.WEATHER:
        data_importer = DWDWeatherDataImporter()
        data_importer.initialize()
        return DataHolder(data_importer.data.values.astype(np.float32), data_importer.get_feature_labels(),
                          np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())))
    else:
        raise ValueError("Not supported yet!")


def get_cgan_trainer(features, noise_vector_size, batch_size, lr, beta1):
    G_net = CGANBasicGenerator(input_size=noise_vector_size + 14, out_size=features, hidden_layers=[200])
    G_optim = torch.optim.Adam(G_net.parameters(), lr=lr, betas=(beta1, 0.999))
    G_sched = StepLR(G_optim, step_size=30, gamma=0.1)
    G = TrainModel(G_net, G_optim, G_sched)

    D_net = CGANBasicDiscriminator(input_size=features + 14, out_size=1, hidden_layers=[100, 50, 20])
    D_optim = torch.optim.Adam(D_net.parameters(), lr=lr, betas=(beta1, 0.999))
    D_sched = StepLR(D_optim, step_size=30, gamma=0.1)
    D = TrainModel(D_net, D_optim, D_sched)

    trainer = CGANTrainer(G, D, data_holder, noise_vector_size, batch_size, 'cpu')

    return trainer


if __name__ == '__main__':
    args = dataclassArgsParser = DataclassArgumentParser(container_class=PGGANArgs).parse_args()
    print("args:", args)

    if args.mode is Mode.TRAIN:
        data_holder = get_data_holder(args.dataset)
        trainer = None
        if args.model is Nets.CGAN:
            trainer = get_cgan_trainer(
                features=data_holder.get_feature_size(),
                noise_vector_size=args.noise_vector_size,
                batch_size=args.batch_size,
                lr=args.lr,
                beta1=args.lr
            )
        else:
            raise ValueError("Not supported yet!")

        trainer.train(args.epochs)
        trainer.save_model(str(args.save_path))
    elif args.mode is Mode.EVAL:
        evaluator = Evaluator.load(str(args.load_path))
        start = convert_input_str_to_date(str(args.start_date))
        end = convert_input_str_to_date(str(args.end_date))
        generated_data = evaluator.generate(start, end).numpy().transpose()

        print(f"{generated_data.shape=}")
        data = pd.DataFrame(
            index=pd.date_range(
                start=datetime(start.year, start.month, start.day),
                end=datetime(end.year, end.month, end.day, 23, 59, 59),
                tz="Europe/Berlin",
                freq="H",
            )
        )
        for i in range(generated_data.shape[0]):
            data[evaluator.feature_labels[i].label] = generated_data[i]
            decompose_weather_data(data[evaluator.feature_labels[i].label]).plot()

        plot_dfs([data])

    else:
        raise ValueError(f"Mode: {args.mode} is not supported.")
