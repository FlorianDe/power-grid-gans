from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.jit import ScriptModule
from torch.nn import Module

from src.constants import GENERATOR_MODEL_FILE_NAME, GENERATOR_NORMALIZER_FILE_NAME, GENERATOR_FEATURE_LABELS_FILE_NAME
from src.data.normalization.base_normalizer import BaseNormalizer
from src.data.serializer.csv_serializer import CsvSerializer
from src.data.typing import Feature
from src.utils.datetime_utils import dates_to_conditional_vectors, get_day_in_year_from_date, interval_generator

from src.utils.path_utils import get_root_project_path


class Evaluator:
    def __init__(
        self,
        model: ScriptModule | Module,
        latent_vector_size: int,
        feature_labels: list[Feature],
        normalizer: Optional[BaseNormalizer] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.model = deepcopy(model).to(device)
        self.latent_vector_size = latent_vector_size
        self.feature_labels = deepcopy(feature_labels)
        self.normalizer = deepcopy(normalizer)
        self.device = device
        self.model.eval()  # Set the model to "eval" mode, alias for model.train(mode=False)

    def eval(self, z: torch.tensor, c: Optional[torch.tensor]) -> Tensor:
        with torch.no_grad():
            y = self.model(z) if c is None else self.model(z, c)
            # reshape, generator should know this
            y = y.view(z.size(0), 24, len(self.feature_labels))
            if self.normalizer is not None:
                y = self.normalizer.renormalize(y)
            return y

    def generate(self, start_date: datetime, end_date: datetime, with_conditions=True) -> Tensor:
        with torch.no_grad():
            # outsource condition generation maybe
            batch_conditions = torch.from_numpy(
                np.array([get_day_in_year_from_date(d) for d in interval_generator(start_date, end_date)])
            )

            generated_sample_count = batch_conditions.size(0)
            batch_noise = torch.randn(generated_sample_count, self.latent_vector_size, device=self.device)

            return self.eval(batch_noise, batch_conditions if with_conditions else None)

            # result: torch.Tensor | None = None
            # batch_size = 24
            # rnd_vector = np.random.normal(0, 1, (1, len(self.feature_labels), self.noise_vector_size))
            # noises = torch.from_numpy(np.repeat(rnd_vector, batch_size, axis=0).astype(dtype=np.float32))
            # for d in interval_generator(start_date, end_date):
            #     generator_input = noises
            #     # TODO GENERIFY LATER
            #     if with_conditions:
            #         months = np.repeat(d.month, batch_size)
            #         days = np.repeat(d.day, batch_size)
            #         hours = np.arange(batch_size)
            #         conditions = torch.tensor(dates_to_conditional_vectors(months, days, hours), dtype=torch.float32, requires_grad=False)
            #         generator_input = torch.cat((noises, conditions), -1)
            #     current_res = self.model(generator_input)
            #     if result is None:
            #         result = current_res
            #     else:
            #         result = torch.cat((result, current_res), dim=-1)

    def generate_dataframe(self, start_date: datetime, end_date: datetime, with_conditions=True) -> pd.DataFrame:
        generated_data = self.generate(start_date, end_date, with_conditions)
        seperated_generated_data = torch.unbind(generated_data, -1)
        dataframe = pd.DataFrame(
            index=pd.date_range(
                start=start_date,
                end=end_date,
                tz="Europe/Berlin",
                freq="h",
            )
        )

        for idx, data in enumerate(seperated_generated_data):
            dataframe[self.feature_labels[idx].label] = data.view(-1).numpy()

        return dataframe

    @staticmethod
    def load(path: Union[str, Path], device: torch.device = torch.device("cpu")) -> Evaluator:
        p = Path(path) if isinstance(path, str) else path
        if not p.exists():
            raise ValueError("The path you've specified does not exist!")

        if not p.is_dir() and p.is_file():
            p = p.parent

        model_path = p / GENERATOR_MODEL_FILE_NAME
        normalizer_path = p / GENERATOR_NORMALIZER_FILE_NAME
        feature_labels_file_path = p / GENERATOR_FEATURE_LABELS_FILE_NAME

        if not model_path.exists():
            raise ValueError(
                f"The path/file you've specified does not contain a valid model file called {GENERATOR_MODEL_FILE_NAME}!"
            )

        # model = torch.load(model_path.absolute())
        model = torch.jit.load(model_path.absolute())

        feature_labels = CsvSerializer.load(feature_labels_file_path, Feature)

        normalizer = None
        if normalizer_path.exists():
            normalizer = BaseNormalizer.load(normalizer_path.absolute())

        return Evaluator(
            model=model, latent_vector_size=100, feature_labels=feature_labels, normalizer=normalizer  # load meeee
        )


if __name__ == "__main__":
    path = get_root_project_path().joinpath("runs").joinpath("model-test").absolute()
    evaluator = Evaluator.load(path)

    print(f"{evaluator.model=}")
    print(evaluator.model.code)
    print(f"{evaluator.normalizer=}")

    evaluator.model.eval()

    noise_vector_size = 50
    batch_size = 24

    noises = torch.from_numpy(
        np.repeat(np.random.normal(0, 1, (1, noise_vector_size)), batch_size, axis=0).astype(dtype=np.float32)
    )

    # Generate a batch for the Date a 01.01. between 0 and 23 o'clock
    months = np.repeat(1, batch_size)  # np.random.randint(1, 12, batch_size)
    days = np.repeat(1, batch_size)  # np.random.randint(1, 31, batch_size)
    hours = np.arange(24)  # np.random.randint(0, 23, batch_size)
    conditions = torch.tensor(
        dates_to_conditional_vectors(months, days, hours), dtype=torch.float32, requires_grad=False
    )

    y = evaluator.model(noises, conditions)
    print(y)
