from __future__ import annotations

from datetime import date
from pathlib import Path

from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.jit import ScriptModule
from torch.nn import Module

from constants import GENERATOR_MODEL_FILE_NAME, GENERATOR_NORMALIZER_FILE_NAME
from data.normalization.base_normalizer import BaseNormalizer
from utils.datetime_utils import dates_to_conditional_vectors, interval_generator

from utils.path_utils import get_root_project_path


class Evaluator:

    def __init__(self, model: ScriptModule | Module, normalizer: Optional[BaseNormalizer] = None) -> None:
        super().__init__()
        self.model: ScriptModule = model
        self.normalizer = normalizer

        self.model.eval()  # Set the model to "eval" mode, alias for model.train(mode=False)

    def eval(self, x):
        with torch.no_grad():
            y = self.model(x)
            if self.normalizer is not None:
                y = self.normalizer.renormalize(y)
            return y

    def generate(self, start_date: date, end_date: date) -> Tensor:
        with torch.no_grad():
            result: torch.Tensor | None = None
            noise_vector_size = 50
            batch_size = 24
            noises = torch.from_numpy(np.repeat(np.random.normal(0, 1, (1, noise_vector_size)), batch_size, axis=0).astype(dtype=np.float32))
            for d in interval_generator(start_date, end_date):
                months = np.repeat(d.month, batch_size)
                days = np.repeat(d.day, batch_size)
                hours = np.arange(batch_size)
                conditions = torch.tensor(dates_to_conditional_vectors(months, days, hours), dtype=torch.float32, requires_grad=False)
                current_res = self.model(noises, conditions)
                if result is None:
                    result = current_res
                else:
                    result = torch.cat((result, current_res))
            return result


    @staticmethod
    def load(path: Union[str, Path]) -> Evaluator:
        p = Path(path) if isinstance(path, str) else path
        if not p.exists():
            raise ValueError("The path you've specified does not exist!")

        if not p.is_dir() and p.is_file():
            p = p.parent

        model_path = p / GENERATOR_MODEL_FILE_NAME
        normalizer_path = p / GENERATOR_NORMALIZER_FILE_NAME

        if not model_path.exists():
            raise ValueError(f"The path/file you've specified does not contain a valid model file called {GENERATOR_MODEL_FILE_NAME}!")

        # model = torch.load(model_path.absolute())
        model = torch.jit.load(model_path.absolute())
        normalizer = None

        if normalizer_path.exists():
            normalizer = BaseNormalizer.load(normalizer_path.absolute())

        return Evaluator(model, normalizer)


if __name__ == '__main__':
    path = get_root_project_path().joinpath('runs').joinpath('model-test').absolute()
    evaluator = Evaluator.load(path)

    print(f"{evaluator.model=}")
    print(evaluator.model.code)
    print(f"{evaluator.normalizer=}")

    evaluator.model.eval()

    noise_vector_size = 50
    batch_size = 24

    noises = torch.from_numpy(np.repeat(np.random.normal(0, 1, (1, noise_vector_size)), batch_size, axis=0).astype(dtype=np.float32))

    # Generate a batch for the Date a 01.01. between 0 and 23 o'clock
    months = np.repeat(1, batch_size) # np.random.randint(1, 12, batch_size)
    days = np.repeat(1, batch_size) # np.random.randint(1, 31, batch_size)
    hours = np.arange(24) # np.random.randint(0, 23, batch_size)
    conditions = torch.tensor(dates_to_conditional_vectors(months, days, hours), dtype=torch.float32, requires_grad=False)

    y = evaluator.model(noises, conditions)
    print(y)
