import abc
import shutil
from pathlib import Path
from typing import Union

import torch

from src.constants import GENERATOR_MODEL_NAME, GENERATOR_NORMALIZER_NAME
from src.data.dataholder import DataHolder
from src.data.normalization.base_normalizer import BaseNormalizer
from src.gan.trainer.trainer_types import TrainModel


class BaseTrainer:
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 generator: TrainModel,
                 discriminator: TrainModel,
                 data_holder: DataHolder,
                 device: Union[torch.device, int, str] = 'cpu'
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.data_holder = data_holder
        self.device = device

    @abc.abstractmethod
    def train(self, max_epochs) -> None:
        raise NotImplementedError("Please implement this method.")

    def save_model(self, path: Union[str, Path], overwrite: bool = True):
        print("Starting to save the trained model.")
        p = Path(path) if isinstance(path, str) else path
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

        model_path = p / GENERATOR_MODEL_NAME
        normalizer_path = p / GENERATOR_NORMALIZER_NAME

        if model_path.exists():
            if overwrite is False:
                raise ValueError("You have specified a directory which already contains a saved model. Allow to overwrite it or specify another folder!")
            else:
                # keep a copy of an old training run in a backup folder
                model_file_stats = model_path.stat()
                backup_dir = p / str(model_file_stats.st_ctime_ns)
                print(f"Already found a saved model in this directory, creating a backup of the old one. Under: {backup_dir}")
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(model_path, backup_dir)
                if normalizer_path.exists():
                    shutil.move(normalizer_path, backup_dir)

        # TODO JSON-PICKLE
        # torch.save(self.generator.model, model_path.absolute())
        m = torch.jit.script(self.generator.model)
        m.save(model_path.absolute())
        print(f"Saved the trained model under: {model_path}")
        if self.data_holder.normalizer is not None:
            BaseNormalizer.save(self.data_holder.normalizer, normalizer_path.absolute())
            print(f"Saved the corresponding normalizer model under: {normalizer_path}")




