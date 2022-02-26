import abc
import shutil
from pathlib import Path
from typing import Union

import torch

from src.data.serializer.csv_serializer import CsvSerializer
from src.data.typing import Feature
from src.constants import GENERATOR_MODEL_FILE_NAME, GENERATOR_NORMALIZER_FILE_NAME, GENERATOR_FEATURE_LABELS_FILE_NAME
from src.data.data_holder import DataHolder
from src.data.normalization.base_normalizer import BaseNormalizer
from src.gan.trainer.typing import TrainModel
from src.utils.datetime_utils import format_timestamp


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
        raise NotImplementedError("Please implement this method in your custom Trainer!")

    def save_model(self, path: Union[str, Path], overwrite: bool = True):
        print("Starting to save the trained model.")
        p = Path(path) if isinstance(path, str) else path
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

        model_file_path = p / GENERATOR_MODEL_FILE_NAME
        feature_labels_file_path = p / GENERATOR_FEATURE_LABELS_FILE_NAME
        normalizer_file_path = p / GENERATOR_NORMALIZER_FILE_NAME
        backup_files = [
            model_file_path,
            feature_labels_file_path,
            normalizer_file_path,
        ]

        if model_file_path.exists():
            if overwrite is False:
                raise ValueError("You have specified a directory which already contains a saved model. Allow to overwrite it or specify another folder!")
            else:
                # keep a copy of an old training run in a backup folder
                model_file_stats = model_file_path.stat()
                backup_folder_name = format_timestamp(model_file_stats.st_ctime_ns)
                backup_dir = p / backup_folder_name
                print(f"Already found a saved model in this directory, creating a backup of the old one. Under: {backup_dir}")
                backup_dir.mkdir(parents=True, exist_ok=True)
                for file in backup_files:
                    if file.exists():
                        shutil.move(file, backup_dir)

        # torch.save(self.generator.model, model_file_path.absolute())
        m = torch.jit.script(self.generator.model)  # TODO Maybe try to use JSON-Pickle instead of torchscript
        m.save(model_file_path.absolute())
        print(f"Saved the trained model under: {model_file_path}")

        CsvSerializer.save(feature_labels_file_path, self.data_holder.data_labels, Feature)
        print(f"Saved the corresponding feature labels list under: {feature_labels_file_path}")

        if self.data_holder.normalizer is not None:
            BaseNormalizer.save(self.data_holder.normalizer, normalizer_file_path.absolute())
            print(f"Saved the corresponding normalizer model under: {normalizer_file_path}")
