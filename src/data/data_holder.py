from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import TensorDataset, SubsetRandomSampler

from data.typing import Feature
from src.data.normalization import BaseNormalizer, NoneNormalizer
from utils.type_utils import check_list_type


class DataHolder:
    def __init__(self,
                 data: npt.ArrayLike,
                 data_labels: Optional[Union[list[Feature], list[str]]] = None,
                 dates: Optional[npt.ArrayLike] = None,
                 normalizer_constructor: Optional[type(BaseNormalizer)] = None
                 ) -> None:
        super().__init__()
        if data is None:
            raise AssertionError('Cannot create an instance without a valid data set.')

        data_feature_size = data.shape[-1]
        if data_labels is None:
            data_labels = [Feature(f"Feature {f}") for f in range(data_feature_size)]
        elif check_list_type(data_labels, str):
            data_labels = [Feature(label, label) for label in data_labels]

        if not check_list_type(data_labels, Feature):
            raise ValueError(f"You have passed in some kind of mixed type feature labels array which is not suitable, {data_labels=}.")
        if data_feature_size != len(data_labels):
            raise ValueError(f"The feature labels for the data do not match in size. {data_feature_size=} and {data_labels=}")

        if dates is not None and len(dates) != len(data):
            raise ValueError("If you are passing corresponding time values x, they have to be of the same length as the passed data")

        if dates is None:
            dates = np.fromiter(range(len(data)), dtype="float32")

        if normalizer_constructor is None:
            normalizer_constructor = NoneNormalizer
        self.normalizer = normalizer_constructor()
        self.normalizer.fit(data)
        self.data = self.normalizer.normalize(data)
        self.data_labels = data_labels
        self.x = dates  # TODO Maybe transform it here already

    def get_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(torch.from_numpy(self.data), torch.from_numpy(self.x))

    def get_feature_size(self) -> int:
        return self.data.shape[-1]

    def get_feature_labels(self):
        return self.data_labels

    @staticmethod
    def create_dataset_sampler(dataset: TensorDataset, validation_split: float = .8, shuffle_dataset: bool = False, random_seed: int = 42):
        # Creating samplers for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(dataset_size * validation_split))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        return train_sampler, valid_sampler
