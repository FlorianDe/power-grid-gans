from typing import Optional

import numpy
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.normalization.base_normalizer import BaseNormalizer
from src.data.normalization.none_normalizer import NoneNormalizer


class DataHolder:
    def __init__(self, data: numpy.array, dates: Optional[numpy.array], normalizer_constructor: Optional[type(BaseNormalizer)] = None) -> None:
        super().__init__()
        if data is None:
            raise AssertionError('Cannot create an instance without a valid data set.')

        if dates is not None and len(dates) != len(data):
            raise ValueError("If you are passing corresponding time values x, they have to be of the same length as the passed data")

        if dates is None:
            dates = numpy.fromiter(range(len(data)), dtype="float32")

        if normalizer_constructor is None:
            normalizer_constructor = NoneNormalizer
        self.normalizer = normalizer_constructor()
        self.normalizer.fit(data)
        self.data = self.normalizer.normalize(data)
        self.x = dates  # TODO Maybe transform it here already

    def get_tensor_dataset(self):
        return TensorDataset(torch.from_numpy(self.data), torch.from_numpy(self.x))

    def get_feature_size(self):
        return self.data.shape[-1]


if __name__ == '__main__':
    x = numpy.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1]
    ])
    data = numpy.array([
        [0.1, 1.0],
        [3.5, 5.0],
        [10.2, 7.5]
    ])
    dh = DataHolder(data, x)
    normalized = dh.normalizer.normalize(data)
    print(f'{normalized=}')
    renormalized = dh.normalizer.renormalize(normalized)
    print(f'{renormalized=}')

    data_loader = DataLoader(
        dh.get_tensor_dataset(),
        batch_size=1,
        shuffle=True
    )
    for i, (real_data, labels) in enumerate(data_loader):
        print(f'{i}: {labels} -> {real_data}')
