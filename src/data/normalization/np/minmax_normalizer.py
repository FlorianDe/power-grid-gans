import numpy as np

from src.data.normalization.base_normalizer import BaseNormalizer


class MinMaxNumpyNormalizer(BaseNormalizer[np.array]):
    def __init__(self) -> None:
        super().__init__()
        self.min = None
        self.max = None

    def is_fitted(self) -> bool:
        return self.min is not None and self.max is not None

    def fit(self, data: np.array, dim: int = 0) -> None:
        if data.ndim > 2:
            raise ValueError('Currently only supporting 2 dimensional input data [sequence, features]')
        self.min = np.min(data, 0)
        self.max = np.max(data, 0)
        print(f'Using {self.min=}, {self.max=}')

    def normalize(self, data: np.array) -> np.array:
        self.check_fitted()
        normalized_data = data - self.min
        normalized_data /= self.max
        return normalized_data

    def renormalize(self, data: np.array) -> np.array:
        self.check_fitted()
        renormalized_data = data * self.max
        renormalized_data += self.min

        return renormalized_data
