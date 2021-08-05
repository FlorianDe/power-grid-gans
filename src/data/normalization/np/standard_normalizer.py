import numpy as np

from src.data.normalization.base_normalizer import BaseNormalizer


class StandardNumpyNormalizer(BaseNormalizer[np.array]):

    def __init__(self) -> None:
        super().__init__()
        self.sigma = None
        self.mu = None

    def is_fitted(self) -> bool:
        return self.sigma is not None and self.mu is not None

    def fit(self, data: np.array, dim: int = 0) -> None:
        if data.ndim > 2:
            raise ValueError('Currently only supporting 2 dimensional input data [sequence, features]')
        self.mu = np.mean(data, dim)
        self.sigma = np.std(data, dim)

    def normalize(self, data: np.array) -> np.array:
        self.check_fitted()
        normalized_data = data - self.mu
        normalized_data /= self.sigma
        return normalized_data

    def renormalize(self, data: np.array) -> np.array:
        self.check_fitted()
        renormalized_data = data * self.sigma
        renormalized_data += self.mu

        return renormalized_data
