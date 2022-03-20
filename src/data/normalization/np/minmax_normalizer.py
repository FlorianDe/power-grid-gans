from typing import Optional
import numpy as np

from src.data.normalization.base_normalizer import BaseNormalizer


class MinMaxNumpyNormalizer(BaseNormalizer[np.array]):
    def __init__(
        self,
        destination_range: tuple[float, float] = (-1, 1),
        source_range: Optional[tuple[np.array, np.array]] = None,
        clip: bool = False,
    ) -> None:
        super().__init__()
        self.destination_range = destination_range
        self.clip = clip
        if source_range is not None:
            self._fit_via_intervals(source_range[0], source_range[1])
        else:
            self._initialize_private_fields()

    def _initialize_private_fields(self):
        self._scale = None
        self._min = None
        self._data_min = None
        self._data_max = None
        self._data_range = None

    def _handle_zero_range(self, scale):
        if np.isscalar(scale):
            if scale == 0.0:
                scale = 1.0
            return scale
        elif isinstance(scale, np.ndarray):
            scale = scale.copy()
            scale[scale == 0.0] = 1.0
        return scale

    def is_fitted(self) -> bool:
        return self._min is not None and self._scale is not None

    def _fit_via_intervals(self, data_min: np.array, data_max: np.array) -> None:
        if data_min.shape != data_max.shape:
            raise ValueError(
                f"The shape of the provided data intervals min, max have to be equal. {data_min.shape=}, {data_max.shape=}"
            )
        self._initialize_private_fields()
        data_range = data_max - data_min
        self._scale = (self.destination_range[1] - self.destination_range[0]) / self._handle_zero_range(data_range)
        self._min = self.destination_range[0] - data_min * self._scale
        self._data_min = data_min
        self._data_max = data_max
        self._data_range = data_range

    def fit(self, data: np.array, dim: int = 0) -> None:
        if data.ndim > 2:
            raise ValueError("Currently only supporting 2 dimensional input data [sequence, features]")
        data_min = np.nanmin(data, axis=dim)
        data_max = np.nanmax(data, axis=dim)
        self._fit_via_intervals(data_min, data_max)

    def normalize(self, data: np.array) -> np.array:
        self.check_fitted()
        normalized_data = data * self._scale
        normalized_data += self._min
        if self.clip:
            np.clip(normalized_data, self.destination_range[0], self.destination_range[1], out=normalized_data)
        # normalized_data = data - self.min
        # normalized_data = normalized_data / self.max
        return normalized_data

    def renormalize(self, data: np.array) -> np.array:
        self.check_fitted()
        # renormalized_data = data * self.max
        # renormalized_data = renormalized_data + self.min
        renormalized_data = data - self._min
        renormalized_data /= self._scale
        return renormalized_data
