import abc
from typing import Generic, TypeVar

import numpy
import torch

T = TypeVar("T", type(torch.tensor), type(numpy.array))


class BaseNormalizer(Generic[T]):
    __metaclass__ = abc.ABCMeta

    def check_fitted(self) -> None:
        if not self.is_fitted():
            raise AssertionError("The normalizer has to be fitted against some kind of data, before it can be used to re/normalize data.")

    @abc.abstractmethod
    def fit(self, data: T) -> None:
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def is_fitted(self) -> bool:
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def normalize(self, data: T) -> T:
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def renormalize(self, data: T) -> T:
        raise NotImplementedError("Please Implement this method")
