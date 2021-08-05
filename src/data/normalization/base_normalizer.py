from __future__ import annotations

import abc
import json
from typing import Generic, TypeVar

import numpy
import torch

from src.data.serializer.class_serializer import ClassSerializer

T = TypeVar("T", type(torch.tensor), type(numpy.array))


class BaseNormalizer(Generic[T]):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def save(instance: BaseNormalizer, filename: str):
        ClassSerializer[BaseNormalizer].save(instance, filename)

    @staticmethod
    def load(filename: str) -> BaseNormalizer:
        return ClassSerializer[BaseNormalizer].load(filename)

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

    def __str__(self) -> str:
        return f'Instance of {self.__module__}.{self.__class__.__name__} with state: {json.dumps(self.__dict__, indent=4, sort_keys=True, default=str)}'


