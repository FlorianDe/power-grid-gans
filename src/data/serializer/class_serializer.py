import pickle
from typing import Generic, TypeVar

T = TypeVar("T")


class ClassSerializer(Generic[T]):
    @staticmethod
    def save(instance: T, filename: str):
        with open(filename, 'wb') as output_file:
            pickle.dump(instance, output_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str) -> T:
        with open(filename, 'rb') as input_file:
            return pickle.load(input_file)
