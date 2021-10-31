from pathlib import Path
from typing import TypeVar, Type, Union

from dataclass_csv import DataclassReader, DataclassWriter

T = TypeVar("T")


class CsvSerializer:
    @staticmethod
    def save(filename: Union[str, Path], data: list[T], cls: Type[T]):
        with open(filename, 'w') as output_file:
            w = DataclassWriter(output_file, data, cls)
            w.write()

    @staticmethod
    def load(filename: Union[str, Path], cls: Type[T]) -> list[T]:
        res: list[T] = []
        with open(filename, 'r') as input_file:
            reader = DataclassReader(input_file, cls)
            for row in reader:
                res.append(row)
        return res

