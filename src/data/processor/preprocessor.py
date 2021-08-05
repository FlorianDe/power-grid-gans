import copy
from typing import TypeVar, Generic

T = TypeVar("T")


class Preprocessor(Generic[T]):
    """
    STEPS:
    Data cleansing
    Data editing <- FOR NOW
    Data reduction
    Data wrangling
    """

    def __init__(self, data: T) -> None:
        super().__init__()
        self.data = copy.deepcopy(data)

    def run(self) -> T:
        return self.data

    def __call__(self):
        return self.run()
