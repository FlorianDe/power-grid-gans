from pandas import Series

from src.data.processor.processor import Processor


class PandasProcessor(Processor[Series]):
    def abs(self):
        self.data = self.data.abs()
        return self

    def clip(self, lower: float = None, upper: float = None):
        self.data = self.data.clip(lower, upper)
        return self

    def modulo(self, modulus: int):
        self.data = self.data.mod(modulus)
        return self
