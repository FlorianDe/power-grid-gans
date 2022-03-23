from pandas import Series

from src.data.processor.preprocessor import Preprocessor


class PandasPreprocessor(Preprocessor[Series]):
    def clip(self, lower: float = None, upper: float = None):
        self.data = self.data.clip(lower, upper)
        return self

    def modulo(self, modulus: int):
        self.data = self.data.mod(modulus)
        return self
