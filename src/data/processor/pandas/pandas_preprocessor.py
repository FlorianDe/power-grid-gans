from pandas import DataFrame

from src.data.processor.preprocessor import Preprocessor


class PandasPreprocessor(Preprocessor[DataFrame]):
    def clip(self, lower: float = None, upper: float = None):
        self.data = self.data.clip(lower, upper)
        return self


