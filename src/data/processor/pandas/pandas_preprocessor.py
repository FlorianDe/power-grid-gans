import numpy as np
import pandas as pd
from pandas import DataFrame

from src.data.processor.preprocessor import Preprocessor


class PandasPreprocessor(Preprocessor[DataFrame]):
    def clip(self, lower: float = None, upper: float = None):
        self.data = self.data.clip(lower, upper)
        return self


if __name__ == '__main__':
    np.random.seed(0)
    n = 20
    dates = np.array('2005-01-01', dtype=np.datetime64) + np.arange(n)
    data = np.sin(np.arange(n))
    df = pd.DataFrame({'data': data}, index=dates)

    a: Preprocessor = PandasPreprocessor(df).clip(0)

    print(df[df <= 0.0].count())
    pf = a.run()
    print(pf[pf == 0.0].count())
