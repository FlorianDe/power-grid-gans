from pandas import Series

from src.data.processor.pandas import PandasPreprocessor


def clip_to_zero(data: Series) -> Series:
    return PandasPreprocessor(data).clip(0).run()
