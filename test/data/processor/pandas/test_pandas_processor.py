import unittest
import numpy as np
import pandas as pd

from src.data.processor.pandas import PandasProcessor
from src.data.processor.processor import Processor


class TestDataHolder(unittest.TestCase):
    def test_clip(self):
        data_lbl = "data"
        np.random.seed(0)
        n = 20
        dates = np.array("2005-01-01", dtype=np.datetime64) + np.arange(n)
        data = np.sin(np.arange(n))
        df = pd.DataFrame({data_lbl: data}, index=dates)

        a: Processor = PandasProcessor(df).clip(0)

        self.assertGreater(df[df < 0.0][data_lbl].count(), 0)

        # clip all values below zero
        pf = a.run()
        self.assertEqual(pf[pf < 0.0][data_lbl].count(), 0)
