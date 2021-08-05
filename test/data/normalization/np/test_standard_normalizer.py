import tempfile
import unittest

import numpy

from src.data.normalization.base_normalizer import BaseNormalizer
from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer
from test.data.normalization.test_base_normalizer import test_serialization_helper, test_is_fitted_helper


class TestStandardNumpyNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = numpy.array([
            [1, 123],
            [2, 234],
            [3, 567],
            [4, 890],
        ])
        self.normalizer = StandardNumpyNormalizer()

    def test_init(self):
        test_is_fitted_helper(self, self.normalizer, False)

    def test_fit_data(self):
        self.normalizer.fit(self.data)
        test_is_fitted_helper(self, self.normalizer, True)
        self.assertTrue(numpy.allclose(self.normalizer.mu, [2.5, 453.5]))
        self.assertTrue(numpy.allclose(self.normalizer.sigma, [1.1180339887499, 300.34355328523]))


class TestStandardNumpyNormalizerSerializability(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = numpy.array([
            [1, 2, 3],
            [2, 3, 4]
        ])
        self.normalizer = StandardNumpyNormalizer()

    def test_serialization(self):
        loaded_normalizer = test_serialization_helper(self, self.normalizer, self.data)
        assert isinstance(loaded_normalizer, StandardNumpyNormalizer)

        self.assertTrue(numpy.allclose(self.normalizer.sigma, loaded_normalizer.sigma))
        self.assertTrue(numpy.allclose(self.normalizer.mu, loaded_normalizer.mu))


if __name__ == '__main__':
    unittest.main()
