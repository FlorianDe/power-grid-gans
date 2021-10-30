import unittest

import numpy

from src.data.normalization.np import StandardNumpyNormalizer
from test.data.normalization._test_base_normalizer import _test_serialization_helper, _test_is_fitted_helper


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
        _test_is_fitted_helper(self, self.normalizer, False)

    def test_fit_data(self):
        self.normalizer.fit(self.data)
        _test_is_fitted_helper(self, self.normalizer, True)
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
        loaded_normalizer = _test_serialization_helper(self, self.normalizer, self.data)
        assert isinstance(loaded_normalizer, StandardNumpyNormalizer)

        self.assertTrue(numpy.allclose(self.normalizer.sigma, loaded_normalizer.sigma))
        self.assertTrue(numpy.allclose(self.normalizer.mu, loaded_normalizer.mu))


if __name__ == '__main__':
    unittest.main()
