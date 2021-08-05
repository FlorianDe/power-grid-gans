import unittest

import numpy

from src.data.normalization.np.minmax_normalizer import MinMaxNumpyNormalizer
from test.data.normalization.test_base_normalizer import test_is_fitted_helper, test_serialization_helper


class TestMinMaxNumpyNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = numpy.array([
            [1, 123],
            [2, 234],
            [3, 567],
            [4, 890],
        ])
        self.normalizer = MinMaxNumpyNormalizer()

    def test_init(self):
        test_is_fitted_helper(self, self.normalizer, False)

    def test_fit_data(self):
        self.normalizer.fit(self.data)
        test_is_fitted_helper(self, self.normalizer, True)
        self.assertTrue(numpy.allclose(self.normalizer.min, [1, 123]))
        self.assertTrue(numpy.allclose(self.normalizer.max, [4, 890]))


class TestMinMaxNumpyNormalizerSerializability(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = numpy.array([
            [1, 2, 3],
            [2, 3, 4]
        ])
        self.normalizer = MinMaxNumpyNormalizer()

    def test_serialization(self):
        loaded_normalizer = test_serialization_helper(self, self.normalizer, self.data)
        assert isinstance(loaded_normalizer, MinMaxNumpyNormalizer)

        self.assertTrue(numpy.allclose(self.normalizer.min, loaded_normalizer.min))
        self.assertTrue(numpy.allclose(self.normalizer.max, loaded_normalizer.max))


if __name__ == '__main__':
    unittest.main()
