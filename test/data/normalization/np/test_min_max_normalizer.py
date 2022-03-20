import unittest

import numpy

from src.data.normalization.np.minmax_normalizer import MinMaxNumpyNormalizer
from test.data.normalization._test_base_normalizer import _test_is_fitted_helper, _test_serialization_helper


class TestMinMaxNumpyNormalizer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dest_from = -1
        self.dest_to = 1
        self.min_f1 = 1
        self.max_f1 = 999
        self.min_f2 = 2
        self.max_f2 = 888
        self.data = numpy.array(
            [
                [self.max_f1, 100],
                [11, self.min_f2],
                [self.min_f1, self.max_f2],
                [13, 400],
            ]
        )
        self.normalizer = MinMaxNumpyNormalizer()

    def test_init(self):
        _test_is_fitted_helper(self, self.normalizer, False)

    def test_fit_data(self):
        self.normalizer.fit(self.data)
        _test_is_fitted_helper(self, self.normalizer, True)
        self.assertTrue(numpy.allclose(self.normalizer._data_min, [self.min_f1, self.min_f2]))
        self.assertTrue(numpy.allclose(self.normalizer._data_max, [self.max_f1, self.max_f2]))

    def test_normalize_data(self):
        normalizer = MinMaxNumpyNormalizer(destination_range=(self.dest_from, self.dest_to))
        normalizer.fit(self.data)
        self.assertTrue(
            numpy.allclose(normalizer.normalize([self.min_f1, self.min_f2]), [self.dest_from, self.dest_from])
        )
        self.assertTrue(numpy.allclose(normalizer.normalize([self.max_f1, self.max_f2]), [self.dest_to, self.dest_to]))

        self.assertTrue(numpy.all((normalizer.normalize(self.data) >= self.dest_from)))
        self.assertTrue(numpy.all((normalizer.normalize(self.data) <= self.dest_to)))

    def test_renormalize_data(self):
        normalizer = MinMaxNumpyNormalizer(destination_range=(self.dest_from, self.dest_to))
        normalizer.fit(self.data)
        self.assertTrue(
            numpy.allclose(normalizer.renormalize([self.dest_from, self.dest_from]), [self.min_f1, self.min_f2])
        )
        self.assertTrue(
            numpy.allclose(normalizer.renormalize([self.dest_to, self.dest_to]), [self.max_f1, self.max_f2])
        )

    def test_normalize_data_with_clip(self):
        normalizer = MinMaxNumpyNormalizer(destination_range=(self.dest_from, self.dest_to), clip=True)
        normalizer.fit(self.data)

        self.assertTrue(numpy.allclose(normalizer.normalize([-999, -9999]), [self.dest_from, self.dest_from]))
        self.assertTrue(numpy.allclose(normalizer.normalize([999, 9999]), [self.dest_to, self.dest_to]))

    def test_normalize_data_with_static_source(self):
        src_from = numpy.array([self.min_f1 - 10, self.min_f2 - 100])
        src_to = numpy.array([self.max_f1 + 15, self.max_f2 + 200])
        self.dest_from = -1
        self.dest_to = 1
        normalizer = MinMaxNumpyNormalizer(
            destination_range=(self.dest_from, self.dest_to), source_range=(src_from, src_to)
        )
        self.assertTrue(numpy.allclose(normalizer.normalize(src_from), [self.dest_from, self.dest_from]))
        self.assertTrue(numpy.allclose(normalizer.normalize(src_to), [self.dest_to, self.dest_to]))

        normalizer = MinMaxNumpyNormalizer(
            destination_range=(self.dest_from, self.dest_to), source_range=(src_from, src_to), clip=True
        )
        self.assertTrue(numpy.allclose(normalizer.normalize([-9999999, -9999999]), [self.dest_from, self.dest_from]))
        self.assertTrue(numpy.allclose(normalizer.normalize([9999999, 99999999]), [self.dest_to, self.dest_to]))


class TestMinMaxNumpyNormalizerSerializability(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = numpy.array([[1, 2, 3], [2, 3, 4]])
        self.normalizer = MinMaxNumpyNormalizer()

    def test_serialization(self):
        loaded_normalizer = _test_serialization_helper(self, self.normalizer, self.data)
        assert isinstance(loaded_normalizer, MinMaxNumpyNormalizer)

        self.assertTrue(numpy.allclose(self.normalizer._data_min, loaded_normalizer._data_min))
        self.assertTrue(numpy.allclose(self.normalizer._data_max, loaded_normalizer._data_max))


if __name__ == "__main__":
    unittest.main()
