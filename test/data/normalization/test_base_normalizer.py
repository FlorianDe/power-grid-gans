import tempfile
import unittest

import numpy

from src.data.normalization.base_normalizer import BaseNormalizer


def test_is_fitted_helper(tester: unittest.TestCase, normalizer: BaseNormalizer, expected: bool):
    tester.assertEqual(normalizer.is_fitted(), expected, "Normalizer should be fitted." if expected else "Normalizer shouldn't be fitted yet.")


def test_serialization_helper(tester: unittest.TestCase, normalizer: BaseNormalizer, data: numpy.array) -> BaseNormalizer:
    normalizer.fit(data)
    test_is_fitted_helper(tester, normalizer, True)
    normalized_data = normalizer.normalize(data)

    filename = tempfile.NamedTemporaryFile().name
    BaseNormalizer.save(normalizer, filename)
    loaded_normalizer = BaseNormalizer.load(filename)

    print(loaded_normalizer)

    loaded_normalizer.is_fitted()
    test_is_fitted_helper(tester, loaded_normalizer, True)
    ser_normalized_data = loaded_normalizer.normalize(data)

    tester.assertTrue(numpy.allclose(normalized_data, ser_normalized_data))

    return loaded_normalizer
