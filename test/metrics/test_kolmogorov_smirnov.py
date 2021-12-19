import unittest

import numpy as np
from src.metrics.kolmogorov_smirnov import ks2_test, ks2_critical_value


class TestKolmogorovSmirnov(unittest.TestCase):
    x = np.random.normal(0, 0.1, 1000)
    y = np.random.normal(0, 0.1, 900)
    z = np.random.normal(10, 10, 800)

    def setUp(self) -> None:
        super().setUp()

    def test_kolmogorov_smirnov_test(self):
        ks_xy = ks2_test(self.x, self.y)
        ks_yx = ks2_test(self.y, self.x)
        ks_xz = ks2_test(self.x, self.z)

        crit_xy = ks2_critical_value(self.x, self.y, 0.05)
        crit_xz = ks2_critical_value(self.x, self.z, 0.05)

        self.assertAlmostEqual(ks_xy.statistic, ks_yx.statistic, 9)
        self.assertAlmostEqual(ks_xy.pvalue, ks_yx.pvalue, 9)

        self.assertTrue(ks_xy.statistic < crit_xy)
        self.assertTrue(ks_xz.statistic > crit_xz)
