import unittest

from numpy import asarray
from scipy.spatial.distance import jensenshannon

from src.metrics.jensen_shannon import js_divergence, js_distance


class TestJensenShannon(unittest.TestCase):
    p = asarray([0.10, 0.40, 0.50])
    q = asarray([0.80, 0.15, 0.05])

    def setUp(self) -> None:
        super().setUp()

    def test_js_divergence(self):
        js_pq = js_divergence(self.p, self.q)
        js_qp = js_divergence(self.q, self.p)
        print('JS(P || Q) divergence: %.3f bits' % js_pq)
        print('JS(Q || P) divergence: %.3f bits' % js_qp)
        self.assertAlmostEqual(js_pq, js_qp, 9)

    def test_js_distance(self):
        js_dist_pq = js_distance(self.p, self.q)
        js_dist_qp = js_distance(self.p, self.q)

        sci_js_pq_dist = jensenshannon(self.p, self.q, base=2)
        sci_js_qp_dist = jensenshannon(self.q, self.p, base=2)

        print('JS(P || Q) distance: %.3f' % js_dist_pq)
        print('JS(Q || P) scipy distance: %.3f' % sci_js_pq_dist)
        print('JS(Q || P) distance: %.3f' % js_dist_qp)
        print('JS(Q || P) scipy distance: %.3f' % sci_js_qp_dist)

        self.assertAlmostEqual(js_dist_pq, js_dist_qp, 9)
        self.assertAlmostEqual(js_dist_pq, sci_js_pq_dist, 9)
        self.assertAlmostEqual(js_dist_pq, sci_js_qp_dist, 9)

    def test_js_divergence_invalid_input(self):
        self.assertRaises(ValueError, js_divergence, asarray([0.10, 0.40]), asarray([0.10]))

    def test_js_distance_invalid_input(self):
        self.assertRaises(ValueError, js_distance, asarray([0.10, 0.40]), asarray([0.10]))
