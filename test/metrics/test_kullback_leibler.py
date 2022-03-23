import unittest
from numpy import asarray
from scipy.stats import entropy

from src.metrics.kullback_leibler import kl_divergence


class TestKullbackLeibler(unittest.TestCase):
    p = asarray([0.10, 0.40, 0.50])
    q = asarray([0.80, 0.15, 0.05])

    def setUp(self) -> None:
        super().setUp()

    def test_kl_divergence(self):
        js_pq = kl_divergence(self.p, self.q)
        js_qp = kl_divergence(self.q, self.p)
        entropy_pq = entropy(self.p, self.q, 2)
        entropy_qp = entropy(self.q, self.p, 2)
        print("KL(P || Q) divergence: %.3f bits" % js_pq)
        print("Entropy(P, Q): %.3f bits" % entropy_pq)
        print("KL(Q || P) divergence: %.3f bits" % js_qp)
        print("Entropy(Q, P): %.3f bits" % entropy_qp)

        self.assertAlmostEqual(js_pq, entropy_pq, 9)
        self.assertAlmostEqual(js_qp, entropy_qp, 9)

    def test_kl_divergence_invalid_input(self):
        self.assertRaises(ValueError, kl_divergence, asarray([0.10, 0.40]), asarray([0.10]))
