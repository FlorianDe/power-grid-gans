from math import sqrt
import numpy.typing as npt

from metrics.kullback_leibler import __calculate_kl_divergence
from utils.type_utils import LogBase


def js_divergence(p: npt.ArrayLike, q: npt.ArrayLike, base: LogBase = 2) -> float:
    if len(p) is not len(q):
        raise ValueError("The passed in data samples have to have the same length")
    m = 0.5 * (p + q)
    return 0.5 * __calculate_kl_divergence(p, m, base) + 0.5 * __calculate_kl_divergence(q, m, base)


def js_distance(p: npt.ArrayLike, q: npt.ArrayLike, base: LogBase = 2) -> float:
    return sqrt(js_divergence(p, q, base))


# if __name__ == '__main__':
#     x = np.random.normal(0, 1, 1000)
#     y = np.random.normal(0, 1, 1000)
#     min_val = min(np.min(x), np.min(y))
#     max_val = max(np.max(x), np.max(x))
#     hist_x, bin_edges_x = np.histogram(x, bins=100, range=(min_val, max_val), density=True)
#     hist_y, bin_edges_y = np.histogram(y, bins=100, range=(min_val, max_val), density=True)
#
#     plt.bar(x=bin_edges_x[:-1], height=hist_x, )  # width=65536./1000)
#     plt.bar(x=bin_edges_y[:-1], height=hist_y, )  # width=65536./1000)
#     plt.show()
