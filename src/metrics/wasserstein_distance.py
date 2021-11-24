import numpy as np
from scipy.stats import wasserstein_distance
import numpy.typing as npt


def wasserstein_dist(sample_a: npt.ArrayLike, sample_b: npt.ArrayLike):
    return wasserstein_distance(sample_a, sample_b)


if __name__ == '__main__':
    # np.random.seed(12345678)
    # x = np.random.normal(0, 1, 1000)
    # y = np.random.normal(0, 1, 1000)
    # z = np.random.normal(1.1, 0.9, 1000)

    x = np.asarray([5, 2, 3])
    y = np.asarray([0, 7, 3])

    np.random.seed(42)
    n = 5000
    d1 = np.random.normal(50, 10, n)
    d2 = np.random.normal(70, 12, n)
    d3 = np.random.normal(15, 15, n)

    wd12 = wasserstein_dist(d1, d2)
    wd23 = wasserstein_dist(d2, d3)
    wd13 = wasserstein_dist(d1, d3)

    print(f"{wd12=}")
    print(f"{wd23=}")
    print(f"{wd12+wd23=}")
    print(f"{wd13=}")

