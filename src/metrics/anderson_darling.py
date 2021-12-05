from collections import namedtuple

import numpy as np
from scipy.stats import anderson_ksamp
import numpy.typing as npt

Anderson_ksampResult = namedtuple('Anderson_ksampResult',
                                  ('statistic', 'critical_values',
                                   'significance_level'))


def anderson_darling_test(sample_a: npt.ArrayLike, sample_b: npt.ArrayLike) -> Anderson_ksampResult:
    return anderson_ksamp([sample_a, sample_b], midrank=True)


if __name__ == '__main__':
    x = np.random.normal(0, 0.1, 1000)
    y = np.random.normal(0, 0.1, 900)
    z = np.random.normal(10, 10, 800)

    print(f"{anderson_darling_test(x, y)=}")
    print(f"{anderson_darling_test(x, z)=}")