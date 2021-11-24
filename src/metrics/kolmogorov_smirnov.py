from collections import namedtuple
from typing import Union

from math import sqrt, e, log
import numpy.typing as npt
from scipy.stats import ks_2samp  # anderson_ksamp

KSResult = namedtuple('KSResult', 'statistic pvalue')


def ks2_test(sample_a: npt.ArrayLike, sample_b: npt.ArrayLike) -> KSResult:
    return ks_2samp(sample_a, sample_b)


def ks2_critical_value(sample_a: Union[npt.ArrayLike, int], sample_b: Union[npt.ArrayLike, int], alpha: float) -> float:
    n = sample_a if isinstance(sample_a, int) else len(sample_a)
    m = sample_b if isinstance(sample_b, int) else len(sample_b)
    return sqrt(-log(alpha / 2.0, e) * ((1 + m / n) / (2 * m)))
