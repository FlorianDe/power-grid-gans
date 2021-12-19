from math import log
import numpy.typing as npt

from src.utils.type_utils import LogBase


def __calculate_kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, base: LogBase) -> float:
    return sum(p[i] * log(p[i] / q[i], base) for i in range(len(p)))


def kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, base: LogBase = 2) -> float:
    if len(p) is not len(q):
        raise ValueError("The passed in data samples have to have the same length")
    return __calculate_kl_divergence(p, q, base)

