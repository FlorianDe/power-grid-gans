from math import log
from typing import Generator
import numpy.typing as npt

from src.utils.type_utils import LogBase


def __calculate_kl_divergence_terms(p: npt.ArrayLike, q: npt.ArrayLike, base: LogBase) -> Generator[float, None, None]:
    for i in range(len(p)):
        yield p[i] * log(p[i] / q[i], base)
    # return [p[i] * log(p[i] / q[i], base) for i in range(len(p))]


def __calculate_kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, base: LogBase) -> float:
    return sum(__calculate_kl_divergence_terms(p, q, base))


def kl_divergence(p: npt.ArrayLike, q: npt.ArrayLike, base: LogBase = 2) -> float:
    if len(p) != len(q):
        raise ValueError("The passed in data samples have to have the same length")
    return __calculate_kl_divergence(p, q, base)
