import numpy as np
import numpy.typing as npt


def r_squared(y: npt.ArrayLike, pdf: npt.ArrayLike) -> float:
    """
    R_Squared := 1 - (residual_sum_of_squares/total_sum_of_squares)
    """
    return 1 - (residual_sum_of_squares(y, pdf)/total_sum_of_squares(y))


def total_sum_of_squares(y: npt.ArrayLike) -> float:
    """
    total sum of squares (TSS)
    """
    y_mean = np.mean(y)
    return np.sum(np.power(y - y_mean, 2.0))


def residual_sum_of_squares(y: npt.ArrayLike, pdf: npt.ArrayLike) -> float:
    """
    sum of squared estimate of errors / residual sum of squares (RSS)
    """
    return np.sum(np.power(y - pdf, 2.0))