import numpy as np
import numpy.typing as npt


def nll(pdf: npt.ArrayLike):
    return -np.sum(np.log(pdf))  # Calculate negative log likelihood
