import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as st
from scipy.stats import rv_continuous

from scipy.stats._continuous_distns import _distn_names

@dataclass
class DistributionParams:
    loc: float
    scale: float
    args: tuple[float, ...]
    kwds: Any  # dict[str, ]
    raw: tuple[float, ...]


@dataclass
class DistributionFit:
    distribution_name: str
    distribution: rv_continuous  # e.g. beta
    params: DistributionParams
    score: float  # e.g. Residual sum of squares

    def __lt__(self, other):
        return self.score < other.score


def test_fit_against_all_distributions(
        data: npt.ArrayLike,
        bins: int,
        error_fn: Callable[[npt.ArrayLike, npt.ArrayLike], float],
) -> list[DistributionFit]:
    """Model data by finding best fit distribution to data"""

    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    fitted_distributions: list[DistributionFit] = []

    for idx, distribution_name in enumerate([d for d in _distn_names if d not in ['levy_stable', 'studentized_range', 'vonmises']]):
        print("{:>3} / {:<3}: {}".format(idx + 1, len(_distn_names), distribution_name))

        distribution = getattr(st, distribution_name)

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # Try to fit distribution to data
                params_fit = distribution.fit(data)
                params = DistributionParams(
                    loc=params_fit[-2],
                    scale=params_fit[-1],
                    args=params_fit[:-2],
                    kwds=None,
                    raw=params_fit
                )
                # Calculate fitted pdf and error with fit in distribution
                pdf = distribution.pdf(x, loc=params.loc, scale=params.scale, *params.args)
                score = error_fn(y, pdf)
                print(f"${score=}")
                fitted_distributions.append(
                    DistributionFit(
                        distribution_name=distribution_name,
                        distribution=distribution,
                        params=params,
                        score=score
                    )
                )

        except Exception:
            print(f"Error while trying to fit data to ${distribution_name=}", file=sys.stderr)
            pass

    return fitted_distributions


def create_pdf_series_from_distribution(dist: rv_continuous, params, quantile_interval: tuple[float, float] = (0.001, 0.999), size=10000):
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    quantile_start, quantile_end = quantile_interval
    start = dist.ppf(quantile_start, *arg, loc=loc, scale=scale) if arg else dist.ppf(quantile_start, loc=loc, scale=scale)
    end = dist.ppf(quantile_end, *arg, loc=loc, scale=scale) if arg else dist.ppf(quantile_end, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf
