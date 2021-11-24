import numpy as np


import seaborn as sns
from statsmodels.distributions import ECDF

from plots.ecdf_plot import ECDFPlotData, draw_ecdf_plot


def test_draw_ecdf_plot():
    sns.set_theme()
    sns.set_context("paper")
    np.random.seed(42)
    n = 300
    d1 = np.random.normal(1, 1, n)
    d2 = np.random.normal(0.9, 0.9, n)

    draw_ecdf_plot([
        ECDFPlotData(data=ECDF(d1), label="D1", confidence_band_alpha=0.01, confidence_band_fill_alpha=0.3),
        ECDFPlotData(data=ECDF(d2), label="D2", confidence_band_alpha=0.00, confidence_band_fill_alpha=0.3),
    ]).show()
