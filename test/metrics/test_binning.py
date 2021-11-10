import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from metrics.binning import draw_hist_plot, draw_qq_plot
from metrics.types import PlotData

import seaborn as sns

def test_draw_hist_plot():
    sns.set_theme()
    sns.set_context("paper")
    np.random.seed(42)
    n = 5000
    d1 = np.random.normal(50, 10, n)
    d2 = np.random.normal(70, 12, n)
    d3 = np.random.normal(15, 15, n)

    fig = draw_hist_plot([
        PlotData(data=d1, label='d1'),
        PlotData(data=d2, label='d2'),
        PlotData(data=d3, label='d3'),
    ])
    # fig.savefig('to.png')
    fig.show()
    plt.close(fig)
    assert fig is not None


def test_draw_qq_plot():
    sns.set_theme()
    sns.set_context("paper")
    np.random.seed(42)
    n = 5000
    d1 = np.linspace(-1, 1, int(n/2))
    d2 = np.random.normal(50, 10, n)
    fig = draw_qq_plot(
        PlotData(data=d1, label='Real Values'),
        PlotData(data=d2, label='Theoretical quantiles'),
    )
    fig.show()

    stats.probplot(d1, dist=stats.norm, plot=plt)
    plt.show()
