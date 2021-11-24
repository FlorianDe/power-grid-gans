import numpy as np

from plots.histogram_plot import draw_hist_plot, HistPlotData

import seaborn as sns


def test_draw_hist_plot():
    sns.set_theme()
    sns.set_context("paper")
    np.random.seed(42)
    n = 5000
    d1 = np.random.normal(50, 10, n)
    d2 = np.random.normal(70, 12, n)
    d3 = np.random.normal(15, 15, n)

    res = draw_hist_plot([
        HistPlotData(data=d1, label='d1'),
        HistPlotData(data=d2, label='d2'),
        HistPlotData(data=d3, label='d3'),
    ])
    # fig.savefig('to.png')
    res.show()
    res.close()
    assert res.fig is not None
