import numpy as np
from matplotlib import pyplot as plt

from plots.histogram_plot import draw_hist_plot, HistPlotData


def equal_frequency_binning(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


if __name__ == '__main__':
    np.random.seed(42)
    n = 5000
    d1 = np.random.normal(50, 10, n)
    d2 = np.random.normal(60, 11, n)

    res = draw_hist_plot([
        HistPlotData(data=d1, label='d1'),
        HistPlotData(data=d2, label='d2'),
    ])
    res.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    binnings = equal_frequency_binning(np.hstack((d1, d2)), 100)

    ax.hist([d1, d2], binnings, stacked=True, alpha=0.3)
    fig.show()
