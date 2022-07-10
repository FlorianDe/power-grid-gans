import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from src.plots.qq_plot import draw_qq_plot, QQReferenceLine
from src.plots.typing import PlotData


def test_draw_qq_plot():
    sns.set_theme()
    sns.set_context("paper")
    np.random.seed(42)
    n = 5000
    d1 = np.linspace(-1, 1, int(n / 2))
    d2 = np.random.normal(50, 10, n)  # np.linspace(-1, 1, int(n/2))  #
    res = draw_qq_plot(
        PlotData(data=d1, label="Real Values"),
        PlotData(data=d2, label="Theoretical quantiles"),
        50,
        {
            # QQReferenceLine.THEORETICAL_LINE,
            QQReferenceLine.FIRST_THIRD_QUARTIL,
            QQReferenceLine.LEAST_SQUARES_REGRESSION,
        },
        [0.25, 0.5, 0.75],
    )
    # res.fig.show()

    osm = stats.probplot(d1, dist=stats.norm, plot=plt)
    print(f"{osm=}")

    # plt.show()
