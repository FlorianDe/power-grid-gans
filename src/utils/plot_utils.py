import sys
import numpy.typing as npt
from collections import namedtuple
from typing import List, Union

import matplotlib.pyplot as plt
from pandas import DataFrame

MinMaxValue = namedtuple('MinMaxValue', 'min_value max_value')


def get_min_max(samples: list[Union[npt.ArrayLike]]) -> MinMaxValue:
    min_value = sys.float_info.max
    max_value = sys.float_info.min

    for data in samples:
        min_value = min(min(data), min_value)
        max_value = max(max(data), max_value)

    return min_value, max_value


def plot_dfs(data: List[DataFrame]):
    temp_plot_options = {
        'figure.figsize': (20, 5),
        'figure.dpi': 300,
        'lines.linewidth': 2
    }
    with plt.rc_context(temp_plot_options):
        for i, d in enumerate(data):
            plt.figure(i)
            plot = d.plot()
            plt.show()
            # fig = plot.get_figure()
            # fig.savefig(f'cached/test{i}.eps', format='eps', dpi=1200)
