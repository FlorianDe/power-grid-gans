from typing import List

import matplotlib.pyplot as plt
from pandas import DataFrame


def plot(data: List[DataFrame]):
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
