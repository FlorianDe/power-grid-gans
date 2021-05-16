from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


@dataclass
class GraphPlotItem:
    label: str
    x: List[float]
    y: List[float]
    color: str = 'r'


class TensorboardUtils:
    @staticmethod
    def plot_graph_as_figure(
            writer: SummaryWriter,
            plot_data: List[GraphPlotItem],
            tag: str,
            global_step: int = 0,
            auto_close: bool = False
    ) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # plot the functions
        for data in plot_data:
            plt.plot(data.x, data.y, color=data.color, label=data.label)

        plt.legend(loc='upper left')
        writer.add_figure(tag, fig, global_step, close=auto_close)
        plt.close(fig)
