from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


@dataclass
class GraphPlotItem:
    label: str
    x: List[float]
    y: List[float]
    color: Optional[str] = None


class TensorboardUtils:
    COLOR_MAP: list[str] = [
        '#800000',
        '#8B0000',
        '#FFFF00',
        '#9ACD32',
        '#556B2F',
        '#2F4F4F',
        '#40E0D0',
        '#7B68EE',
        '#9400D3',
        '#FFFFE0',
        '#FFDEAD',
        '#FFE4E1',
        '#FF0000',
        '#FF4500',
        '#FFD700',
        '#8FBC8F',
        '#87CEFA',
        '#FF00FF',
        '#FF69B4',
        '#8B4513',
    ]

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
        for i in range(len(plot_data)):
            data = plot_data[i]
            color = data.color if data.color else TensorboardUtils.COLOR_MAP[i % len(TensorboardUtils.COLOR_MAP)]
            plt.plot(data.x, data.y, color=color, label=data.label)

        plt.legend(loc='upper left')
        writer.add_figure(tag, fig, global_step, close=auto_close)
        plt.close(fig)
