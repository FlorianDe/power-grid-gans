from collections import namedtuple
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Union

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

PlotDataType = TypeVar("PlotDataType")
PlotColor = Union[str, list[str]]
Point = namedtuple('Point', 'x y')


@dataclass
class PlotResult:
    fig: Figure
    ax: plt.Axes

    def show(self, warn=True):
        self.fig.show(warn)

    def close(self):
        plt.close(self.fig)


@dataclass
class PlotOptions:
    title: Optional[str]
    legend_location: str = "best",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,


@dataclass
class PlotData(Generic[PlotDataType]):
    data: PlotDataType
    label: str
    color: Optional[PlotColor] = None
