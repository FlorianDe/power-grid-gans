from dataclasses import asdict, dataclass
from typing import Optional, Union
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, BboxConnectorPatch
import numpy as np
import numpy.typing as npt
from src.plots.plot_utils import assert_equal_plot_data_len

from src.plots.typing import PlotColor, PlotData, PlotOptions
from matplotlib.patches import ConnectionPatch

#  Base code from:
#
#  https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html


@dataclass
class MainPlotOptions:
    xaxis_major_ticker_locator: ticker.Locator = mdates.MonthLocator(interval=1)
    xaxis_major_ticket_formatter: ticker.Formatter = mdates.DateFormatter("%b")


@dataclass
class ConnectorOptions:
    edgecolor: Optional[PlotColor] = "black"
    alpha: float = 1.0
    linestyle: str = "-"
    linewidth: float = 1.0


@dataclass
class ConnectorBoxOptions:
    facecolor: Optional[PlotColor] = None
    edgecolor: Optional[PlotColor] = "black"
    alpha: float = 0.2
    linestyle: Optional[str] = None
    linewidth: float = 1.0


@dataclass
class ZoomBoxEffectOptions:
    first_connector_options: ConnectorOptions = ConnectorOptions()
    second_connector_options: ConnectorOptions = ConnectorOptions()
    source_connector_box_options: ConnectorBoxOptions = ConnectorBoxOptions()
    dest_connector_box_options: ConnectorBoxOptions = ConnectorBoxOptions()
    connector_patch_options: ConnectorBoxOptions = ConnectorBoxOptions()


@dataclass
class ZoomPlotOptions:
    x_start: any
    x_end: any
    xaxis_major_ticker_locator: ticker.Locator = mdates.DayLocator(interval=1)
    xaxis_major_ticket_formatter: ticker.Formatter = mdates.DateFormatter("%d.%b")
    effect_options: ZoomBoxEffectOptions = ZoomBoxEffectOptions()


def connect_bbox(bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, zoom_effect_options: ZoomBoxEffectOptions):
    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **asdict(zoom_effect_options.first_connector_options)
    )
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **asdict(zoom_effect_options.second_connector_options)
    )

    bbox_patch1 = BboxPatch(bbox1, **asdict(zoom_effect_options.dest_connector_box_options))
    bbox_patch2 = BboxPatch(bbox2, zorder=100, **asdict(zoom_effect_options.source_connector_box_options))

    p = BboxConnectorPatch(
        bbox1,
        bbox2,
        # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
        loc1a=loc1a,
        loc2a=loc2a,
        loc1b=loc1b,
        loc2b=loc2b,
        clip_on=False,
        **asdict(zoom_effect_options.connector_patch_options),
    )

    return c1, c2, bbox_patch1, bbox_patch2, p


def add_zoom_effect(ax1, ax2, zoom_effect_options: ZoomBoxEffectOptions):
    """
    ax1 : the main axes
    ax1 : the zoomed axes

    Similar to zoom_effect01.  The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        mybbox1, mybbox2, loc1a=3, loc2a=2, loc1b=4, loc2b=1, zoom_effect_options=zoom_effect_options
    )

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def draw_zoom_line_plot(
    raw_plot_data: list[PlotData[npt.ArrayLike]],
    x: npt.ArrayLike,
    zoom_boxes_options: list[ZoomPlotOptions],
    main_box_options: MainPlotOptions = MainPlotOptions(),
    plot_options: PlotOptions = PlotOptions(),
    fig: Optional[Figure] = None,
) -> tuple[Figure, Union[Axes, list[Axes]]]:
    assert_equal_plot_data_len(raw_plot_data)
    if raw_plot_data[0].data.size != x.size:
        raise ValueError(
            f"The length of the x values has to be equal to the length of every plot data {x.size=}, data size={raw_plot_data[0].data.size}."
        )

    fig = fig if fig is not None else plt.figure(figsize=(9, 3))
    axs = fig.subplot_mosaic(
        [
            [str(z) for z in range(len(zoom_boxes_options))],
            ["main" for _ in range(len(zoom_boxes_options))],
        ],
        # gridspec_kw={"width_ratios": [9, 2]},
    )

    for ax in axs.values():
        ax.grid(True, axis="both", linestyle="-")

    ax_main = axs["main"]
    for plot_data in raw_plot_data:
        ax_main.plot(x, plot_data.data, label=plot_data.label)
    ax_main.xaxis.set_major_locator(main_box_options.xaxis_major_ticker_locator)
    ax_main.xaxis.set_major_formatter(main_box_options.xaxis_major_ticket_formatter)
    ax_main.tick_params(axis="x", labelrotation=30)
    ax.legend(loc=plot_options.legend_location)

    for zoom_box_idx, zoom_box_options in enumerate(zoom_boxes_options):
        zoom_box = axs[f"{zoom_box_idx}"]
        zoom_box_options.x_start
        zoom_box_options.x_end

        zoom_x_indexes = np.where(np.logical_and(zoom_box_options.x_start <= x, x <= zoom_box_options.x_end))
        x_zoom_box = x[zoom_x_indexes]
        for plot_data in raw_plot_data:
            zoom_box.plot(x_zoom_box, plot_data.data[zoom_x_indexes])

        zoom_box.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        zoom_box.xaxis.set_major_formatter(mdates.DateFormatter("%d.%b"))
        zoom_box.tick_params(axis="x", labelrotation=30, labeltop=True, bottom=False, top=True, labelbottom=False)
        add_zoom_effect(zoom_box, ax_main, zoom_box_options.effect_options)

    return (fig, axs)
