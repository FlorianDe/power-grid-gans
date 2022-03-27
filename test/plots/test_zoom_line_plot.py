from datetime import timedelta
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from src.plots.typing import PlotData
from src.plots.zoom_line_plot import ConnectorBoxOptions, ZoomBoxEffectOptions, ZoomPlotOptions, draw_zoom_line_plot

# from src.plots.zoom_line_plot import draw_zoom_line_plot
from src.utils.datetime_utils import convert_input_str_to_date, interval_generator


def test_draw_zoom_line_plot():
    # sns.set_theme()
    # sns.set_style("whitegrid")
    # sns.set_context("paper")

    f = lambda x, m, b: m * x + b

    start = convert_input_str_to_date("2023.01.01")
    end = convert_input_str_to_date("2023.12.31")
    dates = np.array([d for d in interval_generator(start, end, delta=timedelta(hours=1))])

    x = np.linspace(0, 10, len(dates))
    y = f(x, 2, 1)

    draw_zoom_line_plot(
        raw_plot_data=[PlotData(data=y, label="test_data")],
        x=dates,
        zoom_boxes_options=[
            ZoomPlotOptions(
                x_start=convert_input_str_to_date("2023.01.01"),
                x_end=convert_input_str_to_date("2023.01.07"),
                effect_options=ZoomBoxEffectOptions(source_connector_box_options=ConnectorBoxOptions()),
            ),
            ZoomPlotOptions(
                x_start=convert_input_str_to_date("2023.08.01"), x_end=convert_input_str_to_date("2023.08.07")
            ),
        ],
    )
    # plt.show()
