from src.plots.typing import PlotData


def assert_equal_plot_data_len(plot_data: PlotData[any]):
    plot_data_iter = iter(plot_data)
    first_len = (next(plot_data_iter).data).size
    if not all(l.data.size == first_len for l in plot_data_iter):
        raise ValueError("Not all provided plot data elements have the same length")
