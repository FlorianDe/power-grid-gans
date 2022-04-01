from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np
from experiments.experiments_utils.plotting import draw_weather_data_zoom_plot_sample
from experiments.experiments_utils.utils import get_experiments_folder

from src.data.normalization.np.minmax_normalizer import MinMaxNumpyNormalizer
from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns
from src.plots.typing import PlotData
from src.plots.zoom_line_plot import ConnectorBoxOptions, ZoomBoxEffectOptions, ZoomPlotOptions, draw_zoom_line_plot
from src.utils.datetime_utils import interval_generator


def draw_weather_data_trainings_data_time_series_zoom_plot(year: int):
    start = datetime.fromisoformat(f"{year}-01-01 00:00:00")
    end = datetime.fromisoformat(f"{year}-12-31 23:00:00")

    data_importer = DWDWeatherDataImporter(start_date=start, end_date=end, auto_preprocess=True)
    data_importer.initialize()

    fig, axes = draw_weather_data_zoom_plot_sample(
        dataframe=data_importer.data,
        start=start,
        end=end,
        zoom_boxes_options=[
            ZoomPlotOptions(
                x_start=datetime.fromisoformat(f"{year}-01-01T00:00:00"),
                x_end=datetime.fromisoformat(f"{year}-01-07T23:00:00"),
            ),
            ZoomPlotOptions(
                x_start=datetime.fromisoformat(f"{year}-08-01T00:00:00"),
                x_end=datetime.fromisoformat(f"{year}-08-07T23:00:00"),
            ),
        ],
    )
    return fig, axes


def check_processed_unprocessed_data():
    start_date = "2019-01-01 00:00:00"
    end_date = "2019-12-31 23:00:00"
    data_importer = DWDWeatherDataImporter(start_date=start_date, end_date=end_date, auto_preprocess=False)
    data_importer.initialize()
    normalizer = MinMaxNumpyNormalizer()
    normalizer.fit(data_importer.data)

    # data_holder = DataHolder(
    #     data=data_importer.data.values.astype(np.float32),
    #     data_labels=data_importer.get_feature_labels(),
    #     dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
    #     # normalizer_constructor=MinMaxNumpyNormalizer,
    # )

    data_importer_preprocessed = DWDWeatherDataImporter(start_date=start_date, end_date=end_date, auto_preprocess=True)
    data_importer_preprocessed.initialize()
    normalizer_preprocessed = MinMaxNumpyNormalizer()
    normalizer_preprocessed.fit(data_importer_preprocessed.data)

    print("###############################################")
    print("### Raw")
    for column, min, max in zip(data_importer.data.columns, normalizer._data_min, normalizer._data_max):
        print(f"{column}: {min=}, {max=}")


if __name__ == "__main__":
    # check_processed_unprocessed_data()
    explore_data_root_folder = get_experiments_folder().joinpath("01_explore_weather_data").joinpath("02_explore_data")
    explore_data_root_folder.mkdir(parents=True, exist_ok=True)

    for year in range(2010, 2020):
        fig, axes = draw_weather_data_trainings_data_time_series_zoom_plot(year)
        plt.savefig(
            explore_data_root_folder / f"dwd_weather_data_{year}_zoom_plot.pdf", bbox_inches="tight", pad_inches=0
        )

    # plt.close()
    # plt.show()
