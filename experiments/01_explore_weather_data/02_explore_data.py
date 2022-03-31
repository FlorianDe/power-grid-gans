from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np

from src.data.normalization.np.minmax_normalizer import MinMaxNumpyNormalizer
from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns
from src.plots.typing import PlotData
from src.plots.zoom_line_plot import ConnectorBoxOptions, ZoomBoxEffectOptions, ZoomPlotOptions, draw_zoom_line_plot
from src.utils.datetime_utils import interval_generator


def create_time_series_zoom_plot():
    start = datetime.fromisoformat(DEFAULT_DATA_START_DATE)
    end = datetime.fromisoformat("2009-12-31 23:00:00")

    columns = [
        WeatherDataColumns.GH_W_PER_M2,
        WeatherDataColumns.DH_W_PER_M2,
        WeatherDataColumns.WIND_DIR_DEGREE,
        WeatherDataColumns.WIND_V_M_PER_S,
        WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
    ]

    data_importer = DWDWeatherDataImporter(start_date=start, end_date=end, auto_preprocess=True)
    data_importer.initialize()

    fig, axes = draw_zoom_line_plot(
        raw_plot_data=[PlotData(data=data_importer.data[col].values, label=col) for col in columns],
        x=np.array([d for d in interval_generator(start, end, delta=timedelta(hours=1))]),
        zoom_boxes_options=[
            ZoomPlotOptions(
                x_start=datetime.fromisoformat("2009-01-01T00:00:00"),
                x_end=datetime.fromisoformat("2009-01-07T23:00:00"),
                effect_options=ZoomBoxEffectOptions(source_connector_box_options=ConnectorBoxOptions()),
            ),
            ZoomPlotOptions(
                x_start=datetime.fromisoformat("2009-08-01T00:00:00"),
                x_end=datetime.fromisoformat("2009-08-07T23:00:00"),
            ),
        ],
    )
    plt.show()


def check_processed_unprocessed_data():
    start_date = DEFAULT_DATA_START_DATE
    end_date = "2009-12-31 23:00:00"
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

    create_time_series_zoom_plot()
