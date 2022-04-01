from datetime import timedelta, datetime
from pathlib import PurePath
import numpy as np
import pandas as pd

import seaborn as sns

import random
from matplotlib import pyplot as plt
import torch
from statsmodels.distributions import ECDF
from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params

from experiments.experiments_utils.plotting import (
    draw_weather_data_zoom_plot_sample,
    plot_model_losses,
    plot_sample,
    save_fig,
)
from src.data.weather.weather_dwd_postprocessor import DWDWeatherPostProcessor
from src.evaluator.evaluator import Evaluator

from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns
from src.metrics.kolmogorov_smirnov import ks2_critical_value, ks2_test
from src.plots.ecdf_plot import ECDFPlotData, draw_ecdf_plot
from src.plots.typing import PlotData

from src.plots.zoom_line_plot import ConnectorBoxOptions, ZoomBoxEffectOptions, ZoomPlotOptions, draw_zoom_line_plot
from src.utils.datetime_utils import (
    convert_input_str_to_date,
    dates_to_conditional_vectors,
    get_day_in_year_from_date,
    interval_generator,
)

DEFAULT_FILE_ENDING = "pdf"


def ks_test_on_every_feature(evaluator: Evaluator):
    start = datetime.fromisoformat("2009-01-01T00:00:00")
    end = datetime.fromisoformat("2009-12-31T23:00:00")
    importer = DWDWeatherDataImporter(start_date=start, end_date=end)
    importer.initialize()
    dataframe = evaluator.generate_dataframe(start, end)
    weather_post_processor = DWDWeatherPostProcessor()
    dataframe = weather_post_processor(dataframe)
    for col in dataframe.columns:
        sample_a = dataframe[col].values
        sample_b = importer.data[col].values
        label = col.replace("_", r"\_")
        ks_test = ks2_test(sample_a=sample_a, sample_b=sample_b)
        ks_crit = ks2_critical_value(sample_a=sample_a, sample_b=sample_b, alpha=0.05)
        print(f"{col} -> {ks_test=} -> {ks_crit=}")
        draw_ecdf_plot(
            [
                ECDFPlotData(
                    data=ECDF(sample_b),
                    label="Theoretical data",
                    confidence_band_alpha=0.50,
                    confidence_band_fill_alpha=0.3,
                    confidence_band_label_supplier=lambda alpha: f"{alpha}% confidence band",
                ),
                ECDFPlotData(
                    data=ECDF(sample_a),
                    label=f"Sample {label}",
                    confidence_band_alpha=0.00,
                    confidence_band_fill_alpha=0.3,
                ),
            ]
        )


def save_sample_zoom_line_plot(
    evaluator: Evaluator, result_path: PurePath, epoch: int, plot_file_ending=DEFAULT_FILE_ENDING
):
    start = datetime.fromisoformat("2023-01-01T00:00:00")
    end = datetime.fromisoformat("2023-12-31T23:00:00")
    dataframe = evaluator.generate_dataframe(start, end)
    dataframe.to_hdf(result_path / "result_generated_data.hdf", "weather", "w")

    def __draw_weather_data_zoom_plot_sample_by_df(dataframe):
        return draw_weather_data_zoom_plot_sample(
            dataframe=dataframe,
            start=start,
            end=end,
            zoom_boxes_options=[
                ZoomPlotOptions(
                    x_start=datetime.fromisoformat("2023-01-01T00:00:00"),
                    x_end=datetime.fromisoformat("2023-01-07T23:00:00"),
                ),
                ZoomPlotOptions(
                    x_start=datetime.fromisoformat("2023-08-01T00:00:00"),
                    x_end=datetime.fromisoformat("2023-08-07T23:00:00"),
                ),
            ],
        )

    fig, axes = __draw_weather_data_zoom_plot_sample_by_df(dataframe)
    fig.savefig(
        result_path / f"zoom_line_sample_plot_{epoch}_before_post_processing.{plot_file_ending}",
        bbox_inches="tight",
        pad_inches=0,
    )

    weather_post_processor = DWDWeatherPostProcessor()
    dataframe = weather_post_processor(dataframe)
    fig, axes = __draw_weather_data_zoom_plot_sample_by_df(dataframe)
    fig.savefig(result_path / f"zoom_line_sample_plot_{epoch}.{plot_file_ending}", bbox_inches="tight", pad_inches=0)


def eval(path: PurePath, epoch: int):
    manualSeed = 1337
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    set_latex_plot_params()
    sns.set_palette("deep", color_codes=True)

    result_path = path / "results" / str(epoch)
    result_path.mkdir(parents=True, exist_ok=True)

    model_path = path / "models" / str(epoch)
    evaluator = Evaluator.load(model_path)

    # save_sample_zoom_line_plot(evaluator=evaluator, result_path=result_path, epoch=epoch)
    ks_test_on_every_feature(evaluator=evaluator)


if __name__ == "__main__":
    simulation_data_gans_path = get_experiments_folder().joinpath("03_simulation_data_gans")
    conditonal_cgan_path = simulation_data_gans_path.joinpath("03_01_conditional_gan_simulation_data")
    fnn_features_all_path = conditonal_cgan_path / "fnn_features_sun_temp"

    eval(path=fnn_features_all_path, epoch=100)
    plt.show()
