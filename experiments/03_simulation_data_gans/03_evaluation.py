from datetime import timedelta, datetime
from pathlib import PurePath
import numpy as np
import pandas as pd

import random
from matplotlib import pyplot as plt
import torch

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
from src.plots.typing import PlotData

from src.plots.zoom_line_plot import ConnectorBoxOptions, ZoomBoxEffectOptions, ZoomPlotOptions, draw_zoom_line_plot
from src.utils.datetime_utils import (
    convert_input_str_to_date,
    dates_to_conditional_vectors,
    get_day_in_year_from_date,
    interval_generator,
)


def eval(path: PurePath, epoch: int, plot_file_ending="pdf"):
    manualSeed = 1337
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    set_latex_plot_params()

    result_path = path / "results" / str(epoch)
    result_path.mkdir(parents=True, exist_ok=True)

    model_path = path / "models" / str(epoch)
    evaluator = Evaluator.load(str(model_path))
    start = datetime.fromisoformat("2023-01-01T00:00:00")
    end = datetime.fromisoformat("2023-12-31T23:00:00")
    generated_data = evaluator.generate(start, end)
    # train_params = ConditionalTrainParameters(batch_size=32, embedding_dim=32)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(30, 1))
    # (fig, ax) = plot_sample(
    #     sample=generated_data, params=train_params, plot=(fig, ax), condition="ALL", generate_single_features=False
    # )

    seperated_generated_data = torch.unbind(generated_data, -1)
    dataframe = pd.DataFrame(
        index=pd.date_range(
            start=start,
            end=end,
            tz="Europe/Berlin",
            freq="h",
        )
    )

    for idx, data in enumerate(seperated_generated_data):
        dataframe[evaluator.feature_labels[idx].label] = data.view(-1).numpy()

    def draw_weather_data_zoom_plot_sample_by_df(dataframe):
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

    fig, axes = draw_weather_data_zoom_plot_sample_by_df(dataframe)
    fig.savefig(
        result_path / f"zoom_line_sample_plot_{epoch}_before_post_processing.{plot_file_ending}",
        bbox_inches="tight",
        pad_inches=0,
    )

    weather_post_processor = DWDWeatherPostProcessor()
    dataframe = weather_post_processor(dataframe)
    fig, axes = draw_weather_data_zoom_plot_sample_by_df(dataframe)
    fig.savefig(result_path / f"zoom_line_sample_plot_{epoch}.{plot_file_ending}", bbox_inches="tight", pad_inches=0)

    plt.close(fig)


if __name__ == "__main__":
    simulation_data_gans_path = get_experiments_folder().joinpath("03_simulation_data_gans")
    conditonal_cgan_path = simulation_data_gans_path.joinpath("03_01_conditional_gan_simulation_data")
    fnn_features_all_path = conditonal_cgan_path / "fnn_features_all"

    eval(path=fnn_features_all_path, epoch=500)
    plt.show()
