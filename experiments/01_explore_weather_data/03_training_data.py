import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params
from experiments.experiments_utils.weather_data_translations import WEATHER_LABEL_MAP
from src.data.data_holder import DataHolder
from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer

from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns
from src.plots.timeseries_decomposition_plot import (
    GERMAN_LATEX_TRANSLATIONS,
    DecomposeResultColumns,
    draw_timeseries_decomposition_plot,
)
from src.utils.datetime_utils import dates_to_conditional_vectors
import seaborn as sns
from statsmodels.tsa.seasonal import STL, seasonal_decompose


def save_fig(fig, path):
    fig.canvas.draw()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    sns.set_theme(context="paper", style="darkgrid")
    set_latex_plot_params()
    # sns.set_palette("deep", color_codes=True)
    explore_training_data_root_folder = (
        get_experiments_folder().joinpath("01_explore_weather_data").joinpath("03_training_data")
    )
    explore_training_data_root_folder.mkdir(parents=True, exist_ok=True)
    start_date_str = DEFAULT_DATA_START_DATE
    end_date_str = "2019-12-31 23:00:00"
    data_importer = DWDWeatherDataImporter(start_date=start_date_str, end_date=end_date_str)
    data_importer.initialize()
    print(f"Imported data with shape: {data_importer.data.shape}")
    day_period = 24
    year_period = 8766
    columns_to_use = set(
        [
            WeatherDataColumns.DH_W_PER_M2,
            WeatherDataColumns.GH_W_PER_M2,
            WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
            WeatherDataColumns.WIND_V_M_PER_S,
            WeatherDataColumns.WIND_DIR_DEGREE,
        ]
    )
    data_holder = DataHolder(
        data=data_importer.get_data_subset(columns_to_use).values.astype(np.float32),
        # data_labels=data_importer.get_feature_labels(),
        dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
        # conditions=conditions,
        normalizer_constructor=StandardNumpyNormalizer,
    )
    print(f"Loaded data into dataholder")
    decomp_save_path = explore_training_data_root_folder / "decomp"
    decomp_save_path.mkdir(parents=True, exist_ok=True)
    print(f"Normal decomposition")
    size = 0.65 * 6.4
    figsize = (size, size)
    for (column_name, values_normalized) in zip(data_importer.data, data_holder.data):
        print(f"Calculation decomposition for {column_name}")

        decomp_result_year_only = seasonal_decompose(
            data_importer.data[column_name], model="additive", period=year_period
        )
        print(f"Yearly only done")
        translations = {
            **GERMAN_LATEX_TRANSLATIONS,
            DecomposeResultColumns.OBSERVED: r"$\displaystyle{\text{" + WEATHER_LABEL_MAP[column_name] + r"}\;Y_t}$",
        }
        res_year_only = draw_timeseries_decomposition_plot(
            data=decomp_result_year_only, translations=translations, figsize=figsize, rasterized=True
        )
        save_fig(res_year_only.fig, decomp_save_path / f"data_{column_name}_year_only.pdf")

        # decomp_result_daily = seasonal_decompose(
        #     data_importer.data[column_name], model="additive", period=day_period
        # )  # STL(data_importer.data[column_name], period=day_period).fit()
        # res_daily = draw_timeseries_decomposition_plot(
        #     data=decomp_result_daily, translations=GERMAN_LATEX_TRANSLATIONS, figsize=figsize, rasterized=True
        # )
        # save_fig(res_daily.fig, decomp_save_path / f"data_{column_name}_daily.pdf")

        # print(f"Daily done")
        # year_data_input = data_importer.data[column_name] - np.array(decomp_result_daily.seasonal)  # .reshape(-1, 1)
        # print(f"{year_data_input.shape=}")
        # decomp_result_year = seasonal_decompose(
        #     year_data_input, model="additive", period=year_period
        # )  # STL(year_data_input, period=year_period).fit()
        # print(f"Yearly done")
        # res = draw_timeseries_decomposition_plot(
        #     data=decomp_result_year, translations=GERMAN_LATEX_TRANSLATIONS, figsize=figsize, rasterized=True
        # )
        # save_fig(res.fig, decomp_save_path / f"data_{column_name}.pdf")

    # print(f"STL decomposition")
    # stl_save_path = explore_training_data_root_folder / "stl"
    # stl_save_path.mkdir(parents=True, exist_ok=True)
    # for (column_name, values_normalized) in zip(data_importer.data, data_holder.data):
    #     print(f"Calculation stl for {column_name}")
    #     decomp_result_daily = STL(data_importer.data[column_name], period=365).fit()
    #     res = draw_timeseries_decomposition_plot(data=decomp_result_year, translations=translations, figsize=(6.4, 6.4))
    #     save_fig(res.fig, stl_save_path / f"data_{column_name}.pdf")
