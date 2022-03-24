from matplotlib import pyplot as plt
import numpy as np
from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params
from src.data.data_holder import DataHolder
from src.data.normalization.np.standard_normalizer import StandardNumpyNormalizer

from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns
from src.plots.timeseries_plot import DecomposeResultColumns, draw_timeseries_plot
from src.utils.datetime_utils import dates_to_conditional_vectors
import seaborn as sns
from statsmodels.tsa.seasonal import STL


def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    sns.set_theme()
    sns.set_context("paper")
    set_latex_plot_params()
    explore_training_data_root_folder = (
        get_experiments_folder().joinpath("01_explore_weather_data").joinpath("03_training_data")
    )

    start_date_str = DEFAULT_DATA_START_DATE
    end_date_str = "2019-12-31 23:00:00"  # "2019-12-31 23:00:00"
    data_importer = DWDWeatherDataImporter(start_date=start_date_str, end_date=end_date_str)
    data_importer.initialize()
    columns_to_use = set(
        [
            WeatherDataColumns.DH_W_PER_M2,
            WeatherDataColumns.GH_W_PER_M2,
            WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
            WeatherDataColumns.WIND_DIR_DEGREE_DELTA,
        ]
    )
    data_holder = DataHolder(
        data=data_importer.get_data_subset(columns_to_use).values.astype(np.float32),
        # data_labels=data_importer.get_feature_labels(),
        dates=np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
        # conditions=conditions,
        normalizer_constructor=StandardNumpyNormalizer,
    )

    for (column_name, values_normalized) in zip(data_importer.data, data_holder.data):
        decomp_result = STL(data_importer.data[column_name], period=365).fit()
        decomp_result2 = STL(values_normalized, period=365).fit()
        translations = {
            DecomposeResultColumns.OBSERVED: r"$\displaystyle{\text{Daten}\;Y_t}$",
            DecomposeResultColumns.SEASONAL: r"$\displaystyle{\text{Saisonal}\;S_t}$",
            DecomposeResultColumns.TREND: r"$\displaystyle{\text{Trend}\;T_t}$",
            DecomposeResultColumns.RESID: r"$\displaystyle{\text{Rest}\;R_t}$",
            DecomposeResultColumns.WEIGHTS: r"$\displaystyle{\text{Gewichte}\;W_t}$",
        }
        res = draw_timeseries_plot(data=decomp_result, translations=translations, figsize=(6.4, 6.4))
        save_fig(res.fig, explore_training_data_root_folder / f"data_{column_name}.pdf")
        res2 = draw_timeseries_plot(data=decomp_result2, translations=translations, figsize=(6.4, 6.4))
        save_fig(res2.fig, explore_training_data_root_folder / f"data_normed_{column_name}.pdf")
