import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params
from experiments.experiments_utils.weather_data_translations import (
    WEATHER_LABEL_MAP,
    WEATHER_UNIT_LATEX_MAP,
    get_weather_data_latex_label,
)

from src.data.distribution.distribution_fit import DistributionFit
from src.data.weather.weather_dwd_importer import (
    DWDWeatherDataImporter,
    WeatherDataColumns,
    WEATHER_DATA_MAPPING,
    DEFAULT_DATA_START_DATE,
)
from src.data.weather.weather_filters import create_night_time_replace_handler
from src.metrics.kullback_leibler import kl_divergence
from src.metrics.r_squared import r_squared
from src.metrics.wasserstein_distance import wasserstein_dist
from src.plots.distribution_fit_plot import (
    DistributionPlotColumn,
    draw_best_fit_plot,
    default_distribution_legend_label_provider,
    __Keys,
)
from src.plots.typing import PlotOptions, PlotResult


@dataclass
class DistributionFitOptions:
    error_fn_name: str
    error_fn: tuple[str, Callable[[npt.ArrayLike, npt.ArrayLike], float]] = (("R²", r_squared),)
    best_score_finder: Callable[[list[DistributionFit]], DistributionFit] = (max,)
    distribution_names: Optional[list[str]] = None


# Already fitted the data against all possible distributions with R^2 and these are the ones which were fitted the best over all samples,
# specify them to speed up rerenderings!
distribution_names = [
    "gennorm",
    "norm",
    "triang",
    "uniform",
    "burr12",
    "rayleigh",
    "alpha",
    "beta",
    "chi",
    "weibull_min",
    "exponpow",
]

latex_de_translations: dict[__Keys, str] = {
    __Keys.LEGEND_DATA_DEFAULT: r"Daten aufgeteilt in {X} Klassen",
    __Keys.Y_LABEL_DEFAULT: r"Relative Häufigkeitsdichte $P(x)$",
    __Keys.X_LABEL_DEFAULT: r"Wert",
}


def latex_distribution_legend_label_provider(fit: DistributionFit, error_label: str) -> str:
    return default_distribution_legend_label_provider(fit, error_label).replace(r"_", r"\_")


def kl_adjusted(p, q) -> float:
    q_zero_indexes = np.where(q == 0)
    q_zeroes_but_p_not = np.where(p[q_zero_indexes] != 0)
    if len(q_zeroes_but_p_not[0]) > 0:
        return math.inf
    return kl_divergence(q, p, math.e)

    # p_zero_indexes = np.where(p == 0)
    # q_zero_indexes = np.where(q == 0)
    # q[q_zero_indexes] = 1e-8  # to not divide by zero!
    # q_zeroes_but_p_not = np.where(p[q_zero_indexes] != 0)
    # both_zero_indexes = np.setdiff1d(p_zero_indexes, q_zero_indexes)
    # p[both_zero_indexes] = 1e8
    # return kl_divergence(p, q, math.e)


def explore_data_distributions(options: DistributionFitOptions):
    sns.set_theme()
    sns.set_context("paper")
    set_latex_plot_params()
    explore_dists_root_folder = (
        get_experiments_folder().joinpath("01_explore_weather_data").joinpath("01_explore_data_distributions")
    )
    start_date = DEFAULT_DATA_START_DATE
    end_date = "2019-12-31 23:00:00"

    start_date_path = start_date.split()[0].replace("-", "_")
    end_date_path = end_date.split()[0].replace("-", "_")
    explore_dists_folder = explore_dists_root_folder.joinpath(f"{start_date_path}_{end_date_path}").joinpath(
        options.error_fn_name
    )
    explore_dists_folder.mkdir(parents=True, exist_ok=True)
    importer = DWDWeatherDataImporter(start_date=start_date, end_date=end_date)
    importer.initialize()
    # extract all used targetColumns

    exclude_night_time_values = create_night_time_replace_handler()

    target_column_extra_info: dict[str, list[DistributionPlotColumn]] = {
        WeatherDataColumns.T_AIR_DEGREE_CELSIUS: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=get_weather_data_latex_label(WeatherDataColumns.T_AIR_DEGREE_CELSIUS)),
                extra_dist_plots=["norm", "gennorm"],
                legend_spacing=(0.0, 0.08),
            )
        ],
        WeatherDataColumns.DH_W_PER_M2: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=get_weather_data_latex_label(WeatherDataColumns.DH_W_PER_M2)),
                extra_dist_plots=[],
            ),
            DistributionPlotColumn(
                plot_options=PlotOptions(
                    x_label=get_weather_data_latex_label(WeatherDataColumns.DH_W_PER_M2) + " (lichter Tag)."
                ),
                extra_dist_plots=["weibull_min", "beta"],
                transformer=lambda x: exclude_night_time_values(x),
            ),
        ],
        WeatherDataColumns.GH_W_PER_M2: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=get_weather_data_latex_label(WeatherDataColumns.GH_W_PER_M2)),
                extra_dist_plots=[],
            ),
            DistributionPlotColumn(
                plot_options=PlotOptions(
                    x_label=get_weather_data_latex_label(WeatherDataColumns.GH_W_PER_M2) + " (lichter Tag)."
                ),
                transformer=lambda x: exclude_night_time_values(x),
                extra_dist_plots=["weibull_min", "beta"],
            ),
        ],
        WeatherDataColumns.WIND_V_M_PER_S: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=get_weather_data_latex_label(WeatherDataColumns.WIND_V_M_PER_S)),
                extra_dist_plots=["rayleigh"],
                legend_spacing=(0.0, 0.08),
            )
        ],
        WeatherDataColumns.WIND_DIR_DEGREE: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=get_weather_data_latex_label(WeatherDataColumns.WIND_DIR_DEGREE)),
                extra_dist_plots=["uniform", "random"],
                bins=36,  # Since 360 degree and only in steps*10
                legend_spacing=(0.0, 0.08),
            )
        ],
        WeatherDataColumns.WIND_DIR_DEGREE_DELTA: [
            DistributionPlotColumn(
                plot_options=PlotOptions(
                    x_label=get_weather_data_latex_label(WeatherDataColumns.WIND_DIR_DEGREE_DELTA)
                    + r" pro Stunde $[-180^{\circ}, 180^{\circ}]$"
                ),
                extra_dist_plots=["norm"],
                bins=36,  # Since 360 degree and only in steps*10
                legend_spacing=(0.0, 0.08),
            )
        ],
        WeatherDataColumns.CLOUD_PERCENT: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=r"Prozentuale Wolkenbedeckung in \%"), extra_dist_plots=[]
            )
        ],
        WeatherDataColumns.SUN_HOURS_MIN_PER_H: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=r"Sonnenstunden $\frac{min}{h}$"), extra_dist_plots=[]
            )
        ],
    }
    extra_mappings_target_columns = [
        extra_column.targetColumn
        for nested in WEATHER_DATA_MAPPING.values()
        for col_map in nested.columns
        for extra_column in col_map.extraMappings
    ]
    target_columns = [col_map.targetColumn for nested in WEATHER_DATA_MAPPING.values() for col_map in nested.columns]
    used_target_columns = extra_mappings_target_columns + target_columns

    data_info = [
        (column, target_column_extra_info[column])
        for column in used_target_columns
        if column in target_column_extra_info
    ]

    for target_column, column_plot_metadata_entries in data_info:
        data = importer.data[target_column]
        for idx, column_plot_metadata in enumerate(column_plot_metadata_entries):
            if column_plot_metadata.transformer is not None:
                data = column_plot_metadata.transformer(data)
            scale = 0.75
            margin_x = 0.5
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(scale * 6.4 + margin_x / scale, scale * 4.8))
            # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4 + margin, 4.8 + margin))
            fit_res = draw_best_fit_plot(
                data=data,
                plot_metadata=column_plot_metadata,
                error_fn=options.error_fn,
                best_score_finder=options.best_score_finder,
                distribution_legend_label_provider_fn=latex_distribution_legend_label_provider,
                translations=latex_de_translations,
                distribution_names=options.distribution_names,
                plot=PlotResult(fig, ax),
            )
            # save file
            path = explore_dists_folder.joinpath(f"{target_column}_{idx}.pdf").absolute()
            fit_res.plot_res.fig.show()
            fit_res.plot_res.fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    explore_data_distributions(
        DistributionFitOptions(
            error_fn_name="r_squared",
            error_fn=("$R^2$", r_squared),
            best_score_finder=max,
            distribution_names=distribution_names,
        )
        # DistributionFitOptions(
        #     error_fn_name="wasserstein",
        #     error_fn=("$W_{1}$", wasserstein_dist),
        #     best_score_finder=min,
        #     # distribution_names=distribution_names
        # )
        # DistributionFitOptions(
        #     error_fn_name="kl_divergence",
        #     error_fn=("$D_{KL}$", kl_adjusted),
        #     best_score_finder=min,
        #     distribution_names=distribution_names
        # )
    )
