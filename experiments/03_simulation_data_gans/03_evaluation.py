from collections import defaultdict
from datetime import timedelta, datetime
from pathlib import PurePath
from typing import Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import pandas as pd

import seaborn as sns

import random
from matplotlib import pyplot as plt
import torch
from scipy import stats
from statsmodels.distributions import ECDF
from statsmodels.tsa.seasonal import seasonal_decompose

from experiments.experiments_utils.utils import get_experiments_folder, set_latex_plot_params

from experiments.experiments_utils.plotting import (
    draw_weather_data_zoom_plot_sample,
    plot_model_losses,
    plot_sample,
    save_fig,
)
from experiments.experiments_utils.weather_data_translations import (
    WEATHER_LABEL_MAP,
    WEATHER_UNIT_LATEX_MAP,
    get_weather_data_latex_label,
)
from src.data.weather.weather_dwd_postprocessor import DWDWeatherPostProcessor
from src.data.weather.weather_filters import create_night_time_replace_handler
from src.evaluator.evaluator import Evaluator

from src.data.weather.weather_dwd_importer import DEFAULT_DATA_START_DATE, DWDWeatherDataImporter, WeatherDataColumns
from src.metrics.kolmogorov_smirnov import ks2_critical_value, ks2_test
from src.metrics.pca import dimension_reduction_visualization
from src.metrics.wasserstein_distance import wasserstein_dist
from src.plots.ecdf_plot import ECDFPlotData, draw_ecdf_plot
from src.plots.histogram_plot import HistPlotData, draw_hist_plot
from src.plots.qq_plot import QQReferenceLine, draw_qq_plot
from src.plots.timeseries_decomposition_plot import (
    GERMAN_LATEX_TRANSLATIONS,
    DecomposeResultColumns,
    draw_timeseries_decomposition_plot,
)
from src.plots.typing import PlotData, PlotOptions, PlotResult

from src.plots.zoom_line_plot import ConnectorBoxOptions, ZoomBoxEffectOptions, ZoomPlotOptions, draw_zoom_line_plot
from src.utils.datetime_utils import (
    convert_input_str_to_date,
    dates_to_conditional_vectors,
    get_day_in_year_from_date,
    interval_generator,
)

DEFAULT_FILE_ENDING = "pdf"


def save_hist_plots_on_every_feature(
    evaluator: Evaluator, path: PurePath, result_path: PurePath, epoch: int, plot_file_ending=DEFAULT_FILE_ENDING
):
    def __draw_violin_plot_every_hour(
        sample_a: pd.Series, sample_b: pd.Series, col: WeatherDataColumns, plot: Optional[PlotResult] = None
    ):
        scale = 1.5
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(scale * 5, scale * 1)) if plot is None else plot
        sequence_len = 24
        sample_a_len = len(sample_a.index)
        sample_b_len = len(sample_b.index)
        if sample_a_len % sequence_len != 0 or sample_b_len % sequence_len != 0:
            raise ValueError(f"The samples have to have be divisible by ${sequence_len} without a remainder.")

        gen_col_type = np.repeat("Generatorausgabe", sample_a_len)
        theoretical_col_type = np.repeat("Trainingsdaten", sample_b_len)
        # hour_values = np.arange(sequence_len)
        # hours = np.tile(hour_values, (sample_a_len + sample_b_len) // sequence_len)
        # Workaround forgot to include the timezone
        sample_a_hours = np.mod((sample_a.index.hour.values + 1), sequence_len)
        sample_b_hours = np.mod((sample_b.index.hour.values + 1), sequence_len)
        tips = pd.DataFrame(
            {
                "hour": np.concatenate([sample_a_hours, sample_b_hours]),
                "type": np.concatenate([gen_col_type, theoretical_col_type]),
                "values": np.concatenate([sample_a, sample_b]),
            }
        )
        ax = sns.violinplot(x="hour", y="values", hue="type", data=tips, split=True, linewidth=0.75, ax=ax)
        # Remove crappy legend title hack with labels...
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles[1:], labels=labels[1:])
        # ax.legend(loc="best")
        ax.legend().set_visible(False)
        ax.set_xlabel(r"$[t]_{s}$")
        ax.set_ylabel(get_weather_data_latex_label(col))

        return fig, ax

    def __draw_qq_plot(
        sample_a: npt.ArrayLike, sample_b: npt.ArrayLike, col: WeatherDataColumns, plot: Optional[PlotResult] = None
    ):
        plot_res = draw_qq_plot(
            PlotData(data=sample_a, label="Generatorausgabe"),
            PlotData(data=sample_b, label="Trainingsdaten"),
            1000,
            {
                QQReferenceLine.THEORETICAL_LINE,
                # QQReferenceLine.FIRST_THIRD_QUARTIL,
                # QQReferenceLine.LEAST_SQUARES_REGRESSION,
            },
            [0.25, 0.5, 0.75],
            plot_options=PlotOptions(
                title=get_weather_data_latex_label(col)
                # x_label=get_weather_data_latex_label(col), y_label=r"Relative Häufigkeitsdichte $P(x)$"
            ),
            plot=plot,
        )
        return plot_res.fig, plot_res.ax

    def __draw_timeseries_decomposition_plot(series: pd.Series, col: WeatherDataColumns):
        year_period = 8766
        size = 0.65 * 6.4
        figsize = (size, size)
        decomp_result_year_only = seasonal_decompose(series, model="additive", period=year_period)
        translations = {
            **GERMAN_LATEX_TRANSLATIONS,
            DecomposeResultColumns.OBSERVED: r"$\displaystyle{\text{" + WEATHER_LABEL_MAP[col] + r"}\;Y_t}$",
        }
        plot_res = draw_timeseries_decomposition_plot(
            data=decomp_result_year_only, translations=translations, figsize=figsize, rasterized=True
        )
        return plot_res.fig, plot_res.ax

    def __draw_single_hist_plot(
        sample_a: npt.ArrayLike, sample_b: npt.ArrayLike, col: WeatherDataColumns, plot: Optional[PlotResult] = None
    ):
        label = WEATHER_LABEL_MAP[col]
        fig, ax = draw_hist_plot(
            [
                HistPlotData(data=sample_a, label=f"Generatorausgabe"),
                HistPlotData(data=sample_b, label="Trainingsdaten"),
            ],
            bins=col_hist_options.get("bins", 100),
            normalized=True,
            plot=plot,
            plot_options=PlotOptions(
                x_label=get_weather_data_latex_label(col), y_label=r"Relative Häufigkeitsdichte $P(x)$"
            ),
        )
        return fig, ax

    def __draw_single_ecdf_plot(
        sample_a: npt.ArrayLike, sample_b: npt.ArrayLike, col: WeatherDataColumns, plot: Optional[PlotResult] = None
    ):
        label = WEATHER_LABEL_MAP[col]
        fig, ax = draw_ecdf_plot(
            [
                ECDFPlotData(
                    data=ECDF(sample_b),
                    label="Trainingsdaten",
                    confidence_band_alpha=0.05,
                    confidence_band_fill_alpha=0.3,
                    confidence_band_label_supplier=lambda alpha: f"{alpha}" + r"\% Konfidenzband",
                ),
                ECDFPlotData(
                    data=ECDF(sample_a),
                    label=f"Generatorausgabe",
                    confidence_band_alpha=0.00,
                    confidence_band_fill_alpha=0.3,
                ),
            ],
            plot=plot,
            plot_options=PlotOptions(
                x_label=get_weather_data_latex_label(col), y_label=r"Empirische Verteilungsfunktion (ECDF)"
            ),
        )
        return fig, ax

    start = datetime.fromisoformat("2009-01-01T00:00:00")
    end = datetime.fromisoformat("2019-12-31T23:00:00")
    weather_post_processor = DWDWeatherPostProcessor()
    importer = DWDWeatherDataImporter(start_date=start, end_date=end)
    importer.initialize()
    # for epoch in range(1000, 1050, 10):
    start_generator = datetime.fromisoformat("2020-01-01T00:00:00")
    end_generator = datetime.fromisoformat("2030-12-31T23:00:00")
    raw_dataframe = evaluator.generate_dataframe(start_generator, end_generator)
    dataframe = weather_post_processor(raw_dataframe)
    exclude_night_time_values = create_night_time_replace_handler()
    special_hist_plot_column_options: dict[WeatherDataColumns, dict[str, any]] = {
        WeatherDataColumns.GH_W_PER_M2: {"transformer": exclude_night_time_values},
        WeatherDataColumns.DH_W_PER_M2: {"transformer": exclude_night_time_values},
        WeatherDataColumns.WIND_DIR_DEGREE: {"bins": 36},
    }

    model_path = path / "models" / str(epoch)
    evaluator = Evaluator.load(model_path)

    # Can be used to create a multi grid figure to display all at once! Just pass plot=PlotResult(hist_fig, hist_axes[col] into the draw methods
    def create_mosaic_figure() -> tuple[Figure, Axes]:
        fig = plt.figure(figsize=(6.4 * 3, 4.8 * 2))  # figsize=(12, 2.4 * len(raw_plot_data_rows)))
        axs = fig.subplot_mosaic(
            [
                [
                    WeatherDataColumns.WIND_DIR_DEGREE,
                    WeatherDataColumns.WIND_V_M_PER_S,
                    WeatherDataColumns.T_AIR_DEGREE_CELSIUS,
                ],
                [WeatherDataColumns.GH_W_PER_M2, WeatherDataColumns.DH_W_PER_M2, WeatherDataColumns.DH_W_PER_M2],
            ],
        )
        return fig, axs

    multi_fig, multi_axes = create_mosaic_figure()
    multi_fig.suptitle(str(epoch))

    hist_result_path = result_path / "hist"
    hist_result_path.mkdir(parents=True, exist_ok=True)
    raw_hist_result_path = hist_result_path / "raw"
    raw_hist_result_path.mkdir(parents=True, exist_ok=True)
    ecdf_result_path = result_path / "ecdf"
    ecdf_result_path.mkdir(parents=True, exist_ok=True)
    decomp_result_path = result_path / "decomp"
    decomp_result_path.mkdir(parents=True, exist_ok=True)
    violin_result_path = result_path / "violin"
    violin_result_path.mkdir(parents=True, exist_ok=True)
    qq_result_path = result_path / "qq_plot"
    qq_result_path.mkdir(parents=True, exist_ok=True)

    for col in dataframe.columns:
        sample_a = dataframe[col]
        sample_b = importer.data[col]
        col_hist_options = special_hist_plot_column_options.get(col, {})
        print(f"{col=}, {col_hist_options}")
        if col_hist_options is not None:
            transformer = col_hist_options.get("transformer")
            if transformer is not None:
                sample_a = transformer(dataframe[col])
                sample_b = transformer(importer.data[col])

            # raw_hist_fig, _ = __draw_single_hist_plot(
            #     sample_a=raw_dataframe[col].values, sample_b=importer.data[col].values, col=col
            # )
            # raw_hist_fig.savefig(
            #     raw_hist_result_path
            #     / f"hist_plot_data_comparison_{col}_{epoch}_before_post_processing.{plot_file_ending}",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # hist_fig, _ = __draw_single_hist_plot(sample_a=sample_a.values, sample_b=sample_b.values, col=col)
            # hist_fig.savefig(
            #     hist_result_path / f"hist_plot_data_comparison_{col}_{epoch}.{plot_file_ending}",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # ecdf_fig, _ = __draw_single_ecdf_plot(sample_a=sample_a.values, sample_b=sample_b.values, col=col)
            # ecdf_fig.savefig(
            #     ecdf_result_path / f"ecdf_plot_data_comparison_{col}_{epoch}.{plot_file_ending}",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            # decomp_fig, _ = __draw_timeseries_decomposition_plot(series=dataframe[col], col=col)
            # decomp_fig.savefig(
            #     decomp_result_path / f"decomp_plot_{col}_{epoch}.{plot_file_ending}",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            violin_fig, _ = __draw_violin_plot_every_hour(sample_a=sample_a, sample_b=sample_b, col=col)
            violin_fig.savefig(
                violin_result_path / f"violin_plot_hourly_{col}_{epoch}.{plot_file_ending}",
                bbox_inches="tight",
                pad_inches=0,
            )
            # qq_fig, _ = __draw_qq_plot(
            #     sample_a=sample_a, sample_b=sample_b, col=col, plot=PlotResult(multi_fig, multi_axes[col])
            # )
            # qq_fig.savefig(
            #     qq_result_path / f"qq_plot_{col}_{epoch}.{plot_file_ending}",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )


def ks_test_on_every_feature(evaluator: Evaluator):
    start = datetime.fromisoformat("2009-01-01T00:00:00")
    end = datetime.fromisoformat("2019-12-31T23:00:00")
    weather_post_processor = DWDWeatherPostProcessor()
    importer = DWDWeatherDataImporter(start_date=start, end_date=end)
    importer.initialize()
    generated_dataframe_samples: list[pd.DataFrame] = []
    ks_test_values = defaultdict(list)
    wasserstein_distances = defaultdict(list)
    t_test_pvalues = defaultdict(list)
    sample_count = 50
    ks_crit = None
    ks_n, ks_m = None, None
    ks_alpha = 0.05
    for take in range(sample_count):
        start_generated = datetime.fromisoformat("2023-01-01T00:00:00")
        end_generated = datetime.fromisoformat("2023-12-31T23:00:00")
        dataframe = evaluator.generate_dataframe(start_generated, end_generated)
        dataframe = weather_post_processor(dataframe)
        generated_dataframe_samples.append(dataframe)
        for col in dataframe.columns:
            sample_a = dataframe[col].values
            sample_b = importer.data[col].values

            # median_test_log_like_res = median_test(sample_a, sample_b, lambda_="log-likelihood")
            # print(f"{col=} -> {median_test_log_like_res=}")

            ttest_statistic, ttest_p_value = stats.ttest_ind(sample_a, sample_b, equal_var=True, trim=0.2)
            t_test_pvalues[col].append(ttest_p_value)

            ks_D, ks_p = ks2_test(sample_a=sample_a, sample_b=sample_b)
            ks_test_values[col].append(ks_D)
            ks_n, ks_m = len(sample_a), len(sample_b)

            wasserstein_distance = wasserstein_dist(sample_a=sample_a, sample_b=sample_b)
            wasserstein_distances[col].append(wasserstein_distance)
            # print(f"{col} -> {ks_test=} -> {ks_crit=}, {wasserstein_distance=}")
            if take == (sample_count - 1):
                ks_crit = ks2_critical_value(sample_a=sample_a, sample_b=sample_b, alpha=ks_alpha)

    # # just print some results
    # for col in dataframe.columns:
    #     t_test_values = np.array(t_test_pvalues[col])
    #     t_test_values_mean = np.mean(t_test_values)
    #     t_test_values_std = np.std(t_test_values)
    #     print(f"{col=}, {t_test_values_mean=}, {t_test_values_std=}")

    latex_table = r"\begin{table}[htb]" + "\n"
    latex_table += r"\begin{adjustbox}{center,max width=1.0\textwidth}" + "\n"
    latex_table += r"\begin{tabular}{|l|c|c|c|c|}" + "\n"
    latex_table += r"\cline{2-5}" + "\n"
    latex_table += (
        r"\multicolumn{1}{c}{} & \multicolumn{2}{|c|}{KS-Test} & \multicolumn{1}{c|}{Yuen T-Test} & \multicolumn{1}{c|}{Wasserstein-Metrik} \\ "
        + "\n"
    )
    # latex_table += r"\cline{2-4}" + "\n"
    latex_table += r"\hline" + "\n"
    latex_table += (
        r"\multicolumn{1}{|c|}{Merkmal} & $D_{n,m}$ & $c(\alpha ){\sqrt {\frac {n+m}{n\cdot m}}}$ & $p_{0.2}$ & $W_{1}(\mu ,\nu)$ \\"
        + "\n"
    )
    latex_table += r"\hline" + "\n"
    for col in dataframe.columns:
        col_name = WEATHER_LABEL_MAP[col]
        row = f"{col_name}&"
        ks_values = np.array(ks_test_values[col])
        ks_col_mean = np.mean(ks_values)
        ks_col_std = np.std(ks_values)
        row += f"${ks_col_mean:.4f}" + r" \pm " + f"{ks_col_std:.4f}$ &"
        row += f"{ks_crit:.4f}&"

        t_test_values = np.array(t_test_pvalues[col])
        t_test_values_mean = np.mean(t_test_values)
        t_test_values_std = np.std(t_test_values)
        row += f"${t_test_values_mean:.4f}" + r" \pm " + f"{t_test_values_std:.4f}$ &"

        w_values = np.array(wasserstein_distances[col])
        w_col_mean = np.mean(w_values)
        w_col_std = np.std(w_values)
        row += f"${w_col_mean:.4f}" + r" \pm " + f"{w_col_std:.4f}$"
        latex_table += row + r"\\" + "\n"
        # print(f"{col=}, {ks_col_mean=}, {ks_col_std=}, {ks_crit=}, {w_col_mean=}, {w_col_std=}")
    latex_table += r"\hline" + "\n"
    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\end{adjustbox}" + "\n"
    latex_table += (
        r"\caption{Ergebnisse des KS-Tests, dem Yuen T-Test und der Wasserstein-Metrik zwischen der Gesamtheit der Trainingsdaten $\mu$ von 10 Jahre und "
        + str(sample_count)
        + r" zufälligen Generatorausgaben $\nu_t$ eines nicht Schaltjahres unter Angabe des Mittelwertes und der Standardabweichung. Durch die Wahl von 10 Jahren und einem Jahr gilt $n="
        + str(ks_n)
        + r"$ und $m="
        + str(ks_m)
        + r"$"
        + r" und $\alpha = "
        + str(ks_alpha)
        + "$"
        + r".}"
        + "\n"
    )
    latex_table += r"\label{table:cgan_basic_metrics_evaluation}" + "\n"
    latex_table += r"\end{table}"
    print(latex_table)


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
            # cols_top=[WeatherDataColumns.T_AIR_DEGREE_CELSIUS],
            # cols_mid=[
            #     WeatherDataColumns.GH_W_PER_M2,
            #     WeatherDataColumns.DH_W_PER_M2,
            # ],
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
    # with sns.axes_style("darkgrid"):
    # save_hist_plots_on_every_feature(evaluator=evaluator, path=path, result_path=result_path, epoch=epoch)
    ks_test_on_every_feature(evaluator=evaluator)


if __name__ == "__main__":
    simulation_data_gans_path = get_experiments_folder().joinpath("03_simulation_data_gans")
    conditonal_cgan_path = simulation_data_gans_path.joinpath("03_01_conditional_gan_simulation_data")
    fnn_features_all_path = conditonal_cgan_path / "fnn_features_all"

    eval(path=fnn_features_all_path, epoch=320)
    plt.show()
