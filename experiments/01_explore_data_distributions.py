import seaborn as sns

from data.importer.weather.weather_dwd_importer import DWDWeatherDataImporter, WeatherDataColumns, WEATHER_DATA_MAPPING

from experiments.utils import get_experiments_folder

from src.plots.distribution_fit_plot import DistributionPlotColumn, draw_best_fit_plot
from src.plots.typing import PlotOptions

if __name__ == '__main__':
    sns.set_theme()
    sns.set_context("paper")
    explore_dists_folder = get_experiments_folder().joinpath("01_explore_data_distributions")
    explore_dists_folder.mkdir(parents=True, exist_ok=True)
    importer = DWDWeatherDataImporter(end_date="2009-12-31 23:00:00")
    importer.initialize()
    # extract all used targetColumns
    used_target_columns = [col_map.targetColumn for nested in WEATHER_DATA_MAPPING.values() for col_map in nested.columns]
    target_column_extra_info: dict[str, DistributionPlotColumn] = {
        WeatherDataColumns.T_AIR_DEGREE_CELSIUS: DistributionPlotColumn(
            plot_options=PlotOptions(x_label="Temperatur in °C"),
            extra_dist_plots=["norm", "gennorm"],
            legend_spacing=True
        ),
        WeatherDataColumns.DH_W_PER_M2: DistributionPlotColumn(
            plot_options=PlotOptions(x_label="Stundensumme der diffusen Solarstrahlung"),
            extra_dist_plots=["exponential", "beta"],
        ),
        WeatherDataColumns.GH_W_PER_M2: DistributionPlotColumn(
            plot_options=PlotOptions(x_label="Stundensumme der Globalstrahlung"),
            extra_dist_plots=["exponential", "beta"],
        ),
        WeatherDataColumns.WIND_V_M_PER_S: DistributionPlotColumn(
            plot_options=PlotOptions(x_label="Windgeschwindigkeit in m/s"),
            extra_dist_plots=["rayleigh"],
        ),
        WeatherDataColumns.WIND_DIR_DEGREE: DistributionPlotColumn(
            plot_options=PlotOptions(x_label="Windrichtung in Grad (0 - 359)"),
            extra_dist_plots=["uniform", "random"],
            bins=36,  # Since 360 degree and only in steps*10
            legend_spacing=True
        ),
        WeatherDataColumns.CLOUD_PERCENT: DistributionPlotColumn(
            plot_options=PlotOptions(x_label="Prozentuale Wolkenbedeckung in %"),
            extra_dist_plots=[]
        ),
        WeatherDataColumns.SUN_HOURS_MIN_PER_H: DistributionPlotColumn(
            plot_options=PlotOptions(x_label="Sonnenstunden min/h"),
            extra_dist_plots=[]
        ),
    }

    data_info = [(column, target_column_extra_info[column]) for column in used_target_columns if target_column_extra_info[column] is not None]

    for target_column, column_plot_metadata in data_info:
        data = importer.data[target_column]
        fit_res = draw_best_fit_plot(
            data=data,
            plot_metadata=column_plot_metadata,
        )
        # save file
        path = explore_dists_folder.joinpath(f"{target_column}.pdf").absolute()
        fit_res.plot_res.fig.show()
        fit_res.plot_res.fig.savefig(path, bbox_inches='tight')
