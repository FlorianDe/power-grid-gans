import seaborn as sns
from matplotlib import pyplot as plt

from data.distribution.distribution_fit import DistributionFit
from experiments.utils import get_experiments_folder, set_latex_plot_params
from metrics.r_squared import r_squared
from src.data.weather.weather_dwd_importer import DWDWeatherDataImporter, WeatherDataColumns, WEATHER_DATA_MAPPING, DEFAULT_DATA_START_DATE
from src.data.weather.weather_filters import exclude_night_time_values
from src.plots.distribution_fit_plot import DistributionPlotColumn, draw_best_fit_plot, default_distribution_legend_label_provider, __Keys
from src.plots.typing import PlotOptions, PlotResult

# Already fitted the data against all possible distributions but these are the ones which were used, specify them to speed up rerenderings!
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
    __Keys.X_LABEL_DEFAULT: r"Wert"
}

def latex_distribution_legend_label_provider(fit: DistributionFit, error_label: str) -> str:
    return default_distribution_legend_label_provider(fit, error_label).replace(r"_", r"\_")


if __name__ == '__main__':
    sns.set_theme()
    sns.set_context("paper")
    set_latex_plot_params()
    explore_dists_root_folder = get_experiments_folder().joinpath("01_explore_data_distributions")
    start_date = DEFAULT_DATA_START_DATE
    end_date = "2019-12-31 23:00:00"

    start_date_path = start_date.split()[0].replace("-", "_")
    end_date_path = end_date.split()[0].replace("-", "_")
    explore_dists_folder = explore_dists_root_folder.joinpath(f"{start_date_path}_{end_date_path}")
    explore_dists_folder.mkdir(parents=True, exist_ok=True)
    importer = DWDWeatherDataImporter(start_date=start_date, end_date=end_date)
    importer.initialize()
    # extract all used targetColumns
    target_column_extra_info: dict[str, list[DistributionPlotColumn]] = {
        WeatherDataColumns.T_AIR_DEGREE_CELSIUS: [DistributionPlotColumn(
            plot_options=PlotOptions(x_label=r"Temperatur in $^{\circ}C$"),
            extra_dist_plots=["norm", "gennorm"],
            legend_spacing=True
        )],
        WeatherDataColumns.DH_W_PER_M2: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=r"Stundensumme der diffusen Solarstrahlung in $\frac{W}{m^2}$"),
                extra_dist_plots=[],
            ),
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=r"Stundensumme der diffusen Solarstrahlung in $\frac{W}{m^2}$ (lichter Tag)."),
                extra_dist_plots=["weibull_min", "beta"],
                transformer=lambda x: exclude_night_time_values(x),
            ),
        ],
        WeatherDataColumns.GH_W_PER_M2: [
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=r"Stundensumme der Globalstrahlung in $\frac{W}{m^2}$"),
                extra_dist_plots=[],
            ),
            DistributionPlotColumn(
                plot_options=PlotOptions(x_label=r"Stundensumme der Globalstrahlung in $\frac{W}{m^2}$ (lichter Tag)."),
                transformer=lambda x: exclude_night_time_values(x),
                extra_dist_plots=["weibull_min", "beta"],
            ),
        ],
        WeatherDataColumns.WIND_V_M_PER_S: [DistributionPlotColumn(
            plot_options=PlotOptions(x_label=r"Windgeschwindigkeit in $\frac{m}{s}$"),
            extra_dist_plots=["rayleigh"],
        )],
        WeatherDataColumns.WIND_DIR_DEGREE: [DistributionPlotColumn(
            plot_options=PlotOptions(x_label=r"Windrichtung in Grad $(0^{\circ} - 359^{\circ})$"),
            extra_dist_plots=["uniform", "random"],
            bins=36,  # Since 360 degree and only in steps*10
            legend_spacing=True
        )],
        WeatherDataColumns.CLOUD_PERCENT: [DistributionPlotColumn(
            plot_options=PlotOptions(x_label=r"Prozentuale Wolkenbedeckung in \%"),
            extra_dist_plots=[]
        )],
        WeatherDataColumns.SUN_HOURS_MIN_PER_H: [DistributionPlotColumn(
            plot_options=PlotOptions(x_label=r"Sonnenstunden $\frac{min}{h}$"),
            extra_dist_plots=[]
        )],
    }
    used_target_columns = [col_map.targetColumn for nested in WEATHER_DATA_MAPPING.values() for col_map in nested.columns]

    data_info = [(column, target_column_extra_info[column]) for column in used_target_columns if target_column_extra_info[column] is not None]

    for target_column, column_plot_metadata_entries in data_info:
        data = importer.data[target_column]
        for idx, column_plot_metadata in enumerate(column_plot_metadata_entries):
            if column_plot_metadata.transformer is not None:
                data = column_plot_metadata.transformer(data)
            margin=0.5
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4+margin, 4.8+margin))
            fit_res = draw_best_fit_plot(
                data=data,
                plot_metadata=column_plot_metadata,
                error_fn=("$R^2$", r_squared),
                best_score_finder=max,
                distribution_legend_label_provider_fn=latex_distribution_legend_label_provider,
                translations=latex_de_translations,
                distribution_names=distribution_names,
                plot=PlotResult(fig, ax)
            )
            # save file
            path = explore_dists_folder.joinpath(f"{target_column}_{idx}.pdf").absolute()
            fit_res.plot_res.fig.show()
            fit_res.plot_res.fig.savefig(path, bbox_inches='tight')
