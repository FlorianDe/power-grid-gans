from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable

import matplotlib.pyplot as plt
import numpy.typing as npt
from pandas import Series

from src.data.distribution.distribution_fit import test_fit_against_all_distributions, create_pdf_series_from_distribution, DistributionFit
from src.metrics.r_squared import r_squared
from src.plots.typing import PlotResult, PlotOptions, Locale


class __Keys(Enum):
    LEGEND_DATA_DEFAULT = auto(),
    BINNING = auto(),
    Y_LABEL_DEFAULT = auto(),
    X_LABEL_DEFAULT = auto(),


__PLOT_DICT: dict[__Keys, dict[Locale, str]] = {
    __Keys.LEGEND_DATA_DEFAULT: {
        Locale.EN: "Data split into {X} bins",
        Locale.DE: "Daten aufgeteilt in {X} Klassen"
    },
    __Keys.Y_LABEL_DEFAULT: {
        Locale.EN: "Relative frequencies P(x)",
        Locale.DE: "Relative Häufigkeitsdichte P(x)"
    },
    __Keys.X_LABEL_DEFAULT: {
        Locale.EN: "Value",
        Locale.DE: "Wert"
    },
}


@dataclass(frozen=True, eq=True)
class DistributionPlotColumn:
    bins: int = 100
    plot_options: PlotOptions = PlotOptions()
    extra_dist_plots: Optional[list[str]] = None
    legend_spacing: bool = False
    transformer: Optional[Callable[[Series], Series]] = None
    drop_na_values: bool = True


@dataclass(frozen=True, eq=True)
class DistributionFitPlotResult:
    fits: list[DistributionFit]
    best_fit: DistributionFit
    plot_res: PlotResult


def default_distribution_legend_label_provider(fit: DistributionFit, error_label: str) -> str:
    param_names = (fit.distribution.shapes + ', loc, scale').split(', ') if fit.distribution.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, fit.params.raw)])
    dist_str = '{}({}), {}: {:.3f}'.format(fit.distribution_name, param_str, error_label, fit.score)
    return dist_str


def draw_best_fit_plot(
        data: Series,
        plot_metadata: DistributionPlotColumn,
        error_fn: tuple[str, Callable[[npt.ArrayLike, npt.ArrayLike], float]] = ("R²", r_squared),
        best_score_finder: Callable[[list[DistributionFit]], DistributionFit] = max,
        distribution_legend_label_provider_fn: Callable[[DistributionFit, str], str] = default_distribution_legend_label_provider,
        translations: Optional[dict[__Keys, str]] = None,
        distribution_names: Optional[list[str]] = None
) -> DistributionFitPlotResult:
    def translate(key: __Keys) -> str:
        return translations[key] if translations is not None else __PLOT_DICT[key][plot_metadata.plot_options.locale]

    error_label, error = error_fn

    # Drop na values if option set to true
    if plot_metadata.drop_na_values is True:
        data = data.dropna()

    # Create subfig
    fig, ax = plt.subplots(nrows=1, ncols=1)

    print(f"{plot_metadata=}")
    # Plot data pdf
    data_legend_label = translate(__Keys.LEGEND_DATA_DEFAULT).replace("{X}", str(plot_metadata.bins))
    data.plot(kind='hist', bins=plot_metadata.bins, density=True, alpha=0.5, label=data_legend_label, legend=True, ax=ax)

    # Save plot limits
    data_ylim = ax.get_ylim()
    data_xlim = ax.get_xlim()

    # Find best fit distribution
    fitted_distributions = test_fit_against_all_distributions(
        data=data,
        bins=plot_metadata.bins,
        error_fn=error,
        distribution_names=distribution_names
    )
    best_fit = best_score_finder(fitted_distributions)

    # Make PDF with best params
    best_pdf = create_pdf_series_from_distribution(best_fit.distribution, best_fit.params.raw)
    best_pdf_label = distribution_legend_label_provider_fn(best_fit, error_label)
    best_pdf.plot(lw=2, label=best_pdf_label, legend=True, ax=ax)

    legends_plotted = 2
    extra_plots: Optional[list[str]] = plot_metadata.extra_dist_plots
    if extra_plots is not None:
        for dist_name in extra_plots:
            dist = next((x for x in fitted_distributions if x.distribution_name == dist_name and x.distribution_name != best_fit.distribution_name), None)
            if dist is not None:
                other_pdf_label = distribution_legend_label_provider_fn(dist, error_label)
                other_pdf = create_pdf_series_from_distribution(dist.distribution, dist.params.raw)
                other_pdf.plot(lw=2, label=other_pdf_label, legend=True, ax=ax)
                legends_plotted += 1

    # Set plot limits and params, if legends spacing enabled create a 5% spacing for each legend entry
    ax.set_ylim(data_ylim[0], data_ylim[1] if plot_metadata.legend_spacing is False else data_ylim[1] * (1 + (5 * legends_plotted) / 100))
    ax.set_xlim(data_xlim)

    if plot_metadata.plot_options.title is not None:
        ax.set_title(plot_metadata.plot_options.title)
    ax.set_xlabel(translate(__Keys.X_LABEL_DEFAULT) if plot_metadata.plot_options.x_label is None else plot_metadata.plot_options.x_label)
    ax.set_ylabel(translate(__Keys.Y_LABEL_DEFAULT) if plot_metadata.plot_options.y_label is None else plot_metadata.plot_options.y_label)
    ax.legend(loc="upper right")

    return DistributionFitPlotResult(
        fits=fitted_distributions,
        best_fit=best_fit,
        plot_res=PlotResult(fig, ax)
    )
