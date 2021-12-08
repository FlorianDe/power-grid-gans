from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable

import numpy.typing as npt
from pandas import Series
import matplotlib.pyplot as plt

from data.fit.distribution_fit import residual_sum_of_squares, test_fit_against_all_distributions, create_pdf_series_from_distribution, \
    DistributionFit
from plots.typing import PlotResult, PlotOptions, Locale


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
    bins: int = 200
    plot_options: PlotOptions = PlotOptions()
    extra_dist_plots: Optional[list[str]] = None
    legend_spacing: bool = False


@dataclass(frozen=True, eq=True)
class DistributionFitPlotResult:
    fits: list[DistributionFit]
    best_fit: DistributionFit
    plot_res: PlotResult


def __get_parameter_label(fit: DistributionFit):
    param_names = (fit.distribution.shapes + ', loc, scale').split(', ') if fit.distribution.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, fit.params.raw)])
    dist_str = '{}({})'.format(fit.distribution_name, param_str)
    return dist_str


def draw_best_fit_plot(
        data: Series,
        plot_metadata: DistributionPlotColumn,
        error_fn: Callable[[npt.ArrayLike, npt.ArrayLike], float] = residual_sum_of_squares,
) -> DistributionFitPlotResult:
    def translate(key: __Keys) -> str:
        return __PLOT_DICT[key][plot_metadata.plot_options.locale]

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
    fitted_distributions = test_fit_against_all_distributions(data, plot_metadata.bins, error_fn)
    best_fit = min(fitted_distributions)

    # Make PDF with best params
    best_pdf = create_pdf_series_from_distribution(best_fit.distribution, best_fit.params.raw)
    best_pdf_label = __get_parameter_label(best_fit)
    best_pdf.plot(lw=2, label=best_pdf_label, legend=True, ax=ax)

    legends_plotted = 2
    extra_plots: Optional[list[str]] = plot_metadata.extra_dist_plots
    if extra_plots is not None:
        for dist_name in extra_plots:
            dist = next((x for x in fitted_distributions if x.distribution_name == dist_name and x.distribution_name != best_fit.distribution_name), None)
            if dist is not None:
                other_pdf_label = __get_parameter_label(dist)
                other_pdf = create_pdf_series_from_distribution(dist.distribution, dist.params.raw)
                other_pdf.plot(lw=2, label=other_pdf_label, legend=True, ax=ax)
                legends_plotted += 1

    # Set plot limits and params, if legends spacing enabled create a 10% spacing for each legend entry
    ax.set_ylim(data_ylim[0], data_ylim[1] if plot_metadata.legend_spacing is False else data_ylim[1] * (1 + legends_plotted / 10))
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
