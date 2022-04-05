from dataclasses import dataclass
from enum import Enum

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import DecomposeResult

from src.plots.typing import PlotResult, PlotOptions, Locale


class DecomposeResultColumns(Enum):
    OBSERVED = "observed"
    SEASONAL = "seasonal"
    TREND = "trend"
    RESID = "resid"
    WEIGHTS = "weights"


__PLOT_DICT: dict[DecomposeResultColumns, dict[Locale, str]] = {
    DecomposeResultColumns.OBSERVED: {Locale.EN: "Observed", Locale.DE: "Beobachtung"},
    DecomposeResultColumns.SEASONAL: {Locale.EN: "Seasonal", Locale.DE: "Saisonal"},
    DecomposeResultColumns.TREND: {Locale.EN: "Trend", Locale.DE: "Trend"},
    DecomposeResultColumns.RESID: {Locale.EN: "Residuals", Locale.DE: "Rest"},
    DecomposeResultColumns.WEIGHTS: {Locale.EN: "Weights", Locale.DE: "Gewichte"},
}

GERMAN_LATEX_TRANSLATIONS = {
    DecomposeResultColumns.OBSERVED: r"$\displaystyle{\text{Daten}\;Y_t}$",
    DecomposeResultColumns.SEASONAL: r"$\displaystyle{\text{Saisonal}\;S_t}$",
    DecomposeResultColumns.TREND: r"$\displaystyle{\text{Trend}\;T_t}$",
    DecomposeResultColumns.RESID: r"$\displaystyle{\text{Rest}\;R_t}$",
    DecomposeResultColumns.WEIGHTS: r"$\displaystyle{\text{Gewichte}\;W_t}$",
}


@dataclass
class DecomposePlotOptions:
    """
    Plot options for the decomposition results

    Parameters
    ----------
    observed : bool
        Include the observed series in the plot
    seasonal : bool
        Include the seasonal component in the plot
    trend : bool
        Include the trend component in the plot
    resid : bool
        Include the residual in the plot
    weights : bool
        Include the weights in the plot (if any)
    """

    observed: bool = True
    seasonal: bool = True
    trend: bool = True
    resid: bool = True
    weights: bool = False


def draw_timeseries_decomposition_plot(
    data: DecomposeResult,
    plot_options: PlotOptions = PlotOptions(),
    translations: dict[DecomposeResultColumns, str] = None,
    decompose_plot_options: DecomposePlotOptions = DecomposePlotOptions(),
    rasterized: bool = False,
    figsize: tuple[float, float] = (6.4, 4.8),
) -> PlotResult:
    def translate(key: DecomposeResultColumns) -> str:
        return translations[key] if translations is not None else __PLOT_DICT[key][plot_options.locale]

    register_matplotlib_converters()
    series = [(data.observed, DecomposeResultColumns.OBSERVED)] if decompose_plot_options.observed else []
    series += [(data.trend, DecomposeResultColumns.TREND)] if decompose_plot_options.trend else []
    series += [(data.seasonal, DecomposeResultColumns.SEASONAL)] if decompose_plot_options.seasonal else []
    series += [(data.resid, DecomposeResultColumns.RESID)] if decompose_plot_options.resid else []
    series += [(data.weights, DecomposeResultColumns.WEIGHTS)] if decompose_plot_options.weights else []

    if isinstance(data.observed, (pd.DataFrame, pd.Series)):
        nobs = data.observed.shape[0]
        xlim = data.observed.index[0], data.observed.index[nobs - 1]
    else:
        xlim = (0, data.observed.shape[0] - 1)

    fig, axs = plt.subplots(nrows=len(series), ncols=1, figsize=figsize)
    for i, (ax, (series, column)) in enumerate(zip(axs, series)):
        # if rasterized:
        # ax.set_axisbelow(True)
        ax.grid(True, axis="both", linestyle="-")
        # ax.set_rasterization_zorder(0)

        if column != DecomposeResultColumns.RESID:
            ax.plot(series, zorder=10, rasterized=rasterized)
        else:
            ax.plot(
                series,
                marker="o",
                markersize=0.5 * plt.rcParams["lines.markersize"],
                linestyle="none",
                zorder=10,
                rasterized=rasterized,
            )
            # ax.plot(series, zorder=10, rasterized=rasterized)
            ax.plot(xlim, (0, 0), color="#000000", zorder=10, rasterized=rasterized)
        name = translate(column)
        if i == 0 and decompose_plot_options.observed:
            ax.set_title(name)
        else:
            ax.set_ylabel(name)
        ax.set_xlim(xlim)
        ax.tick_params(axis="x", labelrotation=30)

    fig.align_ylabels(axs)
    fig.tight_layout()

    return PlotResult(fig, axs)
