from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.text
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn

from utils import get_generated_images_path_folder
from experiments.experiments_utils.utils import set_latex_plot_params

from src.plots.typing import PlotOptions, PlotResult


def wrap_with_latex_makro(value):
    str_value = str(value)
    return str_value if str_value.startswith("$") and str_value.endswith("$") else f"${str_value}$"


@dataclass(frozen=True, eq=True)
class Tick:
    position: float
    label: str


@dataclass(frozen=True, eq=True)
class AddTicks:
    y_ticks: Optional[list[Tick]] = None
    x_ticks: Optional[list[Tick]] = None


@dataclass(frozen=True, eq=True)
class ActivationConfig:
    fn: Callable[[Tensor], Tensor]
    eq: str
    plot_options: Optional[PlotOptions] = PlotOptions(y_label=r"$\sigma(x)$")
    add_ticks: Optional[AddTicks] = None


def plot_activation_function(x_start: int, x_end: int, config: ActivationConfig) -> PlotResult:
    fn = config.fn
    plot_options = config.plot_options
    equation = config.eq

    x = torch.linspace(x_start, x_end, 500)
    y = fn(x)

    figsize = (9, 4)
    label_fontsize = 20
    equation_fontsize = 18
    tick_labels_fontsize = 16
    tick_label_non_numeric_fontsize = 18
    linewidth_function = 2.5

    x_label = "$x$"
    y_label = r"$\sigma(x)$"

    set_latex_plot_params()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    # plt.rcParams['grid.color']
    ax.axhline(y=0, color="k", linewidth=plt.rcParams["grid.linewidth"])
    ax.axvline(x=0, color="k", linewidth=plt.rcParams["grid.linewidth"])
    ax.plot(x, y, linewidth=linewidth_function)
    # ax.set_aspect('equal')
    ax.grid(True, axis="y", linestyle="dashed")
    # ax.grid(True, axis='x')
    # ax.grid(True, which='both')
    sns.despine(ax=ax, offset=0)

    x_default_ticks = list(range(x_min, x_max + 1))
    x_ticks = x_default_ticks.copy()
    x_labels = x_default_ticks
    if config.add_ticks is not None and config.add_ticks.x_ticks is not None:
        for add_tick in config.add_ticks.x_ticks:
            try:
                idx = x_ticks.index(add_tick.position)
                x_labels[idx] = add_tick.label
            except:
                raise ValueError("Shouldn't happen rn")
                pass
        # Take second and pre last label and set them to "..."
        x_labels[-2] = r"$\dots$"
        x_labels[1] = r"$\dots$"

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([wrap_with_latex_makro(x_label) for x_label in x_labels], fontsize=tick_labels_fontsize)

    max_y = max(y)
    y_default_ticks = [-1, 0, 1]
    y_tick_values = [*y_default_ticks, max_y] if abs(1 - max_y) > 0.05 else [-1, 0, 1]
    y_labels = (
        y_tick_values.copy()[:-1] + [r"$\infty$"] if len(y_tick_values) > len(y_default_ticks) else y_default_ticks
    )
    if config.add_ticks is not None and config.add_ticks.y_ticks is not None:
        for add_tick in config.add_ticks.y_ticks:
            try:
                idx = x_ticks.index(add_tick.position)
                raise ValueError("Shouldn't happen rn")
            except:
                # TODO only adding negativ inf for now!
                if y_tick_values[0] > add_tick.position:
                    y_tick_values.insert(0, add_tick.position)
                    y_labels.insert(0, add_tick.label)

    ax.set_yticks(y_tick_values)
    ax.set_yticklabels([wrap_with_latex_makro(y_label) for y_label in y_labels], fontsize=tick_labels_fontsize)

    # Set the tick labels font
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if isinstance(label, matplotlib.text.Text):
            if "infty" in label.get_text():
                label.set_fontsize(f"{tick_label_non_numeric_fontsize}")

    # calculate y axis margins
    y_lim = ax.get_ylim()
    # y_margins = abs(y_lim[1]-y_lim[0])*(0.05) #dynamic calculation
    y_margins = 0.2
    ax.set_ylim(y_lim[0] - y_margins, y_lim[1] + y_margins)

    # ax.tick_params( axis='x', which='minor', direction='out', length=30 )
    # ax.tick_params( axis='x', which='major', bottom='off', top='off' )
    if plot_options.title is not None:
        ax.set_title(plot_options.title)
    ax.set_xlabel(plot_options.x_label if plot_options.x_label is not None else x_label, fontsize=label_fontsize)
    ax.set_ylabel(plot_options.y_label if plot_options.y_label is not None else y_label, fontsize=label_fontsize)
    # ax.text(0.18, 0.18, r'$\displaystyle\sum_{i=0}^\infty x_i$', color="C0", fontsize=16)
    # ax.text(0.18, 0.18, equation, color="C0", fontsize=16)
    ax.annotate(
        equation,
        xy=(0, 0),
        xytext=(0.25, 0.7),
        textcoords="axes fraction",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=equation_fontsize,
    )
    # ax.margins(-0.1, 0.1, tight=False)

    # ax.spines['left'].set_position(('axes', 0.5))
    # ax.legend(loc="best")
    # ax.set_ylim([-1.2, 1.2])
    # fig.gca().set_axis_off()

    # ax.margins(0,0)
    # ax.gca().xaxis.set_major_locator(plt.NullLocator())
    # ax.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.savefig("filename.pdf", bbox_inches = 'tight', pad_inches = 0)
    return PlotResult(fig, ax)


if __name__ == "__main__":
    # sns.set_theme()
    # sns.set(style='ticks')
    # sns.axes_style("darkgrid")
    # sns.set_context("paper")

    generated_images = get_generated_images_path_folder()

    activation_functions = generated_images.joinpath(f"activation_functions")
    activation_functions.mkdir(parents=True, exist_ok=True)

    x_min = -5
    x_max = 5

    tex_pos_inf = r"$\infty$"
    tex_neg_inf = r"$-\infty$"

    default_additional_x_ticks = [Tick(x_min, tex_neg_inf), Tick(x_max, tex_pos_inf)]

    def binary_step(x: Tensor):
        return torch.tensor([0 if e < 0 else 1 for e in x])

    activation_function_config: dict[str, ActivationConfig] = {
        "BinaryStep": ActivationConfig(
            fn=binary_step,
            eq=r"$\displaystyle\sigma(x)=\begin{cases}1, &x \geq 0\\0, &x<0\end{cases}$",
            add_ticks=AddTicks(x_ticks=default_additional_x_ticks),
        ),
        "Sigmoid": ActivationConfig(
            fn=torch.sigmoid,
            eq=r"$\displaystyle\sigma(x)=\frac{1}{1+e^{-x}}$",
            add_ticks=AddTicks(x_ticks=default_additional_x_ticks),
        ),
        "TanH": ActivationConfig(
            fn=torch.tanh,
            eq=r"$\displaystyle\sigma(x)=tanh(x)=\frac{2}{1+e^{-2x}}-1$",
            add_ticks=AddTicks(x_ticks=default_additional_x_ticks),
        ),
        "PReLU": ActivationConfig(
            fn=nn.LeakyReLU(1 / 3),
            eq=r"$\displaystyle\sigma_{\alpha}(x)=\begin{cases}x, &x \geq 0\\ \alpha \cdot x, &x>0\end{cases}$",
            plot_options=PlotOptions(y_label=r"$\sigma_{(\frac{1}{3})}(x)$"),
            add_ticks=AddTicks(
                x_ticks=default_additional_x_ticks,
                y_ticks=[
                    Tick(x_min * (1 / 3), tex_neg_inf),
                ],
            ),
        ),
        "ReLU": ActivationConfig(
            fn=torch.relu,
            eq=r"$\displaystyle\sigma(x)=\begin{cases}x, &x \geq 0\\0, &x<0\end{cases}$",
            add_ticks=AddTicks(x_ticks=default_additional_x_ticks),
        ),
        "ELU": ActivationConfig(
            fn=nn.ELU(1.0),
            eq=r"$\displaystyle\sigma_{\alpha}(x)=\begin{cases}x, &x \geq 0\\ \alpha(e^{x}-1), &x < 0\end{cases}$",
            plot_options=PlotOptions(y_label=r"$\sigma_{(1)}(x)$"),
            add_ticks=AddTicks(x_ticks=default_additional_x_ticks),
        ),
    }

    for key, config in activation_function_config.items():
        fig, ax = plot_activation_function(x_min, x_max, config)
        fig.show()
        fig.savefig(activation_functions / f"{key}.pdf", bbox_inches="tight", pad_inches=0)
