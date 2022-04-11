import torch.nn as nn
from torch import Size, TensorType

from src.net.summary.net_summary import LatexTableOptions, create_summary


def print_net_summary(
    G: nn.Module,
    D: nn.Module,
    discriminator_input_size: Size,
    generator_input_size: Size,
    latex_options: LatexTableOptions,
    dtypes: list[TensorType] = None,
):
    placeholder = "{net_type}"
    d_latex_options = LatexTableOptions(
        label=latex_options.label.replace(placeholder, "discriminator"),
        caption=latex_options.caption.replace(placeholder, "Diskriminator"),
        positioning=latex_options.positioning,
        style=latex_options.style,
    )
    g_latex_options = LatexTableOptions(
        label=latex_options.label.replace(placeholder, "generator"),
        caption=latex_options.caption.replace(placeholder, "Generator"),
        positioning=latex_options.positioning,
        style=latex_options.style,
    )
    row_len = 90
    spacer = "%"
    spacer_line = spacer * row_len
    padding_space = row_len - len(latex_options.label) - 2
    left_pad = padding_space // 2 + (1 if padding_space % 2 == 1 else 0)
    right_pad = padding_space // 2
    print(spacer_line)
    print(f"{spacer*left_pad} {latex_options.label} {spacer*right_pad}")
    print(spacer_line)

    g_summary = create_summary(model=G, input_size=generator_input_size, dtypes=dtypes)
    d_summary = create_summary(model=D, input_size=discriminator_input_size, dtypes=dtypes)
    print(g_summary)
    print(d_summary)
    print("")
    print(g_summary.to_latex_table(options=g_latex_options))
    print("")
    print(d_summary.to_latex_table(options=d_latex_options))
    print(spacer_line)
