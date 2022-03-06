from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Type
import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np

"""
Base code from: https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py
"""
Language = Literal["en", "de"]


class SummaryColumn(Enum):
    LAYER_IDX = "layer_idx"
    LAYER_TYPE = "layer_type"
    LAYER_PARAMETERS = "layer_parameters"
    INPUT_SHAPE = "input_shape"
    OUTPUT_SHAPE = "output_shape"
    NUMBER_PARAMS = "nb_params"


__PLOT_DICT: dict[SummaryColumn, dict[Language, str]] = {
    SummaryColumn.LAYER_IDX: {"en": "Layer", "de": "Layer"},
    SummaryColumn.LAYER_TYPE: {"en": "Type", "de": "Typ"},
    SummaryColumn.LAYER_PARAMETERS: {"en": "Options", "de": "Optionen"},
    SummaryColumn.INPUT_SHAPE: {"en": "In", "de": "Eingabe"},
    SummaryColumn.OUTPUT_SHAPE: {"en": "Out", "de": "Ausgabe"},
    SummaryColumn.NUMBER_PARAMS: {"en": "Parameters", "de": "Parameter"},
}


def translate(key: SummaryColumn, lang: Language = "de") -> str:
    return __PLOT_DICT[key][lang]


@dataclass
class LatexTableOptions:
    label: Optional[str] = None
    caption: Optional[str] = None
    positioning: Optional[str] = "htb"
    style: Optional[any] = None  # TODO TBD


@dataclass
class LatexColumn:
    key: str
    h_align: Literal["l", "c", "r"] = "l"
    style: Optional[any] = None  # TODO TBD


@dataclass
class ParameterDescription:
    key: str
    label: str
    value: any

    def __str__(self):
        return self.label + " = " + str(self.value)


class ParameterExtractor(ABC):
    def __init__(self, module_type: Type[nn.Module]):
        self.module_type = module_type

    def is_type(self, module: nn.Module) -> bool:
        return isinstance(module, self.module_type)

    @staticmethod
    def createExtractors() -> dict[type[nn.Module], ParameterExtractor]:
        extractors: dict[type[nn.Module], ParameterExtractor] = {}
        for extractor_constructor in ParameterExtractor.__subclasses__():
            extractor = extractor_constructor()
            if extractor.module_type in extractors:
                raise ValueError(
                    f"You cannot create multiple extractors for the same nn.Module type. Found {extractors[extractor.module_type]} for {extractor.module_type} cannot add {extractor_constructor}"
                )
            extractors[extractor.module_type] = extractor
        return extractors

    @abstractmethod
    def _extract(self, module: nn.Module) -> list[ParameterDescription]:
        return []

    def extract(self, module: nn.Module) -> list[ParameterDescription]:
        if self.is_type(module) is False:
            raise ValueError(
                f"Cannot extract anything, since the module is of type {type(module)} and this extractor can only deal with {type(self)}"
            )
        return self._extract(module)


class LinearParameterExtractor(ParameterExtractor):
    def __init__(self):
        ParameterExtractor.__init__(self, nn.Linear)

    def _extract(self, module: nn.Linear) -> list[ParameterDescription]:
        return [
            ParameterDescription(key, label, module.__dict__[key])
            for (key, label) in [("in_features", "in"), ("out_features", "out")]
        ]


class DropoutParameterExtractor(ParameterExtractor):
    def __init__(self):
        ParameterExtractor.__init__(self, nn.Dropout)

    def _extract(self, module: nn.Dropout) -> list[ParameterDescription]:
        return [
            ParameterDescription("p", "percentage", module.p),
        ]


class LeakyReLUParameterExtractor(ParameterExtractor):
    def __init__(self):
        ParameterExtractor.__init__(self, nn.LeakyReLU)

    def _extract(self, module: nn.LeakyReLU) -> list[ParameterDescription]:
        return [
            ParameterDescription("negative_slope", "slope", module.negative_slope),
        ]


class TanhParameterExtractor(ParameterExtractor):
    def __init__(self):
        ParameterExtractor.__init__(self, nn.Tanh)

    def _extract(self, module: nn.Tanh) -> list[ParameterDescription]:
        return []


class SigmoidParameterExtractor(ParameterExtractor):
    def __init__(self):
        ParameterExtractor.__init__(self, nn.Sigmoid)

    def _extract(self, module: nn.Sigmoid) -> list[ParameterDescription]:
        return []


if __name__ == "__main__":
    print(ParameterExtractor.createExtractors())
    print([x() for x in ParameterExtractor.__subclasses__()])


@dataclass
class TotalSummary:
    total_params: int
    trainable_params: int
    total_input_size: float
    total_output_size: float
    total_params_size: float
    total_size: float

    def __str__(self):
        summary_str = "================================================================" + "\n"
        summary_str += "Total params: {0:,}".format(self.total_params) + "\n"
        summary_str += "Trainable params: {0:,}".format(self.trainable_params) + "\n"
        summary_str += "Non-trainable params: {0:,}".format(self.total_params - self.trainable_params) + "\n"
        summary_str += "----------------------------------------------------------------" + "\n"
        summary_str += "Input size (MB): %0.2f" % self.total_input_size + "\n"
        summary_str += "Forward/backward pass size (MB): %0.2f" % self.total_output_size + "\n"
        summary_str += "Params size (MB): %0.2f" % self.total_params_size + "\n"
        summary_str += "Estimated Total Size (MB): %0.2f" % self.total_size + "\n"
        summary_str += "----------------------------------------------------------------" + "\n"
        return summary_str


def default_column_setup() -> list[LatexColumn]:
    return [
        LatexColumn(key=SummaryColumn.LAYER_IDX, h_align="c"),
        LatexColumn(key=SummaryColumn.LAYER_TYPE),
        LatexColumn(key=SummaryColumn.LAYER_PARAMETERS),
        LatexColumn(key=SummaryColumn.INPUT_SHAPE),
        LatexColumn(key=SummaryColumn.OUTPUT_SHAPE),
        LatexColumn(key=SummaryColumn.NUMBER_PARAMS),
    ]


@dataclass
class NetSummary:
    summary: OrderedDict
    total_summary: TotalSummary

    def __str__(self):
        summary_str = "----------------------------------------------------------------" + "\n"
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        summary_str += line_new + "\n"
        summary_str += "================================================================" + "\n"
        for layer in self.summary:
            line_new = "{:>20}  {:>25} {:>15}".format(
                layer,
                str(self.summary[layer][SummaryColumn.OUTPUT_SHAPE]),
                "{0:,}".format(self.summary[layer][SummaryColumn.NUMBER_PARAMS]),
            )
            summary_str += line_new + "\n"

        return f"{summary_str}\n{self.total_summary}"

    def remove_last_layer(self):
        self.summary.popitem()

    def to_latex_table(
        self,
        columns: list[LatexColumn] = default_column_setup(),
        lang: Language = "de",
        options: Optional[LatexTableOptions] = None,
    ) -> str:
        header_names = [translate(col.key, lang) for col in columns]

        table = ""
        table += r"\begin{table}["
        if options.positioning:
            table += options.positioning
        table += r"]" + "\n"
        column_definition = "|".join([col.h_align for col in columns])
        table += r"\begin{tabular}{" + column_definition + r"}" + "\n"
        table += r"\hline" + "\n"
        table += " & ".join(header_names) + r" \\" + "\n"
        table += r"\hline" + "\n"

        # content
        for layer in self.summary:
            line = ""
            line += str(self.summary[layer][SummaryColumn.LAYER_IDX]) + "&"
            line += str(self.summary[layer][SummaryColumn.LAYER_TYPE]) + "&"
            layer_options = self.summary[layer][SummaryColumn.LAYER_PARAMETERS]
            if layer_options and len(layer_options) > 0:
                line += ", ".join([str(option) for option in layer_options]) + "&"
            else:
                line += "-" + "&"
            line += str(self.summary[layer][SummaryColumn.INPUT_SHAPE]) + "&"
            line += str(self.summary[layer][SummaryColumn.OUTPUT_SHAPE]) + "&"
            line += "{0:,}".format(self.summary[layer][SummaryColumn.NUMBER_PARAMS])
            line += r" \\" + "\n"
            table += line

        table += r"\hline" + "\n"
        table += r"\end{tabular}" + "\n"

        # caption
        table += r"\caption{"
        if options.caption:
            table += options.caption
        table += r"}" + "\n"

        # label
        if options and options.label:
            table += r"\label{table:" + options.label + r"}" + "\n"

        table += r"\end{table}"

        return table


def create_summary(
    model,
    input_size,
    batch_size=-1,
    drop_last_layer: bool = True,
    parameter_extractors: dict[type[nn.Module], ParameterExtractor] = ParameterExtractor.createExtractors(),
    device=torch.device("cpu"),
    dtypes=None,
) -> NetSummary:
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key][SummaryColumn.LAYER_IDX] = module_idx
            summary[m_key][SummaryColumn.LAYER_TYPE] = class_name
            parameter_extractor = parameter_extractors.get(type(module))
            if parameter_extractor is None and type(model) != type(module):
                print(f"Haven't found a parameter extractor for {type(module)}")
            summary[m_key][SummaryColumn.LAYER_PARAMETERS] = (
                parameter_extractor.extract(module) if parameter_extractor else []
            )
            summary[m_key][SummaryColumn.INPUT_SHAPE] = list(input[0].size())
            summary[m_key][SummaryColumn.INPUT_SHAPE][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key][SummaryColumn.OUTPUT_SHAPE] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key][SummaryColumn.OUTPUT_SHAPE] = list(output.size())
                summary[m_key][SummaryColumn.OUTPUT_SHAPE][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key][SummaryColumn.NUMBER_PARAMS] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device) for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    with torch.no_grad():
        model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        total_params += summary[layer][SummaryColumn.NUMBER_PARAMS]
        total_output += np.prod(summary[layer][SummaryColumn.OUTPUT_SHAPE])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer][SummaryColumn.NUMBER_PARAMS]

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4.0 / (1024**2.0))
    total_output_size = abs(2.0 * total_output * 4.0 / (1024**2.0))  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024**2.0))
    total_size = total_params_size + total_output_size + total_input_size

    total_summary = TotalSummary(
        total_params=total_params,
        trainable_params=trainable_params,
        total_input_size=total_input_size,
        total_output_size=total_output_size,
        total_params_size=total_params_size,
        total_size=total_size,
    )
    net_summary = NetSummary(summary=summary, total_summary=total_summary)
    if drop_last_layer:
        net_summary.remove_last_layer()
    return net_summary
