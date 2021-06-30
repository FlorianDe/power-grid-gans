import math
from dataclasses import dataclass
import torch.nn as nn

from test import LinearNet
from test import CustomModule


@dataclass
class ConvData:
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    pooling: int = 2


class ConvolutionalNet(CustomModule):
    def __init__(self, img_size: tuple[int, int], conv_layers_data: list[ConvData] = None, linear_hidden_layers: list[int] = None):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        if conv_layers_data is None:
            conv_layers_data = [ConvData(10, 5), ConvData(20, 5)]  # default convolutional layers
        self.conv_layers = nn.ModuleList()
        if len(img_size) != 2:
            raise ValueError("The passed img size tuple had more or less than 2 values.")
        h_out = img_size[0]
        w_out = img_size[1]
        last_out_channels = 1
        for i in range(len(conv_layers_data)):
            c = conv_layers_data[i]
            conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=last_out_channels,
                    out_channels=c.out_channels,
                    kernel_size=c.kernel_size,
                    stride=c.stride,
                    padding=c.padding,
                    dilation=c.dilation
                ),
                nn.ReLU(),
                nn.MaxPool2d(c.pooling),
                nn.ReLU()
            )
            self.conv_layers.append(conv)
            h_out = ConvolutionalNet.calc_conv_out(h_out, c)
            w_out = ConvolutionalNet.calc_conv_out(w_out, c)
            last_out_channels = c.out_channels

        self.flatten = nn.Flatten()
        self.linear = LinearNet(h_out*w_out*last_out_channels, linear_hidden_layers)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

    @staticmethod
    def calc_conv_out(size_in, c: ConvData):
        size_out = math.floor(1+(size_in + 2 * c.padding - c.dilation*(c.kernel_size - 1) - 1)/c.stride)
        size_out = math.floor(size_out / c.pooling)
        return size_out


if __name__ == "__main__":
    ConvolutionalNet((28, 28), [], [50])
