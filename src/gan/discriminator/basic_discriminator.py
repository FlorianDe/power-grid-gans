from typing import Optional

import torch.nn as nn
from torch import Tensor

from src.net.dynamic import FNN


class BasicDiscriminator(FNN):
    def __init__(self, input_size: int, out_size: int, hidden_layers: Optional[list[int]] = None):
        super(BasicDiscriminator, self).__init__(input_size, out_size, hidden_layers)
        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor):
        x = super().forward(x)
        x = self.activation(x)
        x = x.view(-1)
        return x

    def reshape(self, x):
        return x
