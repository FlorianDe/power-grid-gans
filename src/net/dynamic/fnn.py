from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.net.custom_module import CustomModule


class FNN(CustomModule):
    def __init__(self, input_size: int, out_size: int, hidden_layers: Optional[list[int]] = None):
        super(FNN, self).__init__()
        self.input_vector_size = input_size
        if hidden_layers is None:
            hidden_layers = []
        hidden_layers.insert(0, self.input_vector_size)
        hidden_layers.append(out_size)
        self.linear_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.linear_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

    def forward(self, x: Tensor):
        for index, layer in enumerate(self.linear_layers):
            x = layer(x)
            if index < len(self.linear_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x)
        return x
