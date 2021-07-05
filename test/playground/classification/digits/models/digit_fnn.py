from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from src.net.custom_module  import CustomModule


class DigitLinearNet(CustomModule):
    def __init__(self, input_size: int, out_size: int, hidden_layers: Optional[list[int]] = None):
        super(DigitLinearNet, self).__init__()
        self.input_vector_size = input_size
        if hidden_layers is None:
            hidden_layers = []
        hidden_layers.insert(0, self.input_vector_size)
        hidden_layers.append(out_size)
        self.linear_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.linear_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))

    def forward(self, img):
        x = img.view(-1, self.input_vector_size)
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            if i < len(self.linear_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x)
        return x
