import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(self, input_size: int, hidden_layers: list[int] = None):
        super(LinearNet, self).__init__()
        self.input_vector_size = input_size
        if hidden_layers is None:
            hidden_layers = []
        hidden_layers.insert(0, self.input_vector_size)
        hidden_layers.append(10)
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
