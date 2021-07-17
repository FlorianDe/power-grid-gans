import torch
import torch.nn as nn
from src.net.custom_module import CustomModule


class RecurrentNet(CustomModule):
    def __init__(self, sequence_length: int, input_size: int, num_layers: int, hidden_size: int, out_size: int):
        super(RecurrentNet, self).__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        """
        :param x: Shape: [batch, seq, feature].
        :return: Shape: [10] <- Digit Classes
        """
        print(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)  # Shape: [batch_size, seq_length, hidden_size]
        out = out[:, -1, :]  # Reshape to [BATCH, hidden_size] from last SEQ
        out = self.linear(out)
        return out

    def reshape(self, x: torch.Tensor):
        return x.reshape(-1, self.sequence_length, self.input_size)
