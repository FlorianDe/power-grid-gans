import torch
import torch.nn as nn


class RNNHistogramDiscriminator(nn.Module):
    def __init__(self, input_size: int, num_layers: int, hidden_size: int):
        super(RNNHistogramDiscriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f'Generator Parameter Count: {pytorch_total_params}')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        x, hn = self.rnn(x, h0.detach())
        x = x[:, -1, :]
        x = self.linear(x)
        x = self.sigmoid(x)
        x = x.squeeze(dim=1)
        return x


if __name__ == "__main__":
    # torch.manual_seed(1)

    BATCH_SIZE = 15
    D_in = 144  # input size of G
    hidden = 10  # number of  hidden neurons

    fake_data = torch.randn(BATCH_SIZE, 1, D_in)
    fake_data_label = torch.zeros(BATCH_SIZE)
    netD = RNNHistogramDiscriminator(D_in, 5, hidden)
    objective = nn.BCELoss()
    fake_data_pred = netD(fake_data)
    print(f'fake_data_pred {fake_data_pred.shape}: {fake_data_pred}')
    loss = objective(fake_data_pred, fake_data_label)


