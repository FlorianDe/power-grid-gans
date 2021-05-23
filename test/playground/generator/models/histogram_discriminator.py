import torch
import torch.nn as nn
import torch.nn.functional as F


class PrintLayer(nn.Module):
    def __init__(self, name: str):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name} {x.shape}: {x}')
        return x


class HistogramDiscriminator(nn.Module):
    def __init__(self, D_in: int, H: int, D_out: int):
        super(HistogramDiscriminator, self).__init__()
        self.model = nn.Sequential(
            PrintLayer('0. input'),
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            PrintLayer('1. conv1d'),
            nn.ReLU(),
            PrintLayer('1. Relu'),
            # nn.BatchNorm1d(H),
            # PrintLayer('1. BatchNorm'),
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            PrintLayer('2. conv1d'),
            nn.Sigmoid(),  # replace via a softmax for labels!
            PrintLayer('sigmoid'),
        )
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Discriminator Parameter Count: {pytorch_total_params}')

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    torch.manual_seed(1)

    BATCH_SIZE = 2
    D_in = 24  # input size of G
    H = 10  # number of  hidden neurons
    D_out = 6  # output size of G

    fake_data = torch.randn(D_in).unsqueeze(0).unsqueeze(0)
    netD = HistogramDiscriminator(
        D_in=D_in,
        H=H,
        D_out=D_out
    )
    pred = netD(fake_data)
