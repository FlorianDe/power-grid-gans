import torch
import torch.nn as nn
from torch import Tensor

from src.net import CustomModule


class CNNDiscriminator(CustomModule):
    def __init__(self, input_size: int, out_size: int):
        super(CNNDiscriminator, self).__init__()
        filters = 60
        self.input_size = input_size
        self.last_filter = 20
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=filters,
            kernel_size=(1,),
            dilation=(1,),
            padding="same"
        )
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=(8,), dilation=(2,), padding="same")
        self.bn2 = nn.BatchNorm1d(filters)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv1d(filters, filters, kernel_size=(5,), dilation=(2,),  padding="same")
        self.bn3 = nn.BatchNorm1d(filters)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv1d(filters, filters, kernel_size=(5,), dilation=(2,),  padding="same")
        self.bn4 = nn.BatchNorm1d(filters)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv1d(filters, self.last_filter, kernel_size=(5,), dilation=(2,),  padding="same")
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self.input_size*self.last_filter, out_size)
        self.activation = nn.Sigmoid()
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f'CNNDiscriminator Parameter Count: {pytorch_total_params}')

    def forward(self, x: Tensor):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        x = x.view(-1, self.input_size*self.last_filter)

        x = self.dense(x)
        x = self.activation(x)
        x = x.view(-1)
        return x

    def reshape(self, x):
        return x


if __name__ == '__main__':
    # torch.manual_seed(1)

    BATCH_SIZE = 15
    FILTERS = 64
    D_in = 1024  # input size of G
    H = 10  # number of  hidden neurons
    D_out = 1  # output size of G

    fake_data = torch.randn(BATCH_SIZE, 1, D_in)
    fake_data_label = torch.zeros(BATCH_SIZE)
    netD = CNNDiscriminator(D_in, D_out)
    objective = nn.BCELoss()
    fake_data_pred = netD(fake_data)
    print(f'fake_data_pred {fake_data_pred.shape}')
    # print(f'fake_data_pred {fake_data_pred.shape}: {fake_data_pred}')
    # loss = objective(fake_data_pred, fake_data_label)



