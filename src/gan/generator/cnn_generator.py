import torch
import torch.nn as nn
from torch import Tensor

from net import CustomModule


class CNNGenerator(CustomModule):
    def __init__(self, input_size: int, out_size: int):
        super(CNNGenerator, self).__init__()
        self.filters = 60
        self.input_size = input_size
        self.out_size = out_size
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.filters,
            kernel_size=(1,),
            dilation=(1,),
            padding="same"
        )
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv1d(self.filters, self.filters, kernel_size=(8,), dilation=(2,), padding="same")
        self.bn2 = nn.BatchNorm1d(self.filters)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv1d(self.filters, self.filters, kernel_size=(5,), dilation=(2,),  padding="same")
        self.bn3 = nn.BatchNorm1d(self.filters)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv1d(self.filters, self.filters, kernel_size=(5,), dilation=(2,),  padding="same")
        self.bn4 = nn.BatchNorm1d(self.filters)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv1d(self.filters, self.filters, kernel_size=(5,), dilation=(2,),  padding="same")
        self.bn5 = nn.BatchNorm1d(self.filters)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv1d(self.filters, self.filters, kernel_size=(5,), dilation=(2,),  padding="same")

        # self.dense = nn.Linear(self.input_size*self.last_filter)
        self.activation = nn.Tanh()
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f'CNNGenerator Parameter Count: {pytorch_total_params}')

    def forward(self, x: Tensor):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.activation(x)
        x = x.view(-1, self.out_size, self.input_size*self.filters)
        return x

    def reshape(self, x):
        return x


if __name__ == '__main__':
    # torch.manual_seed(1)

    BATCH_SIZE = 15
    NOISE_VECTOR_SIZE = 16  # input size of G
    G_out = 1  # output size of G

    netG = CNNGenerator(NOISE_VECTOR_SIZE)
    # latent_vector_batch = torch.randn(BATCH_SIZE, NOISE_VECTOR_SIZE, 1)
    latent_vector_batch = torch.randn(BATCH_SIZE, 1, NOISE_VECTOR_SIZE)
    generated_data = netG(latent_vector_batch)
    print(f'Generated: {generated_data}, \n{generated_data.shape}')



