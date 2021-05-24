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
    """
        Histogram model to determine whether a histogram is fake or not.
        
        * :attr:`channels_disc_in` number of channels of the feature vector of the input data.
        * :attr:`channels_disc_out` number of channels for the output vector of the model.

        Args:
            filters: Number of filters which should be applied during each convolution
    """

    def __init__(self, filters: int):
        super(HistogramDiscriminator, self).__init__()
        self.channels_disc_in = 1  # For now 1 since we only look at one feature 'wattage'
        self.channels_disc_out = 1  # For now 1 since we only want to check for binary classification

        self.conv1 = nn.Conv1d(
            in_channels=self.channels_disc_in,
            out_channels=filters,
            kernel_size=7,
            stride=2,
            padding=1
        )
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(filters, filters * 2, 5, 2, 1)
        self.bn2 = nn.BatchNorm1d(filters * 2)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv1d(filters * 2, filters * 4, 4, 2, 1)
        self.bn3 = nn.BatchNorm1d(filters * 4)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv1d(filters * 4, filters * 8, 3, 2, 1)
        self.bn4 = nn.BatchNorm1d(filters * 8)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv1d(filters * 8, filters * 16, 3, 2, 1)
        self.bn5 = nn.BatchNorm1d(filters * 16)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6 = nn.Conv1d(filters * 16, filters * 32, 3, 2, 1)
        self.bn6 = nn.BatchNorm1d(filters * 32)
        self.relu6 = nn.LeakyReLU(0.2, inplace=True)
        self.conv7 = nn.Conv1d(filters * 32, filters * 64, 3, 2, 1)
        self.bn7 = nn.BatchNorm1d(filters * 64)
        self.relu7 = nn.LeakyReLU(0.2, inplace=True)
        self.conv8 = nn.Conv1d(filters * 64, self.channels_disc_out, 3, 2, 1)
        self.sigmoid = nn.Sigmoid()
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f'Discriminator Parameter Count: {pytorch_total_params}')

    def forward(self, x: torch.Tensor):
        """Forward the tensor.

        Args:
            x (torch.Tensor): input tensor of shape: [batch, channels, features]
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        x = self.sigmoid(x)  # Outputs need to be positiv for BCE
        x = x.view(-1, 1).squeeze(dim=1)
        return x


if __name__ == "__main__":
    # torch.manual_seed(1)

    BATCH_SIZE = 15
    FILTERS = 20
    D_in = 144  # input size of G
    H = 10  # number of  hidden neurons
    D_out = 6  # output size of G

    fake_data = torch.randn(BATCH_SIZE, 1, D_in)
    fake_data_label = torch.zeros(BATCH_SIZE)
    netD = HistogramDiscriminator(FILTERS)
    objective = nn.BCELoss()
    fake_data_pred = netD(fake_data)
    print(f'fake_data_pred {fake_data_pred.shape}: {fake_data_pred}')
    loss = objective(fake_data_pred, fake_data_label)
