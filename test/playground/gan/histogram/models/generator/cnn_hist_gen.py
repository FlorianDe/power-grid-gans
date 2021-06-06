import torch
import torch.nn as nn


class CNNHistogramGenerator(nn.Module):
    """
        Histogram model to to generate a histogram based n a noise vector input.

        * :attr:`channels_gen_out` number of channels of the feature vector.

        Args:
            noise_vector_size: Size of the noise vector to generate a histogram from.
    """

    def __init__(
            self,
            noise_vector_size: int,
            filters: int,
            gen_features_out: int,
            channels_gen_out: int = 1  # For now 1 since we only want to check for binary classification
    ):
        super(CNNHistogramGenerator, self).__init__()
        self.channels_gen_out = channels_gen_out
        self.conv_trans_1 = nn.ConvTranspose1d(
            in_channels=noise_vector_size,
            out_channels=filters * 64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.bn1 = nn.BatchNorm1d(num_features=filters * 64)
        self.relu1 = nn.ReLU(True)
        self.conv_trans_2 = nn.ConvTranspose1d(filters * 64, filters * 32, 3, 3, 1)
        self.bn2 = nn.BatchNorm1d(num_features=filters * 32)
        self.relu2 = nn.ReLU(True)
        self.conv_trans_3 = nn.ConvTranspose1d(filters * 32, filters * 16, 3, 2, 1)
        self.bn3 = nn.BatchNorm1d(num_features=filters * 16)
        self.relu3 = nn.ReLU(True)
        self.conv_trans_4 = nn.ConvTranspose1d(filters * 16, filters * 8, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(num_features=filters * 8)
        self.relu4 = nn.ReLU(True)
        self.conv_trans_5 = nn.ConvTranspose1d(filters * 8, filters * 4, 3, 2, 0)
        self.bn5 = nn.BatchNorm1d(num_features=filters * 4)
        self.relu5 = nn.ReLU(True)
        self.conv_trans_6 = nn.ConvTranspose1d(filters * 4, filters * 2, 3, 2, 0)
        self.bn6 = nn.BatchNorm1d(num_features=filters * 2)
        self.relu6 = nn.ReLU(True)
        self.conv_trans_7 = nn.ConvTranspose1d(filters * 2, filters * 1, 3, 2, 0)
        self.bn7 = nn.BatchNorm1d(num_features=filters * 1)
        self.relu7 = nn.ReLU(True)
        self.conv_trans_8 = nn.ConvTranspose1d(filters * 1, self.channels_gen_out, 3, 2, 0)
        self.bn8 = nn.BatchNorm1d(num_features=self.channels_gen_out)
        self.relu8 = nn.ReLU(True)
        self.linear = nn.Linear(in_features=223, out_features=gen_features_out)  # Hardcoded for now
        self.tanh = nn.Tanh()

        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f'Generator Parameter Count: {pytorch_total_params}')

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv_trans_1(x)))
        x = self.relu2(self.bn2(self.conv_trans_2(x)))
        x = self.relu3(self.bn3(self.conv_trans_3(x)))
        x = self.relu4(self.bn4(self.conv_trans_4(x)))
        x = self.relu5(self.bn5(self.conv_trans_5(x)))
        x = self.relu6(self.bn6(self.conv_trans_6(x)))
        x = self.relu7(self.bn7(self.conv_trans_7(x)))
        x = self.relu8(self.bn8(self.conv_trans_8(x)))
        x = self.tanh(self.linear(x))

        return x


if __name__ == "__main__":
    # torch.manual_seed(1)

    BATCH_SIZE = 10
    FILTERS = 4
    NOISE_VECTOR_SIZE = 64  # input size of G
    G_out = 144  # output size of G

    netG = CNNHistogramGenerator(NOISE_VECTOR_SIZE, FILTERS, G_out)
    # gen_dist = GeneratorDistribution(range=8)
    latent_vector_batch = torch.randn(BATCH_SIZE, NOISE_VECTOR_SIZE, 1)
    generated_data = netG(latent_vector_batch)
    print(f'Generated: {generated_data}, \n{generated_data.shape}')
