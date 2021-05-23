import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def print_tensor(name: str, t: torch.Tensor):
    print(f'{name} {t.shape}: {t}')


class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N: int):
        samples = np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01
        return samples


class HistogramGenerator(nn.Module):
    """
        Histogram model to determine whether a histogram is fake or not.

        * :attr:`channels_gen_out` number of channels of the feature vector.

        Args:
            latent_vector_size: Size of the noise vector to generate a histogram from this is used as the in_channels
            size for the first conv layer.
    """

    def __init__(
            self,
            latent_vector_size: int,
            filters: int,
            gen_features_out: int,
            channels_gen_out: int = 1  # For now 1 since we only want to check for binary classification
    ):
        super(HistogramGenerator, self).__init__()
        self.channels_gen_out = channels_gen_out
        self.conv_trans_1 = nn.ConvTranspose1d(
            in_channels=latent_vector_size,
            out_channels=filters * 8,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.conv_trans_2 = nn.ConvTranspose1d(filters * 8, filters * 4, 3, 3, 1)
        self.conv_trans_3 = nn.ConvTranspose1d(filters * 4, filters * 2, 3, 2, 1)
        self.conv_trans_4 = nn.ConvTranspose1d(filters * 2, filters * 1, 3, 1, 1)
        self.conv_trans_5 = nn.ConvTranspose1d(filters * 1, self.channels_gen_out, 3, 2, 0)
        self.linear = nn.Linear(in_features=27, out_features=gen_features_out)  # Hardcoded for now

        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print(f'Generator Parameter Count: {pytorch_total_params}')

    def forward(self, x):
        x = F.relu(self.conv_trans_1(x))
        x = F.relu(self.conv_trans_2(x))
        x = F.relu(self.conv_trans_3(x))
        x = F.relu(self.conv_trans_4(x))
        x = F.relu(self.conv_trans_5(x))
        x = self.linear(x)

        return x


if __name__ == "__main__":
    torch.manual_seed(1)

    BATCH_SIZE = 3
    FILTERS = 4
    LATENT_VECTOR_SIZE = 20  # input size of G
    G_out = 24  # output size of G

    netG = HistogramGenerator(LATENT_VECTOR_SIZE, FILTERS, G_out)
    # gen_dist = GeneratorDistribution(range=8)
    latent_vector_batch = torch.randn(BATCH_SIZE, LATENT_VECTOR_SIZE, 1)
    generated_data = netG(latent_vector_batch)
    print(f'Generated: {generated_data}, \n{generated_data.shape}')
