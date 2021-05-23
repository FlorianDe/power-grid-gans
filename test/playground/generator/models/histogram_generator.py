import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def print_tensor(name: str, t: torch.Tensor):
    print(f'{name} {t.shape}: {t}')

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        samples = np.linspace(-self.range, self.range, N) + \
                  np.random.random(N) * 0.01
        return samples


class HistogramGenerator(nn.Module):
    def __init__(self, G_in: int, H: int, G_out: int):
        super(HistogramGenerator, self).__init__()
        self.conv_trans_1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0),
        self.model = nn.Sequential(

            nn.BatchNorm1d(1),
            nn.ReLU(),

            # nn.Tanh()
        )
        # self.linear1 = nn.Linear(G_in, H)
        # self.linear2 = nn.Linear(H, G_out)

        # self.model = nn.Sequential(
        #     nn.Linear(in_features=latent_vector_size, out_features=out_features),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(
        #         in_channels=latent_vector_size,
        #         out_channels=filter_size*2,
        #         kernel_size=3,
        #         stride=1,
        #         padding=0
        #     ),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(filter_size*2),
        #     nn.Conv1d(
        #         in_channels=filter_size*2,
        #         out_channels=filter_size,
        #         kernel_size=3,
        #         stride=2,
        #     ),
        #     nn.Tanh()
        # )

    def forward(self, x):
        print_tensor('input', x)
        conv_trans_out = F.conv_transpose1d()
        print_tensor('input', x)

        return self.model(x)


if __name__ == "__main__":
    BATCH_SIZE = 2
    N = 8  # batch size
    G_in = 12  # input size of G
    H = 10  # number of  hidden neurons
    G_out = 24  # output size of G

    netG = HistogramGenerator(G_in=G_in, H=H, G_out=G_out)
    gen_dist = GeneratorDistribution(range=8)
    # latent_vector = [[]] * BATCH_SIZE
    # for i in range(BATCH_SIZE):
    #     latent_vector[i] = gen_dist.sample(G_in)
    # z = torch.FloatTensor(latent_vector)[..., None]
    # z = torch.from_numpy(np.array(latent_vector)).float()
    # gen_input_v = torch.FloatTensor(
    #     BATCH_SIZE, G_in, 1)
    latent_vector_batch = torch.randn(BATCH_SIZE, 1, G_in)
    print(f'Before: {latent_vector_batch}, \n{latent_vector_batch.shape}')
    fake = netG(latent_vector_batch)
    print(f'After: {fake}, \n{fake.shape}')

    # m = nn.Conv1d(100, 100, 1)
    # out = m(a)
    # print(out.size())
    # print(m)

