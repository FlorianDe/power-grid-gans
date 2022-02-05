from tqdm import trange

from functools import reduce

import math
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from experiments.utils import get_experiments_folder

save_images_path = get_experiments_folder().joinpath("02_sinusoidal_data_basic_gan").joinpath("02_01_vanilla_gan_fnn_sines")
save_images_path.mkdir(parents=True, exist_ok=True)

class DiscriminatorFNN(nn.Module):
    def __init__(
        self,
        features: int,
        sequence_len: int,
        out_features: int = 1
    ):
        super(DiscriminatorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.dense1 = nn.Linear(
            in_features=sequence_len*features,
            out_features=out_features,
        )
        self.activation1 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation1(self.dense1(x))


class GeneratorFNN(nn.Module):
    def __init__(
        self,
        latent_vector_size: int,
        features: int,
        sequence_len: int,
    ):
        super(GeneratorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.dense1 = nn.Linear(
            in_features=latent_vector_size,
            out_features=features*sequence_len,
        )
        self.activation1 = nn.Tanh()  # check dis

    def forward(self, x: Tensor) -> Tensor:
        return self.dense1(x) # removed tanh


def generate_sine_features(sequence_len: int, amplitudes: list[float] = [1], times: int = 1, noise_scale: float = 0.05, seed: int = 42) -> tuple[Tensor, Tensor]:
    """
        Returns a multi dimensional sine wave feature of shape [times, sequence_len, features]
    """
    torch.manual_seed(seed)

    size = times*sequence_len
    features = len(amplitudes)
    a = torch.tensor(amplitudes).view(1, features)
    x = torch.linspace(0, size, size).view(size, 1)
    sine = torch.sin((2*math.pi/sequence_len)*x)
    scaled_sines = (sine*a).view(times, sequence_len, features)
    noises = noise_scale*torch.rand(scaled_sines.shape)

    return (scaled_sines+noises)


def train(
    G: nn.Module,
    D: nn.Module,
    samples: list[npt.ArrayLike],
    epochs: int,
    batch_size: int,
    latent_vector_size: int,
    device: torch.device = torch.device("cpu")
):
    # Learning rate for optimizers
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    # TODO FOR NOW JUST CONCAT THEM!
    for i, s in enumerate(samples):
        print(f"{i}. sample: {s.shape=}")
    flattened_samples = reduce(lambda acc, cur: torch.concat((acc,cur)), samples)
    print(f"{flattened_samples.shape=}")
    # dataset = TensorDataset(flattened_samples)
    dataloader = DataLoader(
        flattened_samples,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=workers
    )


    G_losses = []
    D_losses = []
    iters = 0
    for epoch in trange(epochs, unit="epoch"):
        for batch_idx, (data_batch) in enumerate(dataloader):
            data_batch = data_batch.view(batch_size, -1)
            # print(f"{data_batch.shape=}")

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            real_labels = torch.ones(batch_size, requires_grad=False, device=device)
            fake_labels = torch.zeros(batch_size, requires_grad=False, device=device)
  
            ## Train with all-real batch
            D.zero_grad()
            # label = torch.full((batch_size), real_label_value, dtype=torch.float, device=device)
            d_out_real = D(data_batch).view(-1)
            # print(f"{d_out_real.shape=}")
            d_err_real = criterion(d_out_real, real_labels)
            d_err_real.backward()
            D_x = d_err_real.mean().item()
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, latent_vector_size, device=device)
            fake_generated = G(noise)
            # print(f"{fake_generated.shape=}")
            d_out_fake = D(fake_generated.detach()).view(-1)
            d_err_fake = criterion(d_out_fake, fake_labels)
            d_err_fake.backward()
            D_G_z1 = d_out_fake.mean().item()
            err_D = d_err_real + d_err_fake
            # Update weights of D
            optimizerD.step()


            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            G.zero_grad()
            d_out_fake = D(fake_generated).view(-1)
            err_G = criterion(d_out_fake, real_labels)
            err_G.backward()
            D_G_z2 = d_out_fake.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(dataloader),
                        err_D.item(), err_G.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            # G_losses.append(err_G.item())
            # D_losses.append(err_D.item())

            iters += 1

        if epoch % 500 == 0:
                with torch.no_grad():
                    samples = 10
                    noise = torch.randn(samples, latent_vector_size, device=device)
                    generated_sine = G(noise)
                    generated_sine = generated_sine.view(samples, sequence_len, features_len)
                    print(f"{generated_sine.shape=}")
                    fig, ax = plot_sample(generated_sine)
                    fig.savefig(save_images_path / f"{epoch}.png", bbox_inches='tight', pad_inches=0)


        # print(f"{epoch=}")
    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

def plot_sample(sample: Tensor):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    unbind_sample = torch.unbind(sample)
    flattened_sample = torch.concat(unbind_sample)
    for i, y in enumerate(torch.transpose(flattened_sample, 0, 1)):
         ax.plot(range(len(y)), y)
    return fig, ax

if __name__ == '__main__':
    epochs = 50000
    latent_vector_size = 10
    amplitudes = [0.25, 1.0, 2.0]
    features_len = len(amplitudes)
    sequence_len = 24
    batch_size = 50 #TODO CHECK INSIDE LOOP, COULD BE LESS THAN BATCH SIZE!
    samples = [
        generate_sine_features(sequence_len=sequence_len,
                               amplitudes=amplitudes, times=batch_size*10),
        # generate_sine_features(sequence_len=sequence_len,
        #                        amplitudes=[0.5], times=batch_size*10),
        # generate_sine_features(sequence_len=sequence_len,
        #                 amplitudes=[0.25], times=batch_size*10),
    ]
    # Set a feature to a const value, e.g. 0.5
    # for sample in samples:
    #     for values in sample:
    #         for a in values:
    #             a[2] =0.5

    device = torch.device("cpu")

    G = GeneratorFNN(
            latent_vector_size=latent_vector_size,
            features=features_len,
            sequence_len=sequence_len,
        )

    train(
        epochs=epochs,
        samples=samples,
        batch_size=batch_size,
        latent_vector_size=latent_vector_size,
        G=G,
        D=DiscriminatorFNN(
            features=features_len,
            sequence_len=sequence_len,
            out_features=1
        ),
    )

    # plot_sample(samples[0])


