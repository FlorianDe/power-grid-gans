import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np

from src.tensorboard.utils import TensorboardUtils, GraphPlotItem
from utils.histogram_utils import generate_noisy_normal_distribution


class HistogramDiscriminator(nn.Module):
    def __init__(self, input_resolution: int, filter_size: int):
        super(HistogramDiscriminator, self).__init__()
        self.input_resolution = input_resolution
        self.model = nn.Sequential(
            nn.Linear(in_features=input_resolution, out_features=1),
            nn.Sigmoid()
            # nn.Conv1d(
            #     in_channels=self.input_resolution,
            #     out_channels=filter_size,
            #     kernel_size=3,
            #     stride=2,
            #     padding=1
            # ),
            # nn.ReLU(),
            # nn.BatchNorm1d(filter_size),
            # nn.Conv1d(
            #     in_channels=filter_size,
            #     out_channels=filter_size*2,
            #     kernel_size=3,
            #     stride=1,
            #     padding=0
            # ),
            # nn.Sigmoid()  # replace via a softmax for labels!
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1, 1).squeeze(dim=1)


class HistogramGenerator(nn.Module):
    def __init__(self, latent_vector_size: int, out_features: int):
        super(HistogramGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=latent_vector_size, out_features=out_features),
            nn.ReLU(),
            # nn.ConvTranspose1d(
            #     in_channels=latent_vector_size,
            #     out_channels=filter_size*2,
            #     kernel_size=3,
            #     stride=1,
            #     padding=0
            # ),
            # nn.ReLU(),
            # nn.BatchNorm1d(filter_size*2),
            # nn.Conv1d(
            #     in_channels=filter_size*2,
            #     out_channels=filter_size,
            #     kernel_size=3,
            #     stride=2,
            # ),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    BATCH_SIZE = 20
    LATENT_VECTOR_SIZE = 100
    HISTOGRAM_SIZE = 1440
    LEARNING_RATE = 0.0001
    DISCR_FILTERS = HISTOGRAM_SIZE
    GENER_FILTERS = HISTOGRAM_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_discr = HistogramDiscriminator(HISTOGRAM_SIZE, DISCR_FILTERS).to(device)
    net_gener = HistogramGenerator(LATENT_VECTOR_SIZE, HISTOGRAM_SIZE).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))

    true_labels_v = torch.tensor([1.0]) # torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.tensor([0.0]) # torch.zeros(BATCH_SIZE, device=device)

    gen_losses = []
    dis_losses = []
    iter_no = 0

    writer = SummaryWriter()
    for i in range(1_000_000):
        real_data = torch.from_numpy(generate_noisy_normal_distribution(HISTOGRAM_SIZE)).type(torch.FloatTensor)

        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(LATENT_VECTOR_SIZE)
        gen_input_v.normal_(0, 1)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(real_data)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())
        #
        iter_no += 1
        if iter_no % 1000 == 0:
            print(f'Iter {iter_no}: gen_loss={np.mean(gen_losses)}, dis_loss={np.mean(dis_losses)}')
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % 2000 == 0:
            x = [i for i in range(len(real_data))]
            TensorboardUtils.plot_graph_as_figure(
                tag="fake/real",
                writer=writer,
                plot_data=[
                    GraphPlotItem(
                        label="real",
                        x=x,
                        y=real_data.detach().numpy(),
                        color='c'
                    ),
                    GraphPlotItem(
                        label="pred",
                        x=x,
                        y=gen_output_v.detach().numpy(),
                        color='r'
                    ),
                ],
                global_step=iter_no
            )
