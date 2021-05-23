import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np

from src.tensorboard.utils import TensorboardUtils, GraphPlotItem
from models.histogram_discriminator import HistogramDiscriminator
from models.histogram_generator import HistogramGenerator
from utils.histogram_utils import generate_noisy_normal_distribution


if __name__ == "__main__":
    BATCH_SIZE = 10
    LATENT_VECTOR_SIZE = 20
    HISTOGRAM_SIZE = 24
    LEARNING_RATE = 0.0001
    DISCR_FILTERS = 4
    GENER_FILTERS = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_discr = HistogramDiscriminator(
        filters=DISCR_FILTERS
    ).to(device)
    net_gener = HistogramGenerator(
        latent_vector_size=LATENT_VECTOR_SIZE,
        filters=GENER_FILTERS,
        gen_features_out=HISTOGRAM_SIZE
    ).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    gen_losses = []
    dis_losses = []
    iter_no = 0

    writer = SummaryWriter()
    for i in range(1_000_000):
        real_data = torch.from_numpy(np.array([generate_noisy_normal_distribution(HISTOGRAM_SIZE).astype(dtype=np.float32) for _ in range(BATCH_SIZE)])).reshape(BATCH_SIZE, 1, HISTOGRAM_SIZE)

        # fake samples, input is 3D: batch, filters, x
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1)
        gen_input_v.normal_(0, 1)
        gen_output_v = net_gener(gen_input_v)

        # # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(real_data)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1
        if iter_no % 1000 == 0:
            print(f'Iter {iter_no}: gen_loss={np.mean(gen_losses)}, dis_loss={np.mean(dis_losses)}')
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % 1000 == 0:
            sample_from_real_data = real_data.detach().squeeze(dim=1).numpy()[0]
            x = [i for i in range(len(sample_from_real_data))]
            TensorboardUtils.plot_graph_as_figure(
                tag="fake/real",
                writer=writer,
                plot_data=[
                    GraphPlotItem(
                        label="real",
                        x=x,
                        y=sample_from_real_data,
                        color='c'
                    ),
                    GraphPlotItem(
                        label="pred",
                        x=x,
                        y=gen_output_v.detach().squeeze(dim=1).numpy()[0],
                        color='r'
                    ),
                ],
                global_step=iter_no
            )
