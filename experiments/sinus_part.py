import math

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from matplotlib import pyplot as plt
from time import time
from torch.autograd import Variable

# from google.colab import files
from src.utils.path_utils import get_root_project_path

# import imageio

torch.manual_seed(13333)  # reproducible


# make our networks
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer_1 = torch.nn.Linear(1, 100)
        self.layer_2 = torch.nn.Linear(100, 100)
        self.layer_3 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = F.leaky_relu_(self.layer_1(x))
        x = F.leaky_relu_(self.layer_2(x))
        x = self.layer_3(x)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer_1 = torch.nn.Linear(2, 100)
        self.layer_2 = torch.nn.Linear(100, 100)
        self.layer_3 = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer_1(x))
        x = torch.sigmoid(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))
        return x


def get_generator_input_sampler(n):
    return 2 * torch.rand(n, 1) - 1  # Uniform-dist data into generator, _NOT_ Gaussian


if __name__ == '__main__':
    img_paths = get_root_project_path() / "runs/experiments/02_sine_data/images/"
    img_paths.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(10)  # reproducible

    x = torch.unsqueeze(torch.linspace(-1, 1, 48), dim=1)
    y = torch.sin((2*math.pi/24)*x) * torch.rand(x.size())

    G = Generator()
    D = Discriminator()

    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    BATCH_SIZE = 8
    EPOCH = 5000
    my_images = []

    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)

    fig, ax = plt.subplots(figsize=(16, 5))

    for epoch in range(EPOCH):

        step_count = 0

        for step, (batch_x, batch_y) in enumerate(loader):
            step_count += 1
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            d_real_data = Variable(torch.tensor(np.concatenate((b_x, b_y), axis=1)))
            d_real_decision = D(d_real_data)
            d_real_error = criterion(d_real_decision, Variable(torch.ones([BATCH_SIZE, 1])))  # ones = true
            d_real_error.backward()  # compute/store gradients, but don't change params

            #  1B: Train D on fake
            d_gen_input = Variable(get_generator_input_sampler(BATCH_SIZE))
            d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = D(d_fake_data)
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([BATCH_SIZE, 1])))  # zeros = fake
            d_fake_error.backward()
            d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(step_count):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = Variable(get_generator_input_sampler(BATCH_SIZE))
            g_fake_data = G(gen_input)
            dg_fake_decision = D(g_fake_data)
            g_error = criterion(dg_fake_decision, Variable(torch.ones([BATCH_SIZE, 1])))  # Train G to pretend it's genuine

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        print('epoch = ' + str(epoch), end='\r')
        if epoch % 100 == 0:
            ax.clear()
            # plt.cla()
            ax.set_title('GANS', fontsize=30)
            ax.set_xlabel('x', fontsize=24)
            ax.set_ylabel('y', fontsize=24)
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-2.0, 2.0)
            ax.scatter(x, y, color="orange", alpha=0.04)

            g_input = Variable(get_generator_input_sampler(1000))
            g_output = G(g_input)
            d_output = D(g_output)
            d_correct = g_output[np.where(np.round(d_output.data.numpy()) == 0)[0], :]
            d_fooled = g_output[np.where(np.round(d_output.data.numpy()) == 1)[0], :]

            ax.text(-1.3, -1.2, 'Fooled Discriminator', fontdict={'size': 20, 'color': 'green'})
            ax.text(-1.3, -1.6, 'Did not fool Discriminator', fontdict={'size': 20, 'color': 'indianred'})

            accuracy = len(d_fooled.data.numpy()) / len(d_output.data.numpy())
            ax.text(0.7, -1.2, 'Epoch = %d' % epoch, fontdict={'size': 20, 'color': 'black'})
            ax.text(0.7, -1.6, 'Accuracy = %.4f' % accuracy, fontdict={'size': 20, 'color': 'black'})

            ax.scatter(d_correct[:, 0].data.numpy(), d_correct[:, 1].data.numpy(), color='indianred', alpha=.3)
            ax.scatter(d_fooled[:, 0].data.numpy(), d_fooled[:, 1].data.numpy(), color='green', alpha=.6)

            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            my_images.append(image)

    # save images as a gif
    name = 'gan_' + str(time()) + '.gif'
    ani_path = img_paths / name
    imageio.mimsave(ani_path, my_images, fps=10)
