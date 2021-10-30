from typing import Union, Optional

import numpy as np
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.gan.trainer.base_trainer import BaseTrainer
from src.data.data_holder import DataHolder
from src.data.importer.weather.weather_dwd_importer import DWDWeatherDataImporter
from src.gan.trainer.trainer_types import TrainModel
from src.net import CustomModule
from src.net.dynamic import FNN
from src.utils.datetime_utils import dates_to_conditional_vectors
from src.utils.tensorboard_utils import TensorboardUtils, GraphPlotItem


# torch.autograd.set_detect_anomaly(True)
from src.utils.path_utils import get_root_project_path


class CGANTrainer(BaseTrainer):
    def __init__(self,
                 generator: TrainModel,
                 discriminator: TrainModel,
                 data_holder: DataHolder,
                 noise_vector_size: int,
                 batch_size: int = 10,
                 device: Union[torch.device, int, str] = 'cpu'
                 ) -> None:
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            data_holder=data_holder,
            device=device
        )
        self.noise_vector_size = noise_vector_size
        self.batch_size = batch_size

        print("Dataset Size:", self.data_holder.data.shape)
        print("Labels Size:", self.data_holder.x.shape)

        self.data_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.data_holder.data), torch.from_numpy(self.data_holder.x)),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True
        )
        self.objective = nn.BCELoss()  # TODO REPLACE WITH CUSTOM LOSS FUNCTIONS!
        self.real_labels = torch.ones(self.batch_size, requires_grad=False, device=self.device)
        self.fake_labels = torch.zeros(self.batch_size, requires_grad=False, device=self.device)
        self.gen_losses = []
        self.dis_losses = []
        self.iter_no = 0
        self.writer = SummaryWriter()

    def train_discriminator(self, real_data: Tensor, labels: Tensor):
        pass

    def train_generator(self, generated_data: Tensor):
        pass

    def noise_vector(self):
        # Probably refactor to the generator class, since itself knows its input size!
        # features = self.data.shape[1]
        return torch.rand(self.batch_size, self.noise_vector_size)

    def __reset_running_calculations(self):
        self.gen_losses = []
        self.dis_losses = []

    def __initialize_training(self):
        self.__reset_running_calculations()
        self.iter_no = 0
        self.discriminator.model.train()
        self.generator.model.train()

    def __write_training_stats(self, epoch: int, real_data: Tensor, real_labels: Tensor):
        with torch.no_grad():
            if self.iter_no % 500 == 0:
                print(f'{epoch}. Epoch: Iter {self.iter_no}: gen_loss={np.mean(self.gen_losses)}, dis_loss={np.mean(self.dis_losses)}')
                self.writer.add_scalar("gen_loss", np.mean(self.gen_losses), self.iter_no)
                self.writer.add_scalar("dis_loss", np.mean(self.dis_losses), self.iter_no)
                self.__reset_running_calculations()

            if self.iter_no % 2000 == 0:
                real_data = self.data_holder.normalizer.renormalize(real_data.detach().view(self.batch_size, -1).numpy())
                noise = torch.from_numpy(
                    np.repeat(np.random.normal(0, 1, (1, self.noise_vector_size)).astype(dtype=np.float32), repeats=self.batch_size, axis=0))
                generated_data = self.generator.model(noise, real_labels)
                pred_data = self.data_holder.normalizer.renormalize(generated_data.detach().view(self.batch_size, -1).numpy())
                x = [i for i in range(len(real_data))]
                real_data_trans = real_data.transpose()
                TensorboardUtils.plot_graph_as_figure(
                    tag="real",
                    writer=self.writer,
                    plot_data=[GraphPlotItem(
                        label=f'real_{i}',
                        x=x,
                        y=real_data_trans[i]
                    ) for i in range(len(real_data_trans))],
                    global_step=self.iter_no
                )
                pred_data_trans = pred_data.transpose()
                TensorboardUtils.plot_graph_as_figure(
                    tag="fake",
                    writer=self.writer,
                    plot_data=[GraphPlotItem(
                        label=f'fake_{i}',
                        x=x,
                        y=pred_data_trans[i]
                    ) for i in range(len(pred_data_trans))],
                    global_step=self.iter_no
                )

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            for i, (real_data, labels) in enumerate(self.data_loader):
                # Configure input
                # real_data = data[0].view(self.batch_size, -1)
                # real_labels = Variable(labels.type(LongTensor))

                #  Train Generator
                self.generator.optimizer.zero_grad()

                # Sample noise and labels as generator input
                # multiple random noise vectors
                # z = torch.from_numpy(np.random.normal(0, 1, (self.batch_size, self.noise_vector_size)).astype(dtype=np.float32))
                # same noise vector batch-times
                z = torch.from_numpy(np.repeat(np.random.normal(0, 1, (1, self.noise_vector_size)), self.batch_size, axis=0).astype(dtype=np.float32))

                # TODO Generation should check for invalid days for specific month like 31.02 => isn't valid
                months = np.random.randint(1, 12, self.batch_size)
                days = np.random.randint(1, 31, self.batch_size)
                hours = np.random.randint(0, 23, self.batch_size)
                gen_labels = torch.tensor(dates_to_conditional_vectors(months, days, hours), dtype=torch.float32, requires_grad=False)

                # Generate a batch of data
                generated_data = self.generator.model(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = self.discriminator.model(generated_data, gen_labels)
                g_loss = self.objective(validity, self.real_labels)

                g_loss.backward()
                self.generator.optimizer.step()
                self.gen_losses.append(g_loss.item())

                #  Train Discriminator
                self.discriminator.optimizer.zero_grad()

                # Loss for real data
                validity_real = self.discriminator.model(real_data, labels)
                d_real_loss = self.objective(validity_real, self.real_labels)

                # Loss for fake data
                validity_fake = self.discriminator.model(generated_data.detach(), gen_labels)
                d_fake_loss = self.objective(validity_fake, self.fake_labels)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.discriminator.optimizer.step()
                self.dis_losses.append(d_loss.item())

                self.iter_no += 1
                self.__write_training_stats(epoch, real_data, labels)


class CGANRNNGenerator(CustomModule):
    def __init__(self, input_size: int, out_size: int):
        super(CGANRNNGenerator, self).__init__()

    def forward(self, x, h):
        return x, h


class CGANRNNDiscriminator(CustomModule):
    def __init__(self, input_size: int, out_size: int = 1):
        super(CGANRNNDiscriminator, self).__init__()
        self.hidden_dim = 10
        self.n_layers = 20

        self.gru = nn.GRU(input_size, self.hidden_dim, self.n_layers, batch_first=True, dropout=0.8)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_dim, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, labels):
        x = torch.dstack((x, labels))
        out, h = self.gru(x, h)
        print(f'{out.shape=}, {out=}')
        out = self.fc(self.relu(out[:, -1]))
        out = self.sigmoid(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden


class CGANBasicGenerator(CustomModule):
    def __init__(self, input_size: int, out_size: int, hidden_layers: Optional[list[int]] = None):
        super(CGANBasicGenerator, self).__init__()
        self.fnn = FNN(input_size, out_size, hidden_layers)

    def forward(self, noise: Tensor, labels: Tensor):
        gen_input = torch.cat((labels, noise), -1)
        return self.fnn(gen_input)


class CGANBasicDiscriminator(CustomModule):
    def __init__(self, input_size: int, out_size: int, hidden_layers: Optional[list[int]] = None):
        super(CGANBasicDiscriminator, self).__init__()
        self.fnn = FNN(input_size, out_size, hidden_layers)
        self.activation = nn.Sigmoid()

    def forward(self, data: Tensor, labels: Tensor):
        x = torch.cat((data, labels), -1)
        x = self.fnn(x)
        x = self.activation(x)
        x = x.view(-1)
        return x


if __name__ == '__main__':
    data_importer = DWDWeatherDataImporter()
    data_importer.initialize()
    data_holder = DataHolder(data_importer.data.values.astype(np.float32), np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())))
    epochs = 10
    noise_vector_size = 50
    # sequence_length = 24
    batch_size = 24  # *7
    features = data_holder.get_feature_size()
    G_net = CGANBasicGenerator(input_size=noise_vector_size + 14, out_size=features, hidden_layers=[200])
    G_optim = torch.optim.Adam(G_net.parameters())
    G_sched = StepLR(G_optim, step_size=30, gamma=0.1)
    G = TrainModel(G_net, G_optim, G_sched)

    D_net = CGANBasicDiscriminator(input_size=features + 14, out_size=1, hidden_layers=[100, 50, 20])
    D_optim = torch.optim.Adam(D_net.parameters())
    D_sched = StepLR(D_optim, step_size=30, gamma=0.1)
    D = TrainModel(D_net, D_optim, D_sched)

    trainer = CGANTrainer(G, D, data_holder, noise_vector_size, batch_size, 'cpu')
    trainer.train(epochs)

    path = get_root_project_path().joinpath('runs').joinpath('model-test').absolute()
    trainer.save_model(path)

    # # [batch, sequence, features]
    # b1 = np.array([
    #     [1.1, 2.2],
    #     [3.1, 4.2],
    #     [6.2, 7.3]
    # ])
    # b2 = np.array([
    #     [1.4, 2.5],
    #     [3.4, 4.6],
    #     [6.3, 7.2]
    # ])
    # data_batch = torch.from_numpy(np.array([b1, b2 ])).type(torch.FloatTensor)
    #
    #
    # # [batch, sequence, time_vector]
    # labels_batch = torch.from_numpy(np.array([
    #     [
    #         [0, 0, 0, 1],
    #         [0, 0, 1, 0],
    #         [0, 0, 1, 1]
    #     ], [
    #         [0, 1, 0, 0],
    #         [0, 1, 0, 1],
    #         [0, 1, 1, 0]
    #     ]
    # ])).type(torch.FloatTensor)
    #
    # input_size = data_batch.shape[-1] + labels_batch.shape[-1]
    # disc = CGANRNNDiscriminator(input_size)
    # h = disc.init_hidden(len(data_batch))
    #
    # y, hn = disc(data_batch, h, labels_batch)
    # print(f'{y=}, {hn=}')
