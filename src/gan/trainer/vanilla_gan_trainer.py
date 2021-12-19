from typing import Union

import numpy as np
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_holder import DataHolder
from src.data.weather.weather_dwd_importer import DWDWeatherDataImporter
from src.gan.discriminator.basic_discriminator import BasicDiscriminator
from src.gan.generator.basic_generator import BasicGenerator
from src.gan.trainer.typing import TrainModel
from src.utils.tensorboard_utils import TensorboardUtils, GraphPlotItem


class VanillaGANTrainer:
    def __init__(
            self,
            generator: TrainModel,
            discriminator: TrainModel,
            data_holder: DataHolder,
            noise_vector_size: int,
            sequence_length: int,
            batch_size: int = 10,
            device: Union[torch.device, int, str] = 'cpu'
    ) -> None:
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.data_holder = data_holder
        self.noise_vector_size = noise_vector_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        # self.features = features
        self.device = device

        print("Dataset Size:", self.data_holder.data.shape)
        if len(self.data_holder.data) % sequence_length != 0:
            raise ValueError(
                f'Cannot use a sequence length of {sequence_length} since the data set inside properly dividable {len(self.data_holder.data) % sequence_length=}')

        data = self.data_holder.data.reshape(-1, self.sequence_length, self.data_holder.get_feature_size())
        print("Data Size:", data.shape)
        self.data_loader = DataLoader(
            TensorDataset(torch.from_numpy(data)),
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

    def train_discriminator(self, real_data: Tensor, fake_data: Tensor):
        self.discriminator.optimizer.zero_grad()
        prediction_real = self.discriminator.model(real_data)
        error_real = self.objective(prediction_real, self.real_labels)
        prediction_fake = self.discriminator.model(fake_data)
        error_fake = self.objective(prediction_fake, self.fake_labels)
        discriminator_loss = error_real + error_fake
        discriminator_loss.backward()

        self.discriminator.optimizer.step()

        self.dis_losses.append(discriminator_loss.item())

    def train_generator(self, generated_data: Tensor):
        self.generator.optimizer.zero_grad()
        generated_data_prediction = self.discriminator.model(generated_data)
        error_generator = self.objective(generated_data_prediction, self.real_labels)
        error_generator.backward()
        self.generator.optimizer.step()
        self.gen_losses.append(error_generator.item())

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

    def __write_training_stats(self, real_data: Tensor, generated_data: Tensor):
        with torch.no_grad():
            if self.iter_no % 100 == 0:
                print(f'Iter {self.iter_no}: gen_loss={np.mean(self.gen_losses)}, dis_loss={np.mean(self.dis_losses)}')
                self.writer.add_scalar("gen_loss", np.mean(self.gen_losses), self.iter_no)
                self.writer.add_scalar("dis_loss", np.mean(self.dis_losses), self.iter_no)
                self.__reset_running_calculations()

            if self.iter_no % 500 == 0:
                real_data = real_data.detach().view(self.batch_size, self.sequence_length, -1).numpy()[0]
                pred_data = generated_data.detach().view(self.batch_size, self.sequence_length, -1).numpy()[0]
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
        self.__initialize_training()

        for epoch in range(max_epochs):
            for idx, data in enumerate(self.data_loader, 0):
                real_data = data[0].view(self.batch_size, -1)

                generated_data = self.generator.model(self.noise_vector()).detach()
                self.train_discriminator(real_data, generated_data)
                self.train_generator(generated_data)
                self.iter_no += 1
                self.__write_training_stats(real_data, generated_data)


if __name__ == '__main__':
    data_importer = DWDWeatherDataImporter()
    data_importer.initialize()
    data_holder = DataHolder(data_importer.data.values.astype(np.float32), data_importer.get_feature_labels())
    epochs = 10000
    noise_vector_size = 50
    sequence_length = 24
    batch_size = 10  # 24
    features = 5
    G_net = BasicGenerator(input_size=noise_vector_size, out_size=sequence_length * features, hidden_layers=[200, 300, 150])
    G_optim = torch.optim.Adam(G_net.parameters())
    G_sched = StepLR(G_optim, step_size=30, gamma=0.1)
    G = TrainModel(G_net, G_optim, G_sched)

    D_net = BasicDiscriminator(input_size=sequence_length * features, out_size=1, hidden_layers=[100, 50, 20])
    D_optim = torch.optim.Adam(D_net.parameters())
    D_sched = StepLR(D_optim, step_size=30, gamma=0.1)
    D = TrainModel(D_net, D_optim, D_sched)

    VanillaGANTrainer(G, D, data_holder, noise_vector_size, sequence_length, batch_size, 'cpu').train(epochs)
