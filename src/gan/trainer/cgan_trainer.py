from typing import Optional, Union

from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.gan.discriminator.basic_discriminator import BasicDiscriminator
from src.gan.generator.basic_generator import BasicGenerator
from src.gan.trainer.base_trainer import BaseTrainer
from src.data.data_holder import DataHolder
from src.data.weather.weather_dwd_importer import DWDWeatherDataImporter
from src.gan.trainer.typing import (
    BatchReshaper,
    ConditionalTrainParameters,
    EpochPredicate,
    NoiseGenerator,
    TrainModel,
    TrainParameters,
    TrainerCallback,
)
from src.net.summary.net_parsing import print_net_summary
from src.net.summary.net_summary import LatexTableOptions
from src.utils.datetime_utils import dates_to_conditional_vectors
from src.utils.path_utils import get_root_project_path

# torch.autograd.set_detect_anomaly(True)


def default_fnn_batch_reshaper(data_batch: Tensor, current_batch_size: int, params: TrainParameters) -> Tensor:
    return data_batch.view(current_batch_size, params.sequence_len * params.features_len)


def default_fnn_noise_generator(current_batch_size: int, params: TrainParameters) -> Tensor:
    return torch.randn(current_batch_size, params.latent_vector_size, device=params.device)


class DiscriminatorFNN(nn.Module):
    def __init__(
        self,
        features: int,
        sequence_len: int,
        conditions: int,
        embeddings: int,
        out_features: int = 1,
        dropout: float = 0.5,
    ):
        super(DiscriminatorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        input_size = sequence_len * features

        def dense_block(input: int, output: int, normalize=True):
            negative_slope = 1e-2
            layers: list[nn.Module] = []
            if normalize:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(input, output))
            # layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            return layers

        self.embedding = nn.Embedding(conditions, embeddings)
        self.fnn = nn.Sequential(
            *dense_block(input_size + embeddings, 2 * input_size, False),
            *dense_block(2 * input_size, 4 * input_size),
            *dense_block(4 * input_size, 8 * input_size),
            # *dense_block(8 * input_size, 16 * input_size),
            # *dense_block(16 * input_size, 8 * input_size),
            *dense_block(8 * input_size, 4 * input_size),
            nn.Linear(4 * input_size, out_features),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        embedded_conditions = self.embedding(condition)
        x = torch.cat((x, embedded_conditions), dim=1)
        x = self.fnn(x)
        x = self.sigmoid(x)
        return x


class GeneratorFNN(nn.Module):
    def __init__(
        self,
        latent_vector_size: int,
        features: int,
        sequence_len: int,
        conditions: int,
        embeddings: int,
        dropout: float = 0.5,
    ):
        super(GeneratorFNN, self).__init__()
        self.features = features
        self.sequence_len = sequence_len
        self.embedding = nn.Embedding(conditions, embeddings)

        def dense_block(input: int, output: int, normalize=True):
            negative_slope = 1e-2
            layers: list[nn.Module] = []
            if normalize:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(input, output))
            # layers.append(nn.BatchNorm1d(output, 0.8))
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
            return layers

        self.fnn = nn.Sequential(
            *dense_block(latent_vector_size + embeddings, 2 * latent_vector_size),
            *dense_block(2 * latent_vector_size, 4 * latent_vector_size),
            *dense_block(4 * latent_vector_size, 8 * latent_vector_size),
            # *dense_block(8 * latent_vector_size, 16 * latent_vector_size),
            # *dense_block(16 * latent_vector_size, 8 * latent_vector_size),
            *dense_block(8 * latent_vector_size, 4 * latent_vector_size),
            nn.Linear(4 * latent_vector_size, features * sequence_len),
        )
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        embedded_conditions = self.embedding(condition)
        x = torch.cat((x, embedded_conditions), dim=1)
        x = self.fnn(x)
        # x = self.tanh(x)
        return x


class CGANTrainer(BaseTrainer):
    def __init__(
        self,
        generator: TrainModel,
        discriminator: TrainModel,
        data_holder: DataHolder,
        params: ConditionalTrainParameters,
        noise_generator: NoiseGenerator = default_fnn_noise_generator,
        batch_reshaper: BatchReshaper = default_fnn_batch_reshaper,
        callback_options: list[tuple[EpochPredicate, Union[list[TrainerCallback], TrainerCallback]]] = [],
        latex_options: Optional[LatexTableOptions] = None,
    ) -> None:
        super().__init__(
            generator=generator, discriminator=discriminator, data_holder=data_holder, device=params.device
        )
        self.params = params
        if params.features_len is None:  # Try to set the feature length from the data_holder
            print(
                f"The features length was not provided specifically, use the feature size from the data holder {data_holder.get_feature_size()}"
            )
            params.features_len = data_holder.get_feature_size()
        self.noise_generator = noise_generator
        self.batch_reshaper = batch_reshaper
        self.callback_options = callback_options
        self.latex_options = latex_options

        self.criterion = nn.BCELoss()  # nn.BCEWithLogitsLoss()
        self.gen_losses = []
        self.dis_losses = []
        self.iter_no = 0

        # This part could be generified somewhat
        data = torch.from_numpy(data_holder.data).view(-1, 24, data_holder.get_feature_size())
        data_conditions = torch.from_numpy(data_holder.conditions).view(-1, 24)[..., 0]
        dataset = TensorDataset(data, data_conditions)
        self.dataloader = DataLoader(
            dataset,
            batch_size=params.batch_size,
            shuffle=False,
            # num_workers=workers
        )

        if latex_options:
            (data_batch, conditions_batch) = next(iter(self.dataloader))
            discriminator_input_size = batch_reshaper(data_batch, params.batch_size, params)[0].size()
            generator_input_size = noise_generator(params.batch_size, params)[0].size()
            conditions_size = 1  # extract from conditions_batch
            print_net_summary(
                G=self.generator.model,
                D=self.discriminator.model,
                generator_input_size=[generator_input_size, conditions_size],
                discriminator_input_size=[discriminator_input_size, conditions_size],
                latex_options=latex_options,
                dtypes=[torch.FloatTensor, torch.IntTensor],
            )

    def __reset_running_calculations(self):
        self.gen_losses = []
        self.dis_losses = []

    def __initialize_training(self):
        self.__reset_running_calculations()
        self.iter_no = 0
        self.discriminator.model.train()
        self.generator.model.train()

    # def __prepare_generator_input(self, noise, labels):
    #     return torch.cat((labels, noise), -1)

    # def __prepare_discriminator_input(self, data, labels):
    #     return torch.cat((data, labels), -1)

    def train(self, max_epochs):
        self.__initialize_training()

        for epoch in (
            progress := tqdm(
                range(1, max_epochs + 1), unit="epochs", bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}"
            )
        ):
            for batch_idx, (data_batch, conditions_batch) in enumerate(self.dataloader):
                current_batch_size = min(self.params.batch_size, data_batch.shape[0])
                data_batch = self.batch_reshaper(data_batch, current_batch_size, self.params)
                # ADD NOISE
                data_batch = data_batch + 0.01 * torch.randn(data_batch.shape, device=self.device)

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                real_labels = torch.ones(current_batch_size, requires_grad=False, device=self.device)  # try with 0.9
                # real_labels = torch.squeeze(
                #     torch.full((current_batch_size, 1), 0.9, requires_grad=False, device=params.device)
                # )
                fake_labels = torch.zeros(current_batch_size, requires_grad=False, device=self.device)

                ## Train with all-real batch
                self.discriminator.model.zero_grad()
                # label = torch.full((current_batch_size), real_label_value, dtype=torch.float, device=params.device)
                d_out_real = self.discriminator.model(data_batch, conditions_batch).view(-1)
                # print(f"{d_out_real.shape=}")
                d_err_real = self.criterion(d_out_real, real_labels)
                d_err_real.backward()
                D_x = d_err_real.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = self.noise_generator(current_batch_size, self.params)
                fake_generated = self.generator.model(noise, conditions_batch)
                # print(f"{fake_generated.shape=}")
                d_out_fake = self.discriminator.model(fake_generated.detach(), conditions_batch).view(
                    -1
                )  # TODO OTHER CONDITIONS?
                d_err_fake = self.criterion(d_out_fake, fake_labels)
                d_err_fake.backward()
                D_G_z1 = d_out_fake.mean().item()
                err_D = d_err_real + d_err_fake
                # Update weights of D
                self.discriminator.optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.model.zero_grad()
                d_out_fake = self.discriminator.model(fake_generated, conditions_batch).view(-1)
                err_G = self.criterion(d_out_fake, real_labels)
                err_G.backward()
                D_G_z2 = d_out_fake.mean().item()
                self.generator.optimizer.step()

                if self.iter_no % 100 == 0:
                    # padded_epoch = str(epoch).ljust(len(str(params.epochs)))
                    # padded_batch_idx = str(batch_idx).ljust(len(str(len(dataloader))))
                    progress_str = "Loss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f" % (
                        err_D.item(),
                        err_G.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                    progress.set_description(progress_str)

                # Save Losses for plotting later
                self.gen_losses.append(err_G.item())
                self.dis_losses.append(err_D.item())

                self.iter_no += 1

            for callback_option in self.callback_options:
                with torch.no_grad():
                    (epoch_predicate, callbacks) = callback_option
                    if epoch_predicate(epoch, self) is True:
                        for callback in callbacks:
                            callback(epoch, self)

        print("End training\n--------------------------------------------")


if __name__ == "__main__":
    data_importer = DWDWeatherDataImporter()
    data_importer.initialize()
    data_holder = DataHolder(
        data_importer.data.values.astype(np.float32),
        np.array(dates_to_conditional_vectors(*data_importer.get_datetime_values())),
    )
    epochs = 10
    noise_vector_size = 50
    # sequence_length = 24
    batch_size = 24  # *7
    features = data_holder.get_feature_size()
    G_net = BasicGenerator(input_size=noise_vector_size + 14, out_size=features, hidden_layers=[200])
    G_optim = torch.optim.Adam(G_net.parameters())
    G_sched = StepLR(G_optim, step_size=30, gamma=0.1)
    G = TrainModel(G_net, G_optim, G_sched)

    D_net = BasicDiscriminator(input_size=features + 14, out_size=1, hidden_layers=[100, 50, 20])
    D_optim = torch.optim.Adam(D_net.parameters())
    D_sched = StepLR(D_optim, step_size=30, gamma=0.1)
    D = TrainModel(D_net, D_optim, D_sched)

    trainer = CGANTrainer(G, D, data_holder, noise_vector_size, batch_size, "cpu")
    trainer.train(epochs)

    path = get_root_project_path().joinpath("runs").joinpath("model-test").absolute()
    trainer.save_model(path)
