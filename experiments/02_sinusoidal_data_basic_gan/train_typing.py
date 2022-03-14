from dataclasses import dataclass
from typing import Callable, Union
import torch


@dataclass
class AdamParameters:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class TrainParameters:
    epochs: int
    latent_vector_size: int = 100  # original paper \cite?
    sequence_len: int = 24
    batch_size: int = 8
    optimizer_parameters: AdamParameters = AdamParameters()  # could be Union
    device: torch.device = torch.device("cpu")


@dataclass
class ConditionalTrainParameters(TrainParameters):
    embedding_dim: int = 100


BatchReshaper = (Callable[[torch.Tensor], torch.Tensor],)
NoiseGenerator = (Callable[[int, TrainParameters, int], torch.Tensor],)
