from dataclasses import dataclass
import torch


@dataclass
class TrainParameters:
    epochs: int
    latent_vector_size: int = 100  # original paper \cite?
    sequence_len: int = 24
    batch_size: int = 8
    device: torch.device = torch.device("cpu")

@dataclass
class ConditionalTrainParameters(TrainParameters):
    embedding_dim: int = 100

