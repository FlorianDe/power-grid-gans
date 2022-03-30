from dataclasses import dataclass
from typing import Callable, Optional, Union
import torch

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    LambdaLR,
    CyclicLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from src.net import CustomModule


@dataclass
class TrainParameters:
    latent_vector_size: int = 100  # original paper \cite?
    sequence_len: int = 24
    batch_size: int = 8
    features_len: Optional[int] = None  # if not provided extract
    device: torch.device = torch.device("cpu")


@dataclass
class ConditionalTrainParameters(TrainParameters):
    embedding_dim: int = 100


CurrentBatchSize = int
BatchReshaper = Callable[[torch.Tensor, CurrentBatchSize, TrainParameters], torch.Tensor]
NoiseGenerator = Callable[[CurrentBatchSize, TrainParameters], torch.Tensor]
# Should be CGANTrainer too much of a hastle to create abstract classes etc rn
TrainingEpoch = int
Trainer = any
EpochPredicate = Callable[[TrainingEpoch, Trainer], bool]
TrainerCallback = Callable[[TrainingEpoch, Trainer], any]


@dataclass
class TrainModel:
    model: CustomModule
    optimizer: Optimizer
    scheduler: Optional[
        Union[
            StepLR,
            MultiStepLR,
            LambdaLR,
            CyclicLR,
            ExponentialLR,
            CosineAnnealingLR,
            CosineAnnealingWarmRestarts,
            ReduceLROnPlateau,
        ]
    ]
