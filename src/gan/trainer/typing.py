from dataclasses import dataclass
from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, CyclicLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

from src.net import CustomModule


@dataclass
class TrainModel:
    model: CustomModule
    optimizer: Optimizer
    scheduler: Optional[Union[StepLR, MultiStepLR, LambdaLR, CyclicLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau]]
