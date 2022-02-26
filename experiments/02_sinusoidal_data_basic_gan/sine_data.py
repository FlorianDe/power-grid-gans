from dataclasses import dataclass, field
import torch
from torch import Tensor
import math


@dataclass
class SineGenerationParameters:
    sequence_len: int
    amplitudes: list[float] = field(default_factory=lambda: [1])
    times: int = 1
    noise_scale: float = 0.05

    def __iter__(self):
        return iter((self.sequence_len, self.amplitudes, self.times, self.noise_scale))


def generate_sine_features(params: SineGenerationParameters, seed: int = 42) -> tuple[Tensor, Tensor]:
    """
    Returns a multi dimensional sine wave feature of shape [times, sequence_len, features]
    """
    sequence_len, amplitudes, times, noise_scale = params
    torch.manual_seed(seed)

    features = len(amplitudes)
    a = torch.tensor(amplitudes).view(1, features)
    # x = torch.linspace(0, sequence_len, sequence_len).view(sequence_len, 1).repeat(times, 1)
    x = torch.arange(sequence_len).view(sequence_len, 1).repeat(times, 1)
    sine = torch.sin((2 * math.pi / sequence_len) * x)
    scaled_sines = (sine * a).view(times, sequence_len, features)
    # noises = noise_scale * (2 * torch.rand(scaled_sines.shape) - torch.ones(scaled_sines.shape))   # scaled Uniform dist noise
    noises = noise_scale * torch.randn(scaled_sines.shape)  # scaled Normal dist noise

    return scaled_sines + noises
