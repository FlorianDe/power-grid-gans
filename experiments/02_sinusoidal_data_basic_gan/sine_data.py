from dataclasses import dataclass, field
import torch
import math


@dataclass
class SineGenerationParameters:
    sequence_len: int
    amplitudes: list[float] = field(default_factory=lambda: [1])
    times: int = 1
    noise_scale: float = 0.05

    def __iter__(self):
        return iter((self.sequence_len, self.amplitudes, self.times, self.noise_scale))


def generate_sine_features(
    params: SineGenerationParameters, seed: int = 42, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Returns a multi dimensional sine wave feature of shape [times, sequence_len, features]
    """
    sequence_len, amplitudes, times, noise_scale = params
    torch.manual_seed(seed)

    features = len(amplitudes)
    a = torch.tensor(amplitudes, device=device).view(1, features)
    # x = torch.linspace(0, sequence_len, sequence_len).view(sequence_len, 1).repeat(times, 1)
    x = torch.arange(sequence_len, device=device).view(sequence_len, 1).repeat(times, 1)
    sine = torch.sin((2 * math.pi / sequence_len) * x)
    scaled_sines = (sine * a).view(times, sequence_len, features)
    # noises = noise_scale * (2 * torch.rand(scaled_sines.shape) - torch.ones(scaled_sines.shape))   # scaled Uniform dist noise
    noises = noise_scale * torch.randn(scaled_sines.shape, device=device)  # scaled Normal dist noise

    return scaled_sines + noises


if __name__ == "__main__":
    s = generate_sine_features(
        SineGenerationParameters(sequence_len=24, times=5, amplitudes=[1, 0.5], noise_scale=0.01),
        # device=torch.device("cuda"),
    )
    print(f"{s.size()}")
