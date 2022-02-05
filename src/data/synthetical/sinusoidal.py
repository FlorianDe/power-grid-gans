import math
from typing import Optional, Callable

import numpy as np
import numpy.typing as npt
from attr import dataclass
from matplotlib import pyplot as plt
from scipy import optimize


@dataclass
class TrigFuncParameters:
    frequency: Optional[float] = None
    phase: Optional[float] = None
    amplitude: Optional[float] = None


def __calculate_trigonometric_input(steps: npt.ArrayLike, frequency: float, phase: float) -> npt.ArrayLike:
    return steps * frequency + phase


def generate_sinusoidal_time_series(sample_count: int,
                                    series_length: int,
                                    dimensions: int = 1,
                                    seed: Optional[int] = None,
                                    func: Callable[[npt.ArrayLike], npt.ArrayLike] = np.sin,
                                    norm_max_value: float = None,
                                    normalize: bool = True,
                                    trig_parameters: TrigFuncParameters = TrigFuncParameters()
                                    ):
    """Sine data generation.

    Args:
      - sample_count: the number of samples
      - series_length: length of the time-series
      - dimensions: feature dimensions for each time-series
      - seed: random seed, to reproduce same results
      - func: a periodic function f, which is of the form: A*f(step*frequency+phase)
      - norm_max_value: the maximum of the custom function, needed for normalization for the basic trigonometric functions
        e.g. sin and cos its the amplitude itself and must not be specified explicitly
      - normalize: whether the generated data should be normalized to [0,1]

    Returns:
      - data: generated and normalized
    """
    if seed:
        np.random.seed(seed)

    # Initialize the output
    data: list[npt.ArrayLike] = list()  # list[ndarray(series_length, dimensions)]

    # Generate sinusodial data
    for _ in range(sample_count):
        # Initialize each time-series and the parameters used in each dimension
        temp_series_dimensions: list[npt.ArrayLike] = list()
        params_used_in_dimension: list[TrigFuncParameters] = list()

        # For each dimension create a time series of length series_length with random uniform trigonometry parameters
        for d in range(dimensions):
            # Randomly drawn frequency and phase
            trig_params = TrigFuncParameters(
                frequency=trig_parameters.frequency if trig_parameters.frequency is not None else np.random.uniform(0, 0.1),
                phase=trig_parameters.phase if trig_parameters.phase is not None else np.random.uniform(0, 0.1),
                amplitude=trig_parameters.amplitude if trig_parameters.amplitude is not None else  np.random.uniform(0, 1)
                # We can just use 1 for now since we are normalizing nonetheless
            )

            # Calculate time series based on the trigonometry function provided with the sampled trigonometry parameters
            steps = np.arange(series_length)
            temp_data = trig_params.amplitude * func(__calculate_trigonometric_input(steps, trig_params.frequency, trig_params.phase))
            temp_series_dimensions.append(temp_data)
            params_used_in_dimension.append(trig_params)

        # Transpose the temporary time series with its dimensions so we can append it to the final result list
        arr_temp = np.asarray(temp_series_dimensions)
        t_temp = np.transpose(arr_temp)

        # If normalization is turned on, normalize to [0, max(trig_func)/norm_max] for each dimension,
        # for sinusoidal functions sin/cos this leads to max(trig_func) == amplitude therefore [0,amplitude/amplitude] => [0,1]
        if normalize:
            norm_max: Optional[npt.ArrayLike] = norm_max_value
            if func is np.sin or func is np.cos:
                norm_max = np.array([dim_params.amplitude for dim_params in params_used_in_dimension])

            if norm_max is None:
                approx_max_input = optimize.fmin(lambda x: -func(x), np.pi / 2, xtol=1e-10, ftol=1e-10)
                # epsilon to determine that the specified function is potentially not a periodic function without poles or it is a monotonic function
                norm_max = func(approx_max_input)
                if abs(norm_max) >= 1e10 or norm_max is None:
                    raise ValueError(
                        "Called the generation function with normalization but you haven't provided a norm_max for your custom sin-like function and we couldn't derive a valid max norm!")

            t_temp = (t_temp + norm_max) * (norm_max / 2)

        # Stack the generated data
        data.append(t_temp)

    return data


if __name__ == '__main__':
    seq_len = 24  # 365 * 24
    samples = generate_sinusoidal_time_series(
        sample_count=1,
        series_length=seq_len,
        dimensions=3,
        func=np.sin,
        normalize=False,
        trig_parameters=TrigFuncParameters(2*math.pi/24, 0)
    )  # np.sin, np.cos, np.tan, signal.sawtooth, signal.square
    for sample_idx in range(len(samples)):
        feature_series = np.transpose(samples[sample_idx])
        for feature_idx in range(len(feature_series)):
            plt.plot(feature_series[feature_idx])
        plt.show()
