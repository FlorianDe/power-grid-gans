import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from data.synthetical.sinusoidal import generate_sinusoidal_time_series


def test_generate_sinusoidal_time_series():
    seq_len = 1000  # 365 * 24
    samples = generate_sinusoidal_time_series(1, seq_len, 1, func=signal.sawtooth, normalize=False)  # np.sin, np.cos, np.tan, signal.sawtooth, signal.square
    for sample_idx in range(len(samples)):
        feature_series = np.transpose(samples[sample_idx])
        for feature_idx in range(len(feature_series)):
            plt.plot(feature_series[feature_idx])
        plt.show()
