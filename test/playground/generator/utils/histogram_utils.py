import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def generate_random_normal_distribution(
        sample_size=255,
        mu=0,
        sigma=0.1,
        random_seed: int = None
):
    if random_seed:
        np.random.seed(random_seed)
    values = np.random.normal(mu, sigma, sample_size)
    return values


def plot_distribution(values, bins: int = None):
    mu, std = norm.fit(values)
    plt.hist(values, bins, density=True, alpha=0.6, color='g')
    plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.show()


def generate_noisy_normal_distribution(
        sample_size,
        mu=0,
        sigma=0.1,
        random_seed: int = None,
        noise_scale=0.2
):
    if random_seed:
        np.random.seed(random_seed)
    x = np.linspace(-0.5, 0.5, sample_size)
    p = norm.pdf(x, mu, sigma)
    noise = np.random.normal(0, 1, sample_size) * noise_scale
    p = p + noise
    clipped = np.clip(p, 0, np.max(p))
    # plt.plot(x, clipped, 'k', linewidth=2)
    # plt.show()
    return clipped


if __name__ == "__main__":
    # vals = generate_random_normal_distribution(
    #     sample_size=samples * granularity,
    #     random_seed=1
    # )

    generate_noisy_normal_distribution(24)
