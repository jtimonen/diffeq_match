import numpy as np
import matplotlib.pyplot as plt
from .plotting import draw_plot


def latent_sim(idx: int = 1, N1: int = 300, N2: int = 200, sigma: float = 0.15):
    """Simulate latent data.

    :return: a numpy array with shape (2, N1+N2)
    :rtype: numpy.ndarray
    """
    if idx == 1:
        return latent_sim1(N1, N2, sigma)
    else:
        raise ValueError("idx must be 1!")


def latent_sim1(N1: int, N2: int, sigma: float):
    """Simulate latent data."""
    z1 = np.linspace(0, 2, N1)
    z2 = np.sin(z1)
    w1 = np.linspace(1, 2, N2)
    w2 = np.sin(3 * w1) + 0.65
    z1 = np.concatenate((z1, w1))
    z2 = np.concatenate((z2, w2))

    def normalize(x):
        return (x - np.mean(x)) / np.std(x)

    z1 = normalize(z1) + sigma * np.random.normal(size=z1.shape)
    z2 = normalize(z2) + sigma * np.random.normal(size=z2.shape)
    return np.vstack((z1, z2))


def plot_latent_sim(z: np.ndarray, save_name=None, scatter_kwargs=None, **kwargs):
    """Plot latent simulation."""
    if scatter_kwargs is None:
        scatter_kwargs = dict()
    plt.scatter(z[0, :], z[1, :], **scatter_kwargs)
    draw_plot(save_name, **kwargs)
