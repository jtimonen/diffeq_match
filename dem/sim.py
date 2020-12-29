import numpy as np
import matplotlib.pyplot as plt
from .plotting import draw_plot


def sim(idx: int = 1, N: int = 500, sigma: float = 0.1, **sim_kwargs):
    """Simulate data.

    :return: a numpy array with shape (2, N) and a numpy array with shape (N)
    """
    if idx == 1:
        return sim_line(N, sigma)
    elif idx == 2:
        return sim_spiral(N, sigma, **sim_kwargs)
    else:
        raise ValueError("idx must be 1 or 2!")


def sim_line(N: int, sigma: float):
    """A line."""
    t = np.linspace(0, 1, N)
    z1 = 2.0 * (t - 0.5)
    z2 = np.zeros(z1.shape)
    z = add_noise_and_stack(z1, z2, sigma)
    return z, t


def sim_spiral(N: int, sigma: float, omega: float = np.pi, decay: float = 0.0):
    """A line."""
    t = np.linspace(0, 1, N)
    theta = omega * t
    R = 1.0 * np.exp(-decay * t)
    z1 = R * np.cos(theta)
    z2 = R * np.sin(theta)
    z = add_noise_and_stack(z1, z2, sigma)
    return z, t


def add_noise_and_stack(z1, z2, sigma):
    z1 = z1 + sigma * np.random.normal(size=z1.shape)
    z2 = z2 + sigma * np.random.normal(size=z2.shape)
    z = np.vstack((z1, z2)).T
    return z


def plot_sim(z: np.ndarray, t=None, save_name=None, scatter_kwargs=None, **kwargs):
    """Plot latent simulation."""
    if scatter_kwargs is None:
        scatter_kwargs = dict()
    if t is None:
        plt.scatter(z[:, 0], z[:, 1], **scatter_kwargs)
    else:
        plt.scatter(z[:, 0], z[:, 1], c=t, **scatter_kwargs)
        plt.colorbar()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    draw_plot(save_name, **kwargs)
