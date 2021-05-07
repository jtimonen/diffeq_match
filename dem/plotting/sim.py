import numpy as np
import matplotlib.pyplot as plt
from hdviz import draw_plot


def plot_sim(z: np.ndarray, c=None, save_name=None, scatter_kwargs=None):
    """Plot latent simulation."""
    if scatter_kwargs is None:
        scatter_kwargs = dict()
    D = z.shape[1]
    if D == 2:
        plot_sim_2d(z, c, save_name, scatter_kwargs)
    else:
        plot_sim_3d(z, c, save_name, scatter_kwargs)


def plot_sim_2d(z: np.ndarray, c, save_name, scatter_kwargs):
    """Plot latent simulation."""

    if c is None:
        plt.scatter(z[:, 0], z[:, 1], **scatter_kwargs)
    else:
        plt.scatter(z[:, 0], z[:, 1], c=c, **scatter_kwargs)
        plt.colorbar()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    draw_plot(save_name)


def plot_sim_3d(z: np.ndarray, c, save_name, scatter_kwargs):
    """Plot latent simulation."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if c is None:
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], **scatter_kwargs)
    else:
        ax.scatter(z[:, 0], z[:, 1], z[:, 2], c=c, **scatter_kwargs)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    draw_plot(save_name)
