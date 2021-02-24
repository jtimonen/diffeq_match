from matplotlib import pyplot as plt
import os
import numpy as np
from .utils import create_grid_around
import torch


def draw_plot(save_name, save_dir=".", **kwargs):
    """Function to be used always when a plot is to be shown or saved."""
    if save_name is None:
        plt.show()
    else:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, **kwargs)
        plt.close()


def plot_state_2d(model, z_samp, z_data, idx_epoch, loss, save_dir=".", **kwargs):
    u = create_grid_around(z_data, 20)
    v = model.f_numpy(u)
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    title = "drift, epc = " + epoch_str + ", valid_loss = " + loss_str
    fn = "fig_" + epoch_str + ".png"

    # Create plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    axs[0, 0].quiver(u[:, 0], u[:, 1], v[:, 0], v[:, 1], alpha=0.5)
    axs[0, 0].scatter(z_data[:, 0], z_data[:, 1], alpha=0.7)
    axs[0, 0].scatter(z_samp[:, 0], z_samp[:, 1], marker="x", alpha=0.7)
    axs[0, 0].set_title(title)

    x_min = np.min(z_data) * 1.2
    x_max = np.max(z_data) * 1.2
    axs[0, 0].set_xlim(x_min, x_max)
    axs[0, 0].set_ylim(x_min, x_max)
    axs[0, 1].set_xlim(x_min, x_max)
    axs[0, 1].set_ylim(x_min, x_max)

    S = 40
    u = create_grid_around(z_data, S)
    ut = torch.from_numpy(u).float()
    z_samp = torch.from_numpy(z_samp).float()
    v1 = model.kde(ut, z_samp)
    v1 = v1.cpu().detach().numpy()
    v2 = model.g_numpy(u)

    X = np.reshape(u[:, 0], (S, S))
    Y = np.reshape(u[:, 1], (S, S))
    Z1 = np.reshape(v1, (S, S))
    Z2 = np.reshape(v2, (S, S))

    axs[1, 0].contourf(X, Y, Z1)
    axs[0, 1].contourf(X, Y, Z2)
    axs[1, 0].set_xlim(x_min, x_max)
    axs[1, 0].set_ylim(x_min, x_max)
    axs[1, 1].set_xlim(x_min, x_max)
    axs[1, 1].set_ylim(x_min, x_max)
    axs[0, 1].set_title("diffusion")
    axs[1, 0].set_title("log(KDE) forward")
    draw_plot(fn, save_dir, **kwargs)


def plot_kde(
    kde, z_data, sigma: float = 0.02, plot_data=True, fn=None, save_dir=".", **kwargs
):
    """Visualize 2d KDE."""
    S = 60
    z_data = z_data.cpu().detach().numpy()
    u = create_grid_around(z_data, S)
    ut = torch.from_numpy(u).float()
    z_data = torch.from_numpy(z_data).float()
    val = kde(ut, z_data, sigma)
    val = val.cpu().detach().numpy()
    z_data = z_data.cpu().detach().numpy()

    X = np.reshape(u[:, 0], (S, S))
    Y = np.reshape(u[:, 1], (S, S))
    Z = np.reshape(val, (S, S))

    plt.figure(figsize=(7.0, 6.5))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    if plot_data:
        plt.scatter(z_data[:, 0], z_data[:, 1], c="k", marker=".", alpha=0.3)

    plt.title("log KDE")
    x_min = np.min(z_data) * 1.25
    x_max = np.max(z_data) * 1.25
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    draw_plot(fn, save_dir, **kwargs)


def plot_state_3d(model, z_samp, z_data, idx_epoch, loss, save_dir=".", **kwargs):
    fig = plt.figure(figsize=(13, 13))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.scatter(z_data[:, 0], z_data[:, 1], z_data[:, 2], alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)

    ax2.scatter(z_samp[:, 0], z_samp[:, 1], z_samp[:, 2], color="orange", alpha=0.3)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)

    ax3.scatter(z_data[:, 0], z_data[:, 1], alpha=0.7)
    ax3.scatter(z_samp[:, 0], z_samp[:, 1], marker="x", alpha=0.7)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)

    ax4.scatter(z_data[:, 1], z_data[:, 2], alpha=0.7)
    ax4.scatter(z_samp[:, 1], z_samp[:, 2], marker="x", alpha=0.7)
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)

    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    title = "epoch " + epoch_str + ", valid_loss = " + loss_str
    fn = "fig_" + epoch_str + ".png"
    ax1.set_title(title)
    ax2.set_title("forward")
    ax3.set_title("dim 1 vs. dim 2")
    ax4.set_title("dim 2 vs. dim 3")
    draw_plot(fn, save_dir, **kwargs)


def plot_disc():
    return NotImplementedError


def plot_sde_2d(z_data, z_traj, idx_epoch, save_dir=".", **kwargs):
    plt.figure(figsize=(8, 8))
    plt.scatter(z_data[:, 0], z_data[:, 1], color="black", alpha=0.1)
    J = z_traj.shape[1]
    for j in range(J):
        zj = z_traj[:, j, :]
        plt.plot(zj[:, 0], zj[:, 1], color="red", alpha=0.7)
    epoch_str = "{0:04}".format(idx_epoch)
    title = "sde trajectories, epoch = " + epoch_str
    fn = "sde_" + epoch_str + ".png"
    plt.title(title)
    draw_plot(fn, save_dir, **kwargs)


def plot_sde_3d(z_data, z_traj, idx_epoch, save_dir=".", **kwargs):
    fig = plt.figure(figsize=(13, 13))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    J = z_traj.shape[1]

    ax1.scatter(z_data[:, 0], z_data[:, 1], z_data[:, 2], color="black", alpha=0.3)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(-2, 2)

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-2, 2)

    ax3.scatter(z_data[:, 0], z_data[:, 1], color="black", alpha=0.7)
    ax4.scatter(z_data[:, 1], z_data[:, 2], color="black", alpha=0.7)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    for j in range(J):
        ax2.plot(
            z_traj[:, j, 0], z_traj[:, j, 1], z_traj[:, j, 2], color="red", alpha=0.7
        )
        ax3.plot(z_traj[:, j, 0], z_traj[:, j, 1], color="red", alpha=0.7)
        ax4.plot(z_traj[:, j, 1], z_traj[:, j, 2], color="red", alpha=0.7)

    epoch_str = "{0:04}".format(idx_epoch)
    title = "epoch " + epoch_str
    fn = "sde_" + epoch_str + ".png"
    ax1.set_title(title)
    ax2.set_title("forward")
    ax3.set_title("dim 1 vs. dim 2")
    ax4.set_title("dim 2 vs. dim 3")
    draw_plot(fn, save_dir, **kwargs)
