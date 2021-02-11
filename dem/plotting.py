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


def plot_match(model, z_back, z_forw, z_data, idx_epoch, loss, save_dir=".", **kwargs):
    u = create_grid_around(z_data, 20)
    v = model.defunc_numpy(u)
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    title = "epoch " + epoch_str + ", valid_loss = " + loss_str
    fn = "fig_" + epoch_str + ".png"

    # Create plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 14))
    axs[0, 0].quiver(u[:, 0], u[:, 1], v[:, 0], v[:, 1], alpha=0.5)
    axs[0, 0].scatter(z_data[:, 0], z_data[:, 1], alpha=0.7)
    axs[0, 0].scatter(z_back[:, 0], z_back[:, 1], marker="x", alpha=0.7)
    axs[0, 0].set_title(title)

    axs[0, 1].quiver(u[:, 0], u[:, 1], v[:, 0], v[:, 1], alpha=0.5)
    axs[0, 1].scatter(z_data[:, 0], z_data[:, 1], alpha=0.7)
    axs[0, 1].scatter(z_forw[:, 0], z_forw[:, 1], marker="x", color="red", alpha=0.7)
    axs[0, 1].set_title(title)

    x_min = np.min(z_data) * 1.2
    x_max = np.max(z_data) * 1.2
    axs[0, 0].set_xlim(x_min, x_max)
    axs[0, 0].set_ylim(x_min, x_max)
    axs[0, 1].set_xlim(x_min, x_max)
    axs[0, 1].set_ylim(x_min, x_max)

    S = 40
    u = create_grid_around(z_data, S)
    ut = torch.from_numpy(u).float()
    z_forw = torch.from_numpy(z_forw).float()
    z_back = torch.from_numpy(z_back).float()
    v1 = model.kde(ut, z_forw)
    v2 = model.kde(ut, z_back)
    v1 = v1.cpu().detach().numpy()
    v2 = v2.cpu().detach().numpy()

    X = np.reshape(u[:, 0], (S, S))
    Y = np.reshape(u[:, 1], (S, S))
    Z1 = np.reshape(v1, (S, S))
    Z2 = np.reshape(v2, (S, S))

    axs[1, 0].contourf(X, Y, Z2)
    axs[1, 1].contourf(X, Y, Z1)
    axs[1, 0].set_xlim(x_min, x_max)
    axs[1, 0].set_ylim(x_min, x_max)
    axs[1, 1].set_xlim(x_min, x_max)
    axs[1, 1].set_ylim(x_min, x_max)
    axs[1, 0].set_title("log(KDE) backward")
    axs[1, 1].set_title("log(KDE) forward")
    draw_plot(fn, save_dir, **kwargs)


def plot_disc(disc, z_fake, z_data, idx_epoch, loss, acc, save_dir=".", **kwargs):
    """Visualize discriminator output."""
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    acc_str = "{:.5f}".format(acc)
    title = "epoch " + epoch_str + ", loss = " + loss_str + ", acc = " + acc_str
    fn = "cls_" + epoch_str + ".png"
    S = 30
    u = create_grid_around(z_data, S)
    val = disc.classify_numpy(u)
    X = np.reshape(u[:, 0], (S, S))
    Y = np.reshape(u[:, 1], (S, S))
    Z = np.reshape(val, (S, S))

    plt.figure(figsize=(7.0, 6.5))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    if z_data is not None:
        plt.scatter(z_data[:, 0], z_data[:, 1], c="k", alpha=0.3)
    if z_fake is not None:
        plt.scatter(z_fake[:, 0], z_fake[:, 1], c="red", alpha=0.3)

    plt.title(title)
    x_min = np.min(z_data) * 1.25
    x_max = np.max(z_data) * 1.25
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
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
