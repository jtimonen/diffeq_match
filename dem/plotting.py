from matplotlib import pyplot as plt
import os
import numpy as np
from .utils import create_grid_around


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


def plot_match(
    model, disc, z_back, z_forw, z_data, idx_epoch, loss, save_dir=".", **kwargs
):
    u = create_grid_around(z_data, 16)
    v = model.defunc_numpy(u)
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    title = "epoch " + epoch_str + ", valid_loss = " + loss_str
    fn = "fig_" + epoch_str + ".png"

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.quiver(u[:, 0], u[:, 1], v[:, 0], v[:, 1], alpha=0.5)
    plt.scatter(z_data[:, 0], z_data[:, 1], alpha=0.7)
    plt.scatter(z_back[:, 0], z_back[:, 1], marker="x", alpha=0.7)
    plt.scatter(z_forw[:, 0], z_forw[:, 1], marker="x", color="red", alpha=0.7)
    plt.title(title)
    x_min = np.min(z_data) * 1.5
    x_max = np.max(z_data) * 1.5
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
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
