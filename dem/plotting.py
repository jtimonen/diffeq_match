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


def plot_match(model, disc, z0, z_data, idx_epoch, loss, save_dir=".", **kwargs):
    n_timepoints = 30
    z_traj = model.traj_numpy(z0, n_timepoints)
    u = create_grid_around(z_data, 16)
    v = model.defunc_numpy(u)
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    title = "epoch " + epoch_str + ", loss = " + loss_str
    fn = "fig_" + epoch_str + ".png"
    fn2 = "cls_" + epoch_str + ".png"
    print("plotting " + fn)

    # Create plot
    plt.figure(figsize=(8, 8))
    plt.quiver(u[:, 0], u[:, 1], v[:, 0], v[:, 1], alpha=0.5)
    plt.scatter(z_data[:, 0], z_data[:, 1], alpha=0.75)
    if z_traj is not None:
        L = z_traj.shape[1]
        for j in range(0, L):
            plt.plot(z_traj[:, j, 0], z_traj[:, j, 1], ".-", c="orange", alpha=0.7)
    plt.scatter(z_traj[0, :, 0], z_traj[0, :, 1], c="k", marker="x", alpha=0.7)
    plt.title(title)
    x_min = np.min(z_data) * 1.5
    x_max = np.max(z_data) * 1.5
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    draw_plot(fn, save_dir, **kwargs)

    if disc is not None:

        # Create second plot
        S = 40
        u = create_grid_around(z_data, S)
        val = disc.classify_numpy(u)
        X = np.reshape(u[:, 0], (S, S))
        Y = np.reshape(u[:, 1], (S, S))
        Z = np.reshape(val, (S, S))

        plt.figure(figsize=(8, 8))
        plt.contourf(X, Y, Z)
        plt.colorbar()
        plt.scatter(z_data[:, 0], z_data[:, 1], c="k", alpha=0.3)

        plt.title("Discriminator output")
        x_min = np.min(z_data) * 1.5
        x_max = np.max(z_data) * 1.5
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        draw_plot(fn2, save_dir, **kwargs)
