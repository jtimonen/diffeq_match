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


def plot_match(model, disc, z_data, idx_epoch, loss, save_dir=".", **kwargs):
    n_timepoints = 30
    n_draws = 100
    z_traj = model.traj_numpy(n_draws, n_timepoints)
    u = create_grid_around(z_data, 16)
    v = model.defunc_numpy(u)
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    title = "epoch " + epoch_str + ", loss = " + loss_str
    fn = "fig_" + epoch_str + ".png"
    print("plotting " + fn)

    # Create plot
    plt.figure(figsize=(10, 10))
    if u is not None:
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
