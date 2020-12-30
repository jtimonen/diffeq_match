from matplotlib import pyplot as plt
import os
import numpy as np


def draw_plot(save_name, save_dir=".", **kwargs):
    """Function to be used always when a plot is to be shown or saved."""
    if save_name is None:
        plt.show()
    else:
        save_path = os.path.join(save_dir, save_name)
        # log_info("Saving figure to " + save_path)
        plt.savefig(save_path, **kwargs)
        plt.close()


def plot_match(
    z_data,
    z0,
    title,
    u=None,
    v=None,
    z_traj=None,
    save_name=None,
    save_dir=".",
    **kwargs
):
    plt.figure(figsize=(7, 7))
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
    draw_plot(save_name, save_dir, **kwargs)
