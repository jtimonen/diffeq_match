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


def plot_match(z_data, zf, mmd: float, save_name=None, save_dir=".", **kwargs):
    plt.scatter(z_data[:, 0], z_data[:, 1])
    plt.scatter(zf[:, 0], zf[:, 1])
    plt.title("loss = " + str(mmd))
    plt.legend(["data", "matched"])
    xmin = np.min(z_data) * 1.2
    xmax = np.max(z_data) * 1.2
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, xmax)
    draw_plot(save_name, save_dir, **kwargs)
