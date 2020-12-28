from matplotlib import pyplot as plt
import os


def draw_plot(save_name, save_dir=".", **kwargs):
    """Function to be used always when a plot is to be shown or saved."""
    if save_name is None:
        plt.show()
    else:
        save_path = os.path.join(save_dir, save_name)
        # log_info("Saving figure to " + save_path)
        plt.savefig(save_path, **kwargs)
        plt.close()
