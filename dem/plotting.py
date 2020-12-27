from matplotlib import pyplot as plt

def draw_plot(save_name, **kwargs):
    """Function to be used always when a plot is to be shown or saved."""
    if save_name is None:
        plt.show()
    else:
        save_path = os.path.join(get_outdir(), save_name)
        #log_info("Saving figure to " + save_path)
        plt.savefig(save_path, **kwargs)
        plt.close()
