from matplotlib import pyplot as plt
from hdviz import draw_plot


def plot_scalar_progress(df, name, ax):
    step = df["step"]
    y = df["value"]
    ax.plot(step, y, color="k")
    ax.scatter(step, y, color="k", s=3)
    ax.set_ylabel(name)
    ax.set_xlabel("Step")


def plot_gan_progress(g_loss, d_loss, acc, save_name, save_dir, **save_kwargs):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 7))
    plot_scalar_progress(g_loss, "generator loss", axs[0])
    plot_scalar_progress(d_loss, "discriminator loss", axs[1])
    plot_scalar_progress(acc, "discriminator accuracy", axs[2])
    plt.tight_layout()
    draw_plot(save_name, save_dir, **save_kwargs)
