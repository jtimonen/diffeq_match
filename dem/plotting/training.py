from matplotlib import pyplot as plt
from hdviz import draw_plot


def plot_scalar_progress(df, name, ax):
    step = df["step"]
    y = df["value"]
    ax.plot(step, y, color="k")
    ax.scatter(step, y, color="k", s=3)
    ax.set_ylabel(name)
    ax.set_xlabel("Step")


def plot_gan_progress(g_loss, d_loss, acc, bw, save_name, save_dir, **save_kwargs):
    do_bw = bw is not None
    nrows = 4 if do_bw else 3
    h = nrows * 2.3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, h))
    plot_scalar_progress(g_loss, "generator loss", axs[0])
    plot_scalar_progress(d_loss, "discriminator loss", axs[1])
    plot_scalar_progress(acc, "discriminator accuracy", axs[2])
    if do_bw:
        plot_scalar_progress(bw, "KDE bandwidth", axs[3])
    plt.tight_layout()
    draw_plot(save_name, save_dir, **save_kwargs)
