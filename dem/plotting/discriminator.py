from matplotlib import pyplot as plt
import numpy as np
from hdviz import create_grid_around, draw_plot, create_plotter


def classify_at_2d_grid_around(disc, x: np.ndarray, M: int = 30):
    u = create_grid_around(x, M)
    label, val = disc.classify(u)
    X = np.reshape(u[:, 0], (M, M))
    Y = np.reshape(u[:, 1], (M, M))
    Z = np.reshape(val, (M, M))
    Z_label = np.reshape(label, (M, M))
    return X, Y, Z, Z_label


def plot_disc_2d(
    disc,
    x: np.ndarray,
    true_labels,
    prob=True,
    cm=None,
    scatter_kwargs=None,
    scatter_colors=None,
    scatter_alpha=0.6,
    grid_size: int = 60,
    save_name=None,
    save_dir=".",
    title="",
    points=True,
    contour=True,
    gan_mode=False,
    figsize=(7.0, 5.5),
    **kwargs
):
    """Visualize discriminator output."""
    if disc.is_critic:
        bar_label = "Critic output"
        if not prob:
            raise RuntimeError("prob should be true when plotting critic!")
    else:
        bar_label = "Discriminator output"
    if cm is None:
        cm = plt.cm.RdBu
    X, Y, Z, Z_label = classify_at_2d_grid_around(disc, x, grid_size)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if not prob:
        Z = Z_label
    if scatter_kwargs is None:
        scatter_kwargs = dict(edgecolor="k")
    if scatter_colors is None:
        scatter_colors = ["#FF0000", "#0000FF"]
    if contour:
        if disc.is_critic:
            levels = None
        else:
            levels = [h * 0.05 for h in range(0, 21)]
        cs = ax.contourf(X, Y, Z, cmap=cm, alpha=0.75, levels=levels)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label(bar_label, rotation=270)
    pp = create_plotter(2)
    labels = ["generated", "data"] if gan_mode else None
    pp.add_pointsets(
        x=x,
        labels=true_labels,
        label_names=labels,
        label_name_prefix="class",
        label_colors=scatter_colors,
        alpha=scatter_alpha,
    )
    pp.scatter_kwargs = scatter_kwargs
    if points:
        pp.plot(ax=ax, title=title)
    draw_plot(save_name, save_dir, **kwargs)
