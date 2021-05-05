from matplotlib import pyplot as plt
import numpy as np
from hdviz import create_grid_around, draw_plot, Plotter2d


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
    **kwargs
):
    """Visualize discriminator output."""
    if cm is None:
        cm = plt.cm.RdBu
    X, Y, Z, Z_label = classify_at_2d_grid_around(disc, x, grid_size)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.5))
    if not prob:
        Z = Z_label
    if scatter_kwargs is None:
        scatter_kwargs = dict(edgecolor="k")
    if scatter_colors is None:
        scatter_colors = ["#FF0000", "#0000FF"]
    ax.contourf(X, Y, Z, cmap=cm, alpha=0.8)
    pp = Plotter2d()
    pp.add_pointsets(
        x=x,
        categories=true_labels,
        categ_prefix="class",
        colors=scatter_colors,
        alpha=scatter_alpha,
    )
    pp.scatter_kwargs = scatter_kwargs
    pp.plot(ax=ax)
    draw_plot(save_name, save_dir, **kwargs)


# draw_plot(fn, save_dir, **kwargs)
