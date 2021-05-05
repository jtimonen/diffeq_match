def plot_disc_2d(disc, z_fake, z_data, only_contour=False, save_dir=".", **kwargs):
    """Visualize discriminator output."""
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    acc_str = "{:.5f}".format(acc)
    title = "epoch " + epoch_str + ", loss = " + loss_str + ", acc = " + acc_str
    fn_pre = "c_" if only_contour else "d_"
    fn = fn_pre + epoch_str + ".png"
    S = 30
    u = create_grid_around(z_data, S)
    val = disc.classify_numpy(u)
    X = np.reshape(u[:, 0], (S, S))
    Y = np.reshape(u[:, 1], (S, S))
    Z = np.reshape(val, (S, S))

    plt.figure(figsize=(7.0, 6.5))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    if not only_contour:
        plt.scatter(z_data[:, 0], z_data[:, 1], c="red", marker=".", alpha=0.3)
    if z_fake is not None:
        plt.scatter(z_fake[:, 0], z_fake[:, 1], c="orange", marker=".", alpha=0.3)

    plt.title(title)
    x_min = np.min(z_data) * 1.25
    x_max = np.max(z_data) * 1.25
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    draw_plot(fn, save_dir, **kwargs)
