from sklearn.datasets import make_moons
import dem


def test_neural_discriminator():
    x, labels = make_moons(200, noise=0.3)
    disc = dem.NeuralDiscriminator(D=2)
    y_pred, y_prob = disc.classify(x)
    acc = dem.accuracy(y_pred=y_pred, y_true=labels)
    assert acc >= 0, "accuracy should be non-negative"


def test_kde_discriminator():
    bw = 0.2
    x, labels = make_moons(100, noise=0.2)
    x0, x1 = dem.split_by_labels(x, labels)
    disc = dem.KdeDiscriminator(D=2, bw_init=bw, trainable=False)
    disc.update_numpy(x0, x1)
    y_pred, y_prob = disc.classify(x)
    acc = dem.accuracy(y_pred=y_pred, y_true=labels)
    assert acc >= 0.0, "accuracy should be non-negative"
    title = "bandwidth = " + str(bw)
    dem.plot_disc_2d(
        disc,
        x,
        labels,
        save_dir="tests/out",
        save_name="kde_2d.png",
        title=title,
    )


def test_kde_discriminator_training():
    n_epochs = 51
    x, labels = make_moons(500, noise=0.2)
    x0, _ = dem.split_by_labels(x, labels)
    disc = dem.KdeDiscriminator(D=2, bw_init=1.0, trainable=True)
    out, _ = dem.train_occ(
        disc, x0, plot_freq=10, n_epochs=n_epochs, outdir="tests/out/kde"
    )
    ea = out.read_logged_events()
    df = out.read_logged_scalar(name="valid_accuracy")
    assert df.shape[0] == n_epochs, "logs must have a row for each epoch!"


def test_nn_discriminator_training():
    n_epochs = 51
    x, labels = make_moons(500, noise=0.2)
    _, x1 = dem.split_by_labels(x, labels)
    disc = dem.NeuralDiscriminator(D=2)
    out, _ = dem.train_occ(
        disc, x1, plot_freq=10, n_epochs=n_epochs, outdir="tests/out/nn"
    )
    ea = out.read_logged_events()
    df = out.read_logged_scalar(name="valid_accuracy")
    assert df.shape[0] == n_epochs, "logs must have a row for each epoch!"
