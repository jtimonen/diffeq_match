from sklearn.datasets import make_moons
import numpy as np
import dem


def test_neural_discriminator():
    x, labels = make_moons(200, noise=0.3)
    disc = dem.NeuralDiscriminator(D=2)
    y_pred, y_prob = disc.classify(x)
    acc = dem.accuracy(y_pred=y_pred, y_true=labels)
    assert acc >= 0, "accuracy should be non-negative"


def test_kde_discriminator():
    x, labels = make_moons(100, noise=0.2)
    x0, x1 = dem.split_by_labels(x, labels)
    disc = dem.KdeDiscriminator(D=2, bw_init=1.0, trainable=True)
    disc.set_data_numpy(x0, x1)
    y_pred, y_prob = disc.classify(x)
    acc = dem.accuracy(y_pred=y_pred, y_true=labels)
    assert acc >= 0.0, "accuracy should be non-negative"
    dem.plot_disc_2d(disc, x, labels, save_name="test_plot.png")


def test_kde_discriminator_training():
    x, labels = make_moons(500, noise=0.2)
    x0, _ = dem.split_by_labels(x, labels)
    disc = dem.KdeDiscriminator(D=2, bw_init=1.0, trainable=True)
    out = dem.train_occ(disc, x0, plot_freq=100)
