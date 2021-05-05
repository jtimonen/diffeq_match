from sklearn.datasets import make_moons
from dem.modules.discriminator import KdeDiscriminator, NeuralDiscriminator
import numpy as np


def test_neural_discriminator():
    x, labels = make_moons(100)
    disc = NeuralDiscriminator(D=2, n_hidden=32)
    acc = disc.accuracy(x, y_true=labels)
    cm = disc.confusion_matrix(x, y_true=labels)
    assert acc >= 0, "accuracy must be positive"
    assert cm.shape == (2, 2), "shape of confusion matrix should be (2, 2)"


def test_kde_discriminator():
    x, labels = make_moons(100)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    x0 = x[idx0, :]
    x1 = x[idx1, :]
    disc = KdeDiscriminator(D=2, bw_init=0.5, trainable=True)
    disc.set_data_numpy(x0, x1)
    acc = disc.accuracy(x, y_true=labels)
    cm = disc.confusion_matrix(x, y_true=labels)
    assert acc >= 0.9, "accuracy should be around 0.94"
    assert cm.shape == (2, 2), "shape of confusion matrix should be (2, 2)"
