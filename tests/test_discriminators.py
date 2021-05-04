from sklearn.datasets import make_moons
from dem.discriminator import KdeDiscriminator, NeuralDiscriminator


def test_moons():
    x, labels = make_moons(100)
    disc = NeuralDiscriminator(D=2, n_hidden=32)
    acc = disc.accuracy(x, y_true=labels)
    cm = disc.confusion_matrix(x, y_true=labels)
    print(acc)
    assert acc >= 0, "accuracy must be positive"
    assert cm.shape == (2, 2), "shape of confusion matrix should be (2, 2)"
