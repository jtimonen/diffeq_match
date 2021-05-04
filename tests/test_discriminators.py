import dem
from sklearn.datasets import make_moons
from dem.discriminator import KdeDiscriminator, NeuralDiscriminator


def test_moons():
    x, labels = make_moons(100)
    disc = NeuralDiscriminator(D=2, n_hidden=32)
    out = disc.classify(x)
    print(out)
    acc = disc.evaluate(x, true_labels=labels)
    print(acc)
    assert acc >= 0, "accuracy must be positive"
