from dem.modules.networks import TanhNetTwoLayer
from dem.modules import GenModel, NeuralDiscriminator, KdeDiscriminator


def create_model(D: int, n_hidden: int = 32, stochastic: bool = False):
    """Construct a model with some default settings."""
    net_f = TanhNetTwoLayer(D, D, n_hidden)
    model = GenModel(net_f, stochastic)
    return model


def create_discriminator(D: int, n_hidden: int = 64, kde=False, fixed_kde=False):
    """Construct a discriminator with some default settings."""
    net_f = TanhNetTwoLayer(D, D, n_hidden)
    if fixed_kde:
        disc = KdeDiscriminator(D=D, trainable=False)
    elif kde:
        disc = KdeDiscriminator(D=D)
    else:
        disc = NeuralDiscriminator(D=D, n_hidden=n_hidden)
    return disc
