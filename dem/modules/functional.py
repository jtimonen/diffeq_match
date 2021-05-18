import numpy as np
from dem.modules.networks import TanhNetTwoLayer
from dem.modules import (
    GenerativeModel,
    PriorInfo,
    DynamicModel,
    KdeDiscriminator,
    NeuralDiscriminator,
)


def create_dynamics(D: int, n_hidden: int = 32, stochastic: bool = False):
    """Create a vector field that has methods."""
    net_f = TanhNetTwoLayer(D, D, n_hidden)
    return DynamicModel(net_f, stochastic)


def create_model(x0: np.ndarray, n_hidden: int = 32, stochastic: bool = False):
    """Construct a model with some default settings."""
    D = x0.shape[1]
    dyn = create_dynamics(D, n_hidden=n_hidden, stochastic=stochastic)
    prior_info = PriorInfo(init=x0)
    return GenerativeModel(dynamics=dyn, prior_info=prior_info)


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
