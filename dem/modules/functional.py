import numpy as np
from dem.modules.networks import TanhNetTwoLayer
from dem.modules import (
    GenerativeModel,
    PriorInfo,
    DynamicModel,
    KdeDiscriminator,
    NeuralDiscriminator,
    Stage,
)


def create_dynamics(D: int, n_hidden: int = 32, stochastic: bool = False):
    """Create a vector field that has methods."""
    net_f = TanhNetTwoLayer(D, D, n_hidden)
    return DynamicModel(net_f, stochastic)


def create_model(init: np.ndarray, n_hidden: int = 32, stages=None):
    """Construct a model with some default settings."""
    if init is None:
        init = np.array([[0.0, 0.0]])
    if stages is None:
        stages = [Stage()]
    D = init.shape[1]
    is_stochastic = any([s.sde for s in stages])
    dyn = create_dynamics(D, n_hidden=n_hidden, stochastic=is_stochastic)
    prior_info = PriorInfo(init=init)
    return GenerativeModel(dynamics=dyn, prior_info=prior_info, stages=stages)


def create_discriminator(D: int, n_hidden: int = 64, kde=False, fixed_kde=False):
    """Construct a discriminator with some default settings."""
    if fixed_kde:
        disc = KdeDiscriminator(D=D, trainable=False)
    elif kde:
        disc = KdeDiscriminator(D=D)
    else:
        disc = NeuralDiscriminator(D=D, n_hidden=n_hidden)
    return disc
