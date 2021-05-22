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


def create_dynamics(D: int, n_hidden: int = 32, stochastic: bool = False, net_f=None):
    """Create a vector field that has methods."""
    if net_f is None:
        net_f = TanhNetTwoLayer(D, D, n_hidden)
    return DynamicModel(net_f, stochastic)


def create_model(init: np.ndarray, n_hidden: int = 32, stages=None, net_f=None):
    """Construct a model with some default settings.

    :param init: Initial points as rows of a numpy array.
    :param n_hidden: number of hidden nodes in net_f. Has no effect if net_f is not
    None.
    :param net_f: A neural network module specifying the drift of the vector field.
    :param stages: A list of `Stage`s of the generative model.
    """
    if init is None:
        init = np.array([[0.0, 0.0]])
    if stages is None:
        stages = [Stage()]
    D = init.shape[1]
    is_stochastic = any([s.sde for s in stages])
    dyn = create_dynamics(D, n_hidden, is_stochastic, net_f)
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
