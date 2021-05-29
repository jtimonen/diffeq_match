import numpy as np
import torch.nn as nn
from dem.modules.networks import MultiLayerNet
from dem.modules import (
    GenerativeModel,
    PriorInfo,
    DynamicModel,
    KdeDiscriminator,
    NeuralDiscriminator,
    Stage,
)


def create_dynamics(
    D: int,
    stochastic: bool = False,
    net_f=None,
    n_hidden: int = 48,
    n_hidden_layers: int = 2,
    activation=None,
):
    """Create a vector field that has methods."""
    if net_f is None:
        if activation is None:
            activation = nn.Tanh()
        net_f = MultiLayerNet(
            n_input=D,
            n_output=D,
            n_hidden=n_hidden,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
        )
    return DynamicModel(net_f, stochastic)


def create_model(init: np.ndarray, stages=None, net_f=None, **net_f_kwargs):
    """Construct a model with some default settings.

    :param init: Initial points as rows of a numpy array.
    :param net_f: A neural network module specifying the drift of the vector field.
    :param stages: A list of `Stage`s of the generative model.
    :param net_f_kwargs: keyword arguments to `create_dynamics`, only have effect if
    `net_f` is None
    """
    if init is None:
        init = np.array([[0.0, 0.0]])
    if stages is None:
        stages = [Stage()]
    D = init.shape[1]
    is_stochastic = any([s.sde for s in stages])
    dyn = create_dynamics(D, is_stochastic, net_f, **net_f_kwargs)
    prior_info = PriorInfo(init=init)
    return GenerativeModel(dynamics=dyn, prior_info=prior_info, stages=stages)


def create_discriminator(
    D: int, kde=False, fixed_kde=False, critic: bool = False, **nn_kwargs
):
    """Construct a discriminator with some default settings."""
    if fixed_kde:
        disc = KdeDiscriminator(D=D, trainable=False)
    elif kde:
        disc = KdeDiscriminator(D=D, trainable=True)
    else:
        disc = NeuralDiscriminator(D=D, critic=critic, **nn_kwargs)
    return disc
