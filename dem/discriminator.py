import numpy as np
import torch
import torch.nn as nn
from .networks import LeakyReluNetTwoLayer
from .kde import KDE, ParamKDE


class Discriminator(nn.Module):
    """Abstract binary classifier class."""

    def __init__(self, D: int):
        super().__init__()
        self.D = D

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(torch.mean(x, dim=1))

    def forward_numpy(self, x: np.ndarray):
        x = torch.from_numpy(x).float()
        val = self(x)
        return val.detach().cpu().numpy()

    def classify(self, x: np.ndarray):
        val = self.forward_numpy(x)
        return val > 0.5


class NeuralDiscriminator(Discriminator):
    """Binary classifier using a neural network.

    :param D: dimension
    :param n_hidden: number of hidden nodes
    """

    def __init__(self, D: int, n_hidden: int = 64):
        super().__init__(D)
        self.net = LeakyReluNetTwoLayer(D, 1, n_hidden)

    def forward(self, z):
        z = self.net(z)
        return torch.sigmoid(z)


class KdeDiscriminator(Discriminator):
    """Binary classifier using kernel density estimation.

    :param D: dimension
    :param bw_init: initial bandwidth of kernel
    :param trainable: is the bandwidth a trainable parameter
    """

    def __init__(self, D: int, bw_init: float = 0.1, trainable: bool = False):
        super().__init__(D)
        self.trainable = trainable
        self.kde = ParamKDE(D, bw_init) if trainable else KDE(D)

    def forward(self, z):
        z = self.net(z)
        validity = torch.sigmoid(z)
        return validity

    def classify_numpy(self, z):
        z = torch.from_numpy(z).float()
        val = self(z)
        return val.detach().cpu().numpy()
