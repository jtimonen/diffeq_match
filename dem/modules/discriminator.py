import abc
import numpy as np
import torch
import torch.nn as nn

from dem.modules.networks import MultiLayerNet
from dem.modules.kde import KDE, ParamKDE


class Discriminator(nn.Module, abc.ABC):
    """Abstract binary classifier class."""

    def __init__(self, D: int):
        super().__init__()
        self.D = D

    def forward_numpy(self, x: np.ndarray):
        x = torch.from_numpy(x).float()
        val = self(x)
        return val.detach().cpu().numpy()

    @torch.no_grad()
    def classify(self, x: np.ndarray):
        val = self.forward_numpy(x)
        labels = (val > 0.5).astype(float)
        return labels.ravel(), val.ravel()


class NeuralDiscriminator(Discriminator):
    """Binary classifier using a neural network."""

    def __init__(self, D: int, n_hidden: int = 48, n_hidden_layers=2, activation=None):
        super().__init__(D)
        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)
        self.net = MultiLayerNet(
            n_input=D,
            n_hidden=n_hidden,
            n_hidden_layers=n_hidden_layers,
            n_output=1,
            activation=activation,
        )

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.net(x))

    def __repr__(self):
        str0 = self.net.__repr__()
        return "NeuralDiscriminator with " + str0


class KdeDiscriminator(Discriminator):
    """Binary classifier using kernel density estimation.

    :param D: dimension
    :param bw_init: initial bandwidth of kernel
    :param trainable: is the bandwidth a trainable parameter
    """

    def __init__(self, D: int, bw_init: float = 0.2, trainable: bool = False):
        super().__init__(D)
        self.trainable = trainable
        self.kde = ParamKDE(bw_init) if trainable else KDE(bw_init)
        self.x0 = None
        self.x1 = None

    def __repr__(self):
        str0 = str(self.trainable)
        return "KDEDiscriminator(trainable=" + str0 + ")"

    def set_data(self, x0=None, x1=None):
        if x0 is not None:
            self.x0 = x0
        if x1 is not None:
            self.x1 = x1

    def set_data_numpy(self, x0: np.ndarray, x1: np.ndarray):
        x0 = torch.from_numpy(x0).float()
        x1 = torch.from_numpy(x1).float()
        self.set_data(x0, x1)

    def forward(self, x: torch.Tensor):
        score_class0 = self.kde(x, self.x0)
        score_class1 = self.kde(x, self.x1)
        return score_class1 / (score_class0 + score_class1)
