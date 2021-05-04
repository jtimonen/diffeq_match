import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

from .networks import LeakyReluNetTwoLayer
from .kde import KDE, ParamKDE
from .math import log_eps


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
        labels = (val > 0.5).astype(float)
        return labels.ravel()

    def accuracy(self, x: np.ndarray, y_true, y_pred=None):
        if y_pred is None:
            y_pred = self.classify(x)
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    def confusion_matrix(self, x: np.ndarray, y_true, y_pred=None):
        if y_pred is None:
            y_pred = self.classify(x)
        return confusion_matrix(y_true=y_true, y_pred=y_pred)


class NeuralDiscriminator(Discriminator):
    """Binary classifier using a neural network.

    :param D: dimension
    :param n_hidden: number of hidden nodes
    """

    def __init__(self, D: int, n_hidden: int = 64):
        super().__init__(D)
        self.net = LeakyReluNetTwoLayer(D, 1, n_hidden)

    def forward(self, x: torch.Tensor):
        return torch.sigmoid(self.net(x))


class KdeDiscriminator(Discriminator):
    """Binary classifier using kernel density estimation.

    :param D: dimension
    :param bw_init: initial bandwidth of kernel
    :param trainable: is the bandwidth a trainable parameter
    """

    def __init__(self, D: int, bw_init: float = 0.1, trainable: bool = False):
        super().__init__(D)
        self.trainable = trainable
        self.kde = ParamKDE(bw_init) if trainable else KDE(bw_init)
        self.x0 = None
        self.x1 = None

    def set_data(self, x0: torch.Tensor, x1: torch.Tensor):
        self.x0 = x0
        self.x1 = x1

    @torch.no_grad()
    def set_data_numpy(self, x0: np.ndarray, x1: np.ndarray):
        x0 = torch.from_numpy(x0).float()
        x1 = torch.from_numpy(x1).float()
        self.set_data(x0, x1)

    @property
    def data_is_set(self):
        a = self.x0 is not None
        b = self.x1 is not None
        return a and b

    def forward(self, x: torch.Tensor):
        if not self.data_is_set:
            raise RuntimeError("Data not set!")
        score_class0 = self.kde(x, self.x0)
        score_class1 = self.kde(x, self.x1)
        log_p_class1 = log_eps(score_class1) - log_eps(score_class0 + score_class1)
        return torch.exp(log_p_class1)
