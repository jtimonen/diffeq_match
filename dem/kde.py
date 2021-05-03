import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gaussian_kde

from .math import mvrnorm, log_eps, accuracy, gaussian_kernel_log
from .networks import LeakyReluNetTwoLayer


class KdeDiscriminator(nn.Module):
    """Classifier."""

    def __init__(self, D: int, n_hidden: int = 64):
        super().__init__()
        self.net = LeakyReluNetTwoLayer(D, 1, n_hidden)
        self.D = D

    def forward(self, z):
        z = self.net(z)
        validity = torch.sigmoid(z)
        return validity

    def classify_numpy(self, z):
        z = torch.from_numpy(z).float()
        val = self(z)
        return val.detach().cpu().numpy()


def bandwidth_silverman(x_base):
    """Determine KDE bandwidth using Silverman's rule."""
    kde = gaussian_kde(x_base.T)
    return 0.5 * kde.silverman_factor()


class KDE(nn.Module):
    """KDE using a Gaussian kernel."""

    def __init__(self):
        super().__init__()
        self.sigma = 0.1
        self.eps = 1e-8

    def forward(self, x_eval: torch.Tensor, x_base: torch.Tensor):
        """Returns logarithm of KDE value."""
        N = x_base.size(0)
        D = x_base.size(1)
        t1 = -0.5 * D * np.log(2 * np.pi) - np.log(self.sigma)
        t2 = gaussian_kernel_log(x_eval, x_base, self.sigma ** 2)
        val = 1.0 / N * torch.exp(t1 + t2).sum(dim=1)
        return torch.log(self.eps + val)

    def set_bandwidth(self, x_base):
        bw = bandwidth_silverman(x_base)
        self.sigma = bw
        print("KDE bandwidth set to", bw)


class ParamKDE(nn.Module):
    """KDE using a Gaussian kernel.
    :param sigma_init: initial standard deviation.
    :type sigma_init: float
    """

    def __init__(self, sigma_init: float):
        super().__init__()
        sig_t = torch.Tensor([np.log(sigma_init)]).float()
        self.log_sigma = nn.Parameter(sig_t, requires_grad=True)
        self.eps = 1e-8

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def forward(self, x_eval: torch.Tensor, x_base: torch.Tensor):
        """Returns logarithm of KDE value."""
        N = x_base.size(0)
        D = x_base.size(1)
        t1 = -0.5 * D * np.log(2 * np.pi) - torch.log(self.sigma)
        t2 = gaussian_kernel_log(x_eval, x_base, self.sigma ** 2)
        val = 1.0 / N * torch.exp(t1 + t2).sum(dim=1)
        return torch.log(self.eps + val)
