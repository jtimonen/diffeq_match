import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gaussian_kde

from dem.utils.math import gaussian_kernel_log


def bandwidth_silverman(x_base: np.ndarray):
    """Determine KDE bandwidth using Silverman's rule."""
    kde = gaussian_kde(x_base.T)
    return 0.5 * kde.silverman_factor()


class KDE(nn.Module):
    """Kernel density estimator using a Gaussian kernel.

    :param bw_init: initial bandwidth.
    :type bw_init: float
    """

    def __init__(self, bw_init: float = 0.2):
        super().__init__()
        self.bw_init = bw_init
        self.log_bw = torch.tensor([np.log(bw_init)])
        self.set_bw(bw_init)

    @property
    def bw(self):
        return torch.exp(self.log_bw)

    def set_bw(self, value: float):
        assert value > 0, "bandwidth must be positive"
        self.log_bw = torch.tensor([np.log(value)])
        self.set_bw_finally()

    def set_bw_silverman(self, x_base: np.ndarray):
        bw = bandwidth_silverman(x_base)
        self.set_bw(bw)

    def set_bw_finally(self):
        print("KDE bandwidth set to", self.bw)

    def forward(self, x_eval: torch.Tensor, x_base: torch.Tensor):
        """Returns KDE value."""
        N = x_base.size(0)
        D = x_base.size(1)
        t1 = -0.5 * D * np.log(2 * np.pi) - torch.log(self.bw)
        t2 = gaussian_kernel_log(x_eval, x_base, self.bw ** 2)
        val = 1.0 / N * torch.exp(t1 + t2).sum(dim=1)
        return val


class ParamKDE(KDE):
    """KDE where bandwidth is a trainable parameter."""

    def __init__(self, bw_init: float = 0.2):
        super().__init__(bw_init)

    def set_bw_finally(self):
        print("KDE bandwidth (trainable parameter) set to", self.bw)
        self.log_bw = nn.Parameter(self.log_bw, requires_grad=True)
