import torch
import torch.nn as nn
from torch.distributions import Normal


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor):
    """Compute pairwise distance matrix.

    :param x: a tensor with shape *[n, d]*
    :type x: torch.Tensor
    :param y: a tensor with shape *[m, d]*
    :type y: torch.Tensor
    :return: a tensor with shape *[n, m]*
    :rtype: torch.Tensor
    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    y_norm = y.pow(2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def gaussian_kernel_log(x: torch.Tensor, y: torch.Tensor, ell2):
    """Evaluate log of Gaussian kernel for all pairs of x and y.

    :param x: a tensor with shape *[n, d]*.
    :type x: torch.Tensor
    :param y: a tensor with shape *[m, d]*.
    :type y: torch.Tensor
    :param ell2: kernel length scale squared
    :type ell2: [float, Tensor]

    :return: a tensor with shape *[n, m]*
    :rtype: torch.Tensor
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    x = x.unsqueeze(1)  # reshape to [n, 1, d]
    y = y.unsqueeze(0)  # reshape to [1, m, d]
    X = x.expand(n, m, d)
    Y = y.expand(n, m, d)
    log_K = -0.5 * (X - Y).pow(2).sum(2) / ell2
    return log_K


class MMD(nn.Module):
    """Empirical Maximum Mean Discrepancy of two samples using a Gaussian kernel.
    :param ell2: Kernel length scale squared. If none, is set to ell^2 = d/2.
    :type ell2: float
    :param D: dimension
    :type D: int
    """

    def __init__(self, D: int, ell2=None):
        super().__init__()
        if ell2 is None:
            ell2 = 0.5 * D
        self.ell2 = ell2
        self.D = D

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Forward pass that evaluates MMD.
        :param x: Sample from the first distribution. A tensor with shape *[m, D]*.
        :type x: torch.Tensor
        :param y: Sample from the second distribution. A tensor with shape *[n, D]*.
        :type y: torch.Tensor
        """
        d1 = x.shape[1]
        d2 = y.shape[1]
        D = self.D
        assert d1 == D, "x must have " + str(D) + " columns! Found = " + str(d1)
        assert d2 == D, "y must have " + str(D) + " columns! Found = " + str(d2)
        kxx = torch.exp(gaussian_kernel_log(x, x, self.ell2))
        kyy = torch.exp(gaussian_kernel_log(y, y, self.ell2))
        kxy = torch.exp(gaussian_kernel_log(x, y, self.ell2))
        mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
        return torch.sqrt(mmd)

    def forward_numpy(self, x, y):
        """Forward pass but with numpy arrays as input."""
        xt = torch.from_numpy(x).float()
        yt = torch.from_numpy(y).float()
        mmd = self.forward(xt, yt)
        return mmd


def log_eps(x):
    """Numerically stable logarithm."""
    return torch.log(x + 1e-8)


def mvrnorm(mu, s2):
    """Draw random samples from N(mu, s2)."""
    dist = Normal(mu, s2.sqrt())
    smp = dist.rsample()
    return smp


def dot_torch(a, b):
    """Pairwise dot products of the rows of a and b. Matrices a and b must have equal
    shape.
    """
    N = a.shape[0]
    D = a.shape[1]
    return torch.bmm(a.view(N, 1, D), b.view(N, D, 1)).view(N, -1)


def logit_torch(p, eps=1e-8):
    """Logit transformation using torch."""
    return torch.log((p + eps) / (1 - p + eps))


def normalize_torch(x: torch.tensor):
    """Normalize expression (torch tensor input and output)."""
    s = x.sum(axis=1)
    x = 100 * x / s[:, None]
    return torch.log1p(x)


def normalize_numpy(x):
    """Normalize expression (numpy array input and output)."""
    x = torch.from_numpy(x).float()
    x = normalize_torch(x)
    return x.detach().numpy()
