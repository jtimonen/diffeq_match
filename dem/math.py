import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, kl_divergence
from sklearn.linear_model import LinearRegression


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


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, ell2):
    """Evaluate Gaussian kernel for all pairs of x and y.

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
    log_K = -0.5 * (X - Y).pow(2).mean(2) / ell2
    return torch.exp(log_K)


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
            ell2 = 0.1
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
        kxx = gaussian_kernel(x, x, self.ell2)
        kyy = gaussian_kernel(y, y, self.ell2)
        kxy = gaussian_kernel(x, y, self.ell2)
        mmd = kxx.mean() + kyy.mean() - 2 * kxy.mean()
        return mmd

    def forward_numpy(self, x, y):
        """Forward pass but with numpy arrays as input."""
        xt = torch.from_numpy(x).float()
        yt = torch.from_numpy(y).float()
        mmd = self.forward(xt, yt)
        return mmd


def linreg(x, y):
    """Compute a 1-d linear regression line"""
    model = LinearRegression()
    X = x.reshape(-1, 1)
    model.fit(X, y)
    y_star = model.predict(X)
    k = model.coef_[0]
    R2 = model.score(X, y)
    return k, R2, y_star


def log_nb(x, mu, phi, eps: float = 1e-8):
    """Compute log density of the Negative Binomial distribution at x.

    :param x: input argument, must be tensor with shape *[n_minibatch, G]* and consist of non-negative integers
    :type x: torch.Tensor
    :param mu: positive mean parameter, must be tensor with shape *[n_minibatch, G]*
    :type mu: torch.Tensor
    :param phi: positive inverse over-dispersion parameter, must be tensor with shape *[n_minibatch, G]*
    :type phi: torch.Tensor
    :param eps: numerical stability constant for logarithms
    :type eps: float

    :return: a tensor of shape *[n_minibatch, G]*
    :rtype: torch.Tensor
    :raises: ValueError if log probability evaluates to minus infinity
    """
    lbin = torch.lgamma(phi + x) - torch.lgamma(phi) - torch.lgamma(x + 1)

    terms = (
        x * torch.log(mu + eps)
        + phi * torch.log(phi + eps)
        - (x + phi) * torch.log(mu + phi + eps)
    )

    logp = lbin + terms
    # logp = torch.clamp(logp, max=0.0)
    return logp


class NegBinomialLikelihood(nn.Module):
    """Negative Binomial likelihood module.

    :param G: number of genes
    :type G: int
    """

    def __init__(self, G: int):
        super().__init__()
        self.G = G
        self.log_phi_s = 1 + torch.nn.Parameter(
            0.2 * torch.randn(G), requires_grad=True
        )
        self.log_phi_u = 1 + torch.nn.Parameter(
            0.2 * torch.randn(G), requires_grad=True
        )

    @property
    def phi_s(self):
        return torch.exp(self.log_phi_s)

    @property
    def phi_u(self):
        return torch.exp(self.log_phi_u)

    def forward(self, x: torch.Tensor, lam: torch.Tensor, indicator: int):
        """Forward pass that computes the log likelihood for each cell.

        :param x: Points where to compute the log likelihood. A tensor with shape *[n_minibatch, G]*.
        :type x: torch.Tensor
        :param lam: Means of the NB distributions. A tensor with shape *[n_minibatch, G]*.
        :type lam: torch.Tensor
        :param indicator:

        :return: A tensor with shape *[n_minibatch]*, containing log densities.
        :rtype: torch.Tensor
        """
        phi = self.phi_s if (indicator == 0) else self.phi_u
        log_lh = log_nb(x, lam, phi).sum(dim=1)  # sum over genes
        return log_lh


def log_lh_normal(x, mu, sigma2):
    """Compute log density of the Gaussian distribution at x."""
    pi = 3.14159265359
    logp = -0.5 * torch.log(2.0 * pi * sigma2) - 0.5 * (x - mu).pow(2) / sigma2
    return logp


class GaussianLikelihood(nn.Module):
    """Gaussian likelihood module."""

    def __init__(self, G: int):
        super().__init__()
        self.G = G
        self.log_sigma2 = torch.nn.Parameter(
            0.0 + 1.0 * torch.randn(1), requires_grad=True
        )

    @property
    def sigma2(self):
        return torch.exp(self.log_sigma2)

    def forward(self, x: torch.Tensor, mu: torch.Tensor):
        """Forward pass. Returns average log likelihood of the batch."""
        llh = log_lh_normal(x, mu, self.sigma2).sum(dim=1)  # sum over genes
        return torch.mean(llh)


def kl_div(mu_p, s2_p, mu_q=0.0, s2_q=1.0):
    """Computes average KL Divergence of pairs of two diagonal multivariate Gaussian distributions p and q."""
    mu_q = torch.ones_like(mu_p) * mu_q
    s2_q = torch.ones_like(mu_p) * s2_q
    p = Normal(mu_p, torch.sqrt(s2_p))
    q = Normal(mu_q, torch.sqrt(s2_q))
    kl = kl_divergence(p, q).sum(dim=1)
    average_kl = kl.mean()
    return average_kl


def mvrnorm(mu, s2):
    """Draw random samples from N(mu, s2)."""
    dist = Normal(mu, s2.sqrt())
    smp = dist.rsample()
    return smp


class OdeTarget(nn.Module):
    """Vector field target."""

    def __init__(self, D: int, N: int):
        super().__init__()
        self.ell = nn.Parameter(torch.Tensor([0.5]).float(), requires_grad=False)
        self.sf = nn.Parameter(torch.Tensor([0.5]).float(), requires_grad=False)
        self.D = D
        self.N = N
        self.Delta = 1e-8 * torch.eye(self.N).float()
        self.eps = 1e-8
        self.z = nn.Parameter(
            torch.from_numpy(np.random.normal(size=(self.N, self.D))).float(),
            requires_grad=False,
        )
        self.u = nn.Parameter(
            torch.from_numpy(np.random.normal(size=(self.N, self.D))).float(),
            requires_grad=False,
        )
        self.v = nn.Parameter(
            torch.from_numpy(np.random.normal(size=(self.N, self.D))).float(),
            requires_grad=False,
        )

    def initialize(self, u: np.ndarray, v: np.ndarray, ell: float):
        """Initialize the tree."""
        k = u.shape[0]
        D = u.shape[1]
        assert D == self.D, "u has invalid number of columns"
        assert k == self.N, (
            "u has invalid number of rows (found="
            + str(k)
            + ", expected="
            + str(self.N)
        )
        self.u = nn.Parameter(torch.from_numpy(u).float(), requires_grad=False)
        self.v = nn.Parameter(torch.from_numpy(v).float(), requires_grad=False)
        self.ell = nn.Parameter(torch.Tensor([ell]).float(), requires_grad=False)
        sf = np.var(v, axis=0)
        print("ell=", ell)
        print("sf=", sf)
        sf = np.mean(sf)
        self.sf = nn.Parameter(torch.Tensor([sf]).float(), requires_grad=False)

    def kernel(self, x, y):
        """Computes a kernel matrix using the squared exponential kernel."""
        K = gaussian_kernel(x, y, self.ell ** 2)
        return self.sf * K

    def forward(self, z: torch.Tensor):
        """Evaluate the vector field."""
        Kxu = self.kernel(z, self.u)
        Kuu_inv = torch.inverse(self.kernel(self.u, self.u) + self.Delta)
        f = torch.mm(Kxu, torch.mm(Kuu_inv, self.v))
        return f

    def forward_numpy(self, z):
        """Forward pass but with numpy arrays."""
        zt = torch.from_numpy(z).float()
        f = self.forward(zt)
        return f.detach().cpu().numpy()
