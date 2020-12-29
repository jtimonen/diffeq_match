import torch
import torch.nn as nn
import numpy as np
from .ode import ODE
from .networks import PseudotimeEncoder
from .math import MMD
from .plotting import plot_match
from .training import train_model


class GenODE(nn.Module):
    """Main model module."""

    def __init__(self, D: int, z0, H_pte: int = 32, H_ode: int = 32):
        super().__init__()
        self.pte = PseudotimeEncoder(D, H_pte)
        self.ode = ODE(D, H_ode)
        self.D = self.ode.D
        self.z0_mean = torch.tensor(z0).float()
        self.z0_sigma = 0.05 * torch.ones(2).float()
        # nn.Parameter(0.1 * torch.ones(2), requires_grad=True)
        self.mmd = MMD(D=D, ell2=1.0)

    def draw_z0t0(self, N: int):
        rand_e = torch.randn((N, self.D)).float()
        z0 = self.z0_mean + self.z0_sigma * rand_e
        t = np.linspace(0, 1, N)
        t = torch.from_numpy(t).float()
        return z0, t

    def forward(self, z0, t, n_steps: int = 30):
        zf = self.ode(z0, t, n_steps)
        return zf

    def loss(self, z_data, zf):
        loss = self.mmd(zf, z_data)
        return loss

    def plot_forward(self, z_data, zf):
        loss = self.mmd(zf, z_data).item()
        zf = zf.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        z0 = self.z0_mean
        plot_match(z_data, zf, z0, loss)

    def fit(self, z_data, n_epochs: int = 100, lr: float = 0.001):
        optim = torch.optim.Adam(params=self.parameters(), lr=lr)
        train_model(self, z_data, optim, n_epochs)
