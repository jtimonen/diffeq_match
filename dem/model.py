import torch
import torch.nn as nn
import numpy as np
from .ode import ODE
from .networks import PseudotimeEncoder
from .math import MMD
from .plotting import plot_match
from .training import train_model
from .data import create_dataloader, MyDataset
from .utils import create_grid_around


class GenODE(nn.Module):
    """Main model module."""

    def __init__(self, D: int, z0, H_ode: int = 64):
        super().__init__()
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
        u_grid = create_grid_around(z_data, 16)
        v_grid = self.ode.f_numpy(u_grid)
        plot_match(z_data, zf, z0, loss, u_grid, v_grid)

    def fit(
        self,
        z_data,
        n_draws=None,
        batch_size=None,
        n_epochs: int = 100,
        lr: float = 0.005,
    ):
        optim = torch.optim.Adam(params=self.parameters(), lr=lr)
        ds = MyDataset(z_data)
        dl = create_dataloader(ds, batch_size)
        if n_draws is None:
            n_draws = len(ds)
        train_model(self, dl, optim, n_draws, n_epochs)
