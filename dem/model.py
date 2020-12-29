import torch
import torch.nn as nn
import numpy as np
from torchdyn.models import NeuralDE
import pytorch_lightning as pl

from .math import MMD
from .plotting import plot_match
from .training import Learner
from .data import create_dataloader, MyDataset
from .utils import create_grid_around
from .networks import TanhNetOneLayer


class GenODE(nn.Module):
    """Main model module."""

    def __init__(self, D: int, z0, n_hidden: int = 32):
        super().__init__()
        f = TanhNetOneLayer(D, D, n_hidden)
        self.ode = NeuralDE(f, sensitivity="adjoint", solver="dopri5")
        self.D = D
        assert len(z0) == D, "length of z0 must be equal to D!"
        self.z0_mean = torch.tensor(z0).float()
        self.z0_sigma = 0.1 * torch.ones(D).float()

    def draw_z0t0(self, N: int):
        rand_e = torch.randn((N, self.D)).float()
        z0 = self.z0_mean + self.z0_sigma * rand_e
        t = np.linspace(0, 1, N)
        t = torch.from_numpy(t).float()
        return z0, t

    def forward(self, n_draws: int = 300):
        z0, t = self.model.draw_z0t0(N=n_draws)
        print("z0:", z0.shape)
        print("t", t.shape)
        zf = self.ode(z0, t)
        return zf

    def loss(self, z_data, zf):
        loss = self.mmd(zf, z_data)
        return loss

    def plot_forward(self, z_data, zf, z_traj=None):
        loss = self.mmd(zf, z_data).item()
        zf = zf.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        z0 = self.z0_mean
        u_grid = create_grid_around(z_data, 16)
        v_grid = self.ode.f_numpy(u_grid)
        plot_match(z_data, zf, z0, loss, u_grid, v_grid, z_traj)

    def fit(
        self,
        z_data,
        n_draws=None,
        batch_size=None,
        n_epochs: int = 10,
        lr: float = 0.005,
        mmd_ell: float = 1.0,
    ):
        mmd = MMD(D=self.D, ell2=mmd_ell)
        ds = MyDataset(z_data)
        dataloader = create_dataloader(ds, batch_size)
        min_epochs = n_epochs
        max_epochs = 2 * n_epochs
        learner = Learner(self, dataloader, mmd, lr, n_draws)
        trainer = pl.Trainer(min_epochs=min_epochs, max_epochs=max_epochs)
        trainer.fit(learner)
