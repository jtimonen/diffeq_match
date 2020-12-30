import os
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

    def __init__(self, D: int, z0, n_hidden: int = 128):
        super().__init__()
        f = TanhNetOneLayer(D, D, n_hidden)
        self.ode = NeuralDE(f, sensitivity="adjoint", solver="dopri5")
        self.D = D
        assert len(z0) == D, "length of z0 must be equal to D!"
        self.z0_mean = torch.tensor(z0).float()
        self.z0_log_sigma = torch.nn.Parameter(torch.tensor([-4.0]),
                                               requires_grad=True)

    @property
    def z0_sigma(self):
        sigma = torch.exp(self.z0_log_sigma) * torch.ones(self.D).float()
        return sigma

    def draw_z0t0(self, N: int):
        rand_e = torch.randn((N, self.D)).float()
        z0 = self.z0_mean + self.z0_sigma * rand_e
        t = np.linspace(0, 1, N)
        t = torch.from_numpy(t).float()
        return z0, t

    def forward(self, n_draws: int, n_timepoints: int):
        z0, _ = self.draw_z0t0(N=n_draws)
        t_span = torch.linspace(0, 1, n_timepoints).float()
        z = self.ode.trajectory(z0, t_span)
        return z0, t_span, z

    @torch.no_grad()
    def visualize(self, z_data, n_draws: int, n_timepoints: int,
                  loss=None, out_dir=".", idx_epoch=None):
        z_traj = self.traj_numpy(n_draws, n_timepoints)
        z_data = z_data.detach().cpu().numpy()
        z0 = self.z0_mean
        u_grid = create_grid_around(z_data, 16)
        v_grid = self.defunc_numpy(u_grid)
        loss = round(loss, 5)
        fn = "fig_" + '{0:04}'.format(idx_epoch) + ".png"
        plot_match(z_data, z0, loss, u_grid, v_grid, z_traj,
                   save_dir=out_dir, save_name=fn)

    def defunc_numpy(self, z):
        z = torch.from_numpy(z).float()
        f = self.ode.defunc(0, z).cpu().detach().numpy()
        return f

    def traj_numpy(self, n_draws: int, n_timepoints: int):
        _, _, z_traj = self.forward(n_draws, n_timepoints)
        return z_traj.cpu().detach().numpy()

    def fit(
        self,
        z_data,
        n_draws=100,
        n_timepoints=30,
        batch_size=None,
        n_epochs: int = 100,
        lr: float = 0.005,
        mmd_ell: float = 1.0,
        num_workers: int = 0,
        out_dir="train_output"
    ):
        mmd = MMD(D=self.D, ell2=mmd_ell)
        ds = MyDataset(z_data)
        dataloader = create_dataloader(ds, batch_size, num_workers)
        min_epochs = n_epochs
        max_epochs = n_epochs
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        learner = Learner(self, dataloader, mmd, n_draws, n_timepoints, lr, out_dir)
        trainer = pl.Trainer(min_epochs=min_epochs, max_epochs=max_epochs)
        trainer.fit(learner)
