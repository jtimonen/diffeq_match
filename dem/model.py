import os
import torch
import torch.nn as nn
import numpy as np
from torchdyn.models import NeuralDE
from pytorch_lightning import Trainer

from .math import MMD
from .training import MMDLearner, GANLearner
from .data import create_dataloader, MyDataset
from .networks import TanhNetOneLayer, ReluNetTwoLayer
from .callbacks import MyCallback
from pytorch_lightning.callbacks import LearningRateMonitor


class GenODE(nn.Module):
    """Main model module."""

    def __init__(
        self,
        terminal_loc,
        terminal_std,
        n_hidden: int = 128,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        super().__init__()
        terminal_loc = np.array(terminal_loc)
        D = terminal_loc.shape[1]
        f = TanhNetOneLayer(D, D, n_hidden)
        self.ode = NeuralDE(
            f, sensitivity="adjoint", solver="dopri5", atol=atol, rtol=rtol
        )
        self.D = D
        self.n_terminal = terminal_loc.shape[0]
        assert len(terminal_std) == D, "terminal_std must have length " + D
        self.terminal_loc = torch.from_numpy(terminal_loc).float()
        self.log_terminal_std = torch.log(torch.tensor(terminal_std).float())
        self.outdir = os.getcwd()

    @property
    def terminal_std(self):
        sigma = torch.exp(self.log_terminal_std).view(-1, 1)
        return sigma.repeat(1, self.D)

    def forward(self, z_end):
        N = z_end.size(0)
        t_span = torch.linspace(0, 1, N).float()
        z = self.ode.trajectory(z_end, t_span).diagonal()
        z = torch.transpose(z, 0, 1)
        return z

    def draw_terminal(self, N: int):
        rand_e = torch.randn((N, self.D)).float()
        P = self.n_terminal
        M = int(N / P)
        if M * P != N:
            raise ValueError("N not divisible by number of terminal points")
        m = self.terminal_loc.repeat(M, 1)
        s = self.terminal_std.repeat(M, 1)
        z = m + s * rand_e
        return z

    @torch.no_grad()
    def traj_numpy(self, z_end, n_timepoints: int):
        z_end = torch.from_numpy(z_end).float()
        t_span = torch.linspace(0, 1, n_timepoints).float()
        z = self.ode.trajectory(z_end, t_span)
        return z.detach().cpu().numpy()

    def defunc_numpy(self, z):
        z = torch.from_numpy(z).float()
        f = self.ode.defunc(0, z).cpu().detach().numpy()
        return f

    def fit_mmd(
        self,
        z_data,
        n_draws=100,
        n_timepoints=30,
        batch_size=None,
        n_epochs: int = 100,
        lr: float = 0.005,
        lr_decay: float = 1e-5,
        mmd_ell: float = 1.0,
        num_workers: int = 0,
        plot_freq=0,
    ):
        mmd = MMD(D=self.D, ell2=mmd_ell)
        ds = MyDataset(z_data)
        dataloader = create_dataloader(ds, batch_size, num_workers)
        min_epochs = n_epochs
        max_epochs = n_epochs
        learner = MMDLearner(
            self,
            dataloader,
            mmd,
            n_draws,
            n_timepoints,
            lr,
            lr_decay,
            plot_freq,
        )
        save_path = learner.outdir
        trainer = Trainer(
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            default_root_dir=save_path,
            callbacks=[MyCallback()],
        )
        trainer.fit(learner)

    def fit_gan(
        self,
        z_data,
        batch_size=None,
        n_epochs: int = 100,
        lr: float = 0.001,
        lr_decay: float = 1e-5,
        disc_hidden: int = 64,
        num_workers: int = 0,
        plot_freq=0,
    ):
        disc = Discriminator(D=self.D, n_hidden=disc_hidden)
        ds = MyDataset(z_data)
        dataloader = create_dataloader(ds, batch_size, num_workers)
        min_epochs = n_epochs
        max_epochs = n_epochs
        learner = GANLearner(
            self,
            dataloader,
            disc,
            lr,
            lr_decay,
            plot_freq,
        )
        save_path = learner.outdir
        lr_monitor = LearningRateMonitor(logging_interval="step", log_momentum=True)
        plotter = MyCallback()
        trainer = Trainer(
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            default_root_dir=save_path,
            callbacks=[plotter, lr_monitor],
        )
        trainer.fit(learner)


class Discriminator(nn.Module):
    """Classifier."""

    def __init__(self, D: int, n_hidden: int = 64):
        super().__init__()
        self.net = ReluNetTwoLayer(D, 1, n_hidden)
        self.D = D

    def forward(self, z):
        z = self.net(z)
        validity = torch.sigmoid(z)
        return validity

    def classify_numpy(self, z):
        z = torch.from_numpy(z).float()
        val = self(z)
        return val.detach().cpu().numpy()

    def accuracy(self, val, target):
        N = len(target)
        a = np.round(val)
        corr = np.sum(a == target)
        return corr / N
