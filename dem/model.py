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

    def __init__(self, D: int, z0, n_hidden: int = 128):
        super().__init__()
        f = TanhNetOneLayer(D, D, n_hidden)
        self.ode = NeuralDE(f, sensitivity="adjoint", solver="dopri5")
        self.D = D
        assert len(z0) == D, "length of z0 must be equal to D!"
        self.z0_mean = torch.tensor(z0).float()
        self.z0_log_sigma = torch.log(torch.tensor([0.05]))
        self.outdir = os.getcwd()

    @property
    def z0_sigma(self):
        sigma = torch.exp(self.z0_log_sigma) * torch.ones(self.D).float()
        return sigma

    def forward(self, z0):
        N = z0.size(0)
        t_span = torch.linspace(0, 1, N).float()
        z = self.ode.trajectory(z0, t_span).diagonal()
        z = torch.transpose(z, 0, 1)
        return z

    def draw_z0(self, N: int):
        rand_e = torch.randn((N, self.D)).float()
        z0 = self.z0_mean + self.z0_sigma * rand_e
        return z0

    @torch.no_grad()
    def traj_numpy(self, z0, n_timepoints: int):
        z0 = torch.from_numpy(z0).float()
        t_span = torch.linspace(0, 1, n_timepoints).float()
        z = self.ode.trajectory(z0, t_span)
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
