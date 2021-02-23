import os
import torch
import torch.nn as nn
import numpy as np
import torchdiffeq
import torchsde
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from .plotting import plot_state_2d, plot_state_3d

from .math import KDE
from .data import create_dataloader, MyDataset
from .networks import ReluNetOne, TanhNetTwoLayer
from .callbacks import MyCallback


class Reverser(nn.Module):
    """Reverses the sign of nn.Module output."""

    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, t, y):
        return -self.f(t, y)


class VectorField(nn.Module):
    def __init__(self, D, n_hidden):
        super().__init__()
        self.D = D
        self.net_f = TanhNetTwoLayer(D, D, n_hidden)
        self.net_g = TanhNetTwoLayer(D, 1, 24)
        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def forward(self, t, y):
        return self.f(t, y)

    def f(self, t, y):
        return self.net_f(y)

    def g(self, t, y):
        out = self.net_g(y)
        g = 0.3 * torch.sigmoid(out)
        return g * torch.ones_like(y)


class GenModel(nn.Module):
    """Main model module."""

    def __init__(
        self,
        term_loc,
        term_std,
        n_hidden: int = 24,
        sigma: float = 0.02,
    ):
        super().__init__()
        self.n_init = 1
        self.n_term = len(term_loc)
        D = len(term_loc[0])
        self.field = VectorField(D, n_hidden)
        self.field_b = Reverser(self.field)
        self.D = D
        self.kde = KDE(sigma=sigma)
        self.outdir = os.getcwd()

        self.term_loc = torch.tensor(term_loc).float()
        self.log_term_std = torch.log(torch.tensor(term_std).float())

    @property
    def term_std(self):
        sigma = torch.exp(self.log_term_std).view(-1, 1)
        return sigma.repeat(1, self.D)

    def draw_term(self, N: int):
        P = self.n_term
        M = int(N / P)
        if M * P != N:
            raise ValueError("N not divisible by number of terminal points")
        m = self.term_loc.repeat(M, 1)
        rand_e = torch.randn((N, self.D)).float()
        s = self.term_std.repeat(M, 1)
        z = m + s * rand_e
        return z

    def traj(self, z_init, ts, sde: bool = False, forward: bool = True):
        if forward:
            if sde:
                return torchsde.sdeint(self.field, z_init, ts, method="euler")
            else:
                return torchdiffeq.odeint_adjoint(self.field, z_init, ts)
        else:
            if sde:
                raise ValueError("Cannot integrate SDE backward!")
            else:
                return torchdiffeq.odeint_adjoint(self.field_b, z_init, ts)

    def forward(self, N: int):
        ts = torch.linspace(0, 1, N).float()
        t01 = torch.tensor([0.0, 1.0]).float()
        z_term = self.draw_term(N)
        z_init = self.traj(z_term, t01, sde=False, forward=False)[1, :, :]
        z_forw = self.traj(z_init, ts, sde=True, forward=True)
        z_samples = torch.transpose(z_forw.diagonal(), 0, 1)
        return z_init, z_forw, z_samples

    @torch.no_grad()
    def f_numpy(self, z):
        z = torch.from_numpy(z).float()
        f = self.field.f(0, z).cpu().detach().numpy()
        return f

    @torch.no_grad()
    def g_numpy(self, z):
        z = torch.from_numpy(z).float()
        g = self.field.g(0, z).cpu().detach().numpy()
        return g[:, 0]

    def fit(
        self,
        z_data,
        batch_size=128,
        n_epochs: int = 400,
        lr: float = 0.005,
        plot_freq=0,
    ):
        learner = TrainingSetup(
            self,
            z_data,
            batch_size,
            lr,
            plot_freq,
        )
        save_path = learner.outdir
        trainer = Trainer(
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            default_root_dir=save_path,
            callbacks=[MyCallback()],
        )
        trainer.fit(learner)


class TrainingSetup(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        z_data,
        batch_size: int,
        lr_init: float,
        plot_freq=0,
        outdir="out",
    ):
        super().__init__()
        num_workers = 0
        ds = MyDataset(z_data)
        train_loader = create_dataloader(ds, batch_size, num_workers, shuffle=True)
        valid_loader = create_dataloader(ds, None, num_workers, shuffle=False)
        self.N = len(train_loader.dataset)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_init = lr_init
        self.plot_freq = plot_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    def loss_terms(self, z_data, z_samples):
        loss1 = -torch.mean(self.model.kde(z_samples, z_data))
        loss2 = -torch.mean(self.model.kde(z_data, z_samples))
        return loss1, loss2

    def training_step(self, data_batch, batch_idx):
        z_data = data_batch
        N = z_data.size(0)
        _, _, z_samples = self.model(N)
        lt1, lt2 = self.loss_terms(z_data, z_samples)
        loss = 0.5 * (lt1 + lt2)
        return loss

    def validation_step(self, data_batch, batch_idx):
        z_data = data_batch
        N = z_data.size(0)
        _, z_forw, z_samp = self.model(N)
        lt1, lt2 = self.loss_terms(z_data, z_samp)
        loss = 0.5 * (lt1 + lt2)
        self.log("valid_loss", loss)
        self.log("valid_lt1", lt1)
        self.log("valid_lt2", lt2)
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(z_forw, z_samp, z_data, loss, idx_epoch)
        return loss

    def visualize(self, z_forw, z_samp, z_data, loss, idx_epoch):
        outdir = self.outdir
        fig_dir = os.path.join(outdir, "figs")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        z_forw = z_forw.detach().cpu().numpy()
        z_samp = z_samp.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        if self.model.D == 2:
            plot_state_2d(self.model, z_forw, z_samp, z_data, idx_epoch, loss, fig_dir)
        elif self.model.D == 3:
            plot_state_3d(self.model, z_forw, z_data, idx_epoch, loss, fig_dir)
        else:
            raise RuntimeError("plotting not implemented for D > 3")

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)
        return opt_g

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
