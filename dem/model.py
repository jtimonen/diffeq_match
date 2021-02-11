import os
import torch
import torch.nn as nn
import numpy as np
from torchdyn.models import NeuralDE
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from .plotting import plot_match

from .math import KDE
from .data import create_dataloader, MyDataset
from .networks import TanhNetOneLayer, TanhNetTwoLayer
from .callbacks import MyCallback


class Reverser(nn.Module):
    """Reverses the sign of nn.Module output."""

    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, x):
        return -self.f(x)


class GenODE(nn.Module):
    """Main model module."""

    def __init__(
        self,
        init_loc,
        init_std,
        terminal_loc,
        terminal_std,
        n_hidden: int = 24,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        sensitivity="adjoint",
        solver="dopri5",
        sigma: float = 0.02,
    ):
        super().__init__()
        terminal_loc = np.array(terminal_loc)
        init_loc = np.array(init_loc)
        D = terminal_loc.shape[1]
        f = TanhNetTwoLayer(D, D, n_hidden)
        self.ode = NeuralDE(
            f, sensitivity=sensitivity, solver=solver, atol=atol, rtol=rtol
        )
        self.ode_b = NeuralDE(
            Reverser(f), sensitivity=sensitivity, solver=solver, atol=atol, rtol=rtol
        )
        self.D = D
        self.kde = KDE(sigma=sigma)
        self.outdir = os.getcwd()

        self.n_terminal = terminal_loc.shape[0]
        self.terminal_loc = torch.from_numpy(terminal_loc).float()
        self.log_terminal_std = torch.log(torch.tensor(terminal_std).float())

        self.n_init = init_loc.shape[0]
        self.init_loc = torch.from_numpy(init_loc).float()
        self.log_init_std = torch.log(torch.tensor(init_std).float())

    @property
    def terminal_std(self):
        sigma = torch.exp(self.log_terminal_std).view(-1, 1)
        return sigma.repeat(1, self.D)

    @property
    def init_std(self):
        sigma = torch.exp(self.log_init_std).view(-1, 1)
        return sigma.repeat(1, self.D)

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

    def draw_init(self, N: int):
        rand_e = torch.randn((N, self.D)).float()
        P = self.n_init
        M = int(N / P)
        if M * P != N:
            raise ValueError("N not divisible by number of initial points")
        m = self.init_loc.repeat(M, 1)
        s = self.init_std.repeat(M, 1)
        z = m + s * rand_e
        return z

    def traj(self, z_init, ts, direction: int = 1):
        if direction == 1:
            return self.ode.trajectory(z_init, ts)
        elif direction == -1:
            return self.ode_b.trajectory(z_init, ts)
        else:
            raise ValueError("direction must be -1 or 1!")

    def forward(self, z_start, direction: int):
        N = z_start.size(0)
        ts = torch.linspace(0, 1, N).float()
        z = self.traj(z_start, ts, direction).diagonal()
        z = torch.transpose(z, 0, 1)
        return z

    @torch.no_grad()
    def defunc_numpy(self, z):
        z = torch.from_numpy(z).float()
        f = self.ode.defunc(0, z).cpu().detach().numpy()
        return f

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

    def generate_fake_data(self, z_data):
        """Returns three torch tensors with shape [N, D]."""
        N = z_data.size(0)
        z_term_draw = self.model.draw_terminal(N=N)
        z_back = self.model(z_term_draw, direction=-1)
        z0 = (z_back[-1, :]).repeat(N, 1)
        rand_e = torch.randn((N, self.model.D)).float()
        s = self.model.init_std.repeat(N, 1)
        z_init_draw = z0 + s * rand_e
        z_forw = self.model(z_init_draw, direction=1)
        return z_back, z_forw

    def loss_terms(self, z_data, z_forw, z_back):
        loss1 = -torch.mean(self.model.kde(z_back, z_data))
        loss2 = -torch.mean(self.model.kde(z_forw, z_data))
        loss3 = -torch.mean(self.model.kde(z_data, z_back))
        loss4 = -torch.mean(self.model.kde(z_data, z_forw))
        return loss1, loss2, loss3, loss4

    def training_step(self, data_batch, batch_idx):
        z_data = data_batch
        z_back, z_forw = self.generate_fake_data(z_data)
        lt1, lt2, lt3, lt4 = self.loss_terms(z_data, z_forw, z_back)
        loss = 0.25 * (lt1 + lt2 + lt3 + lt4)
        return loss

    def validation_step(self, data_batch, batch_idx):
        z_data = data_batch
        z_back, z_forw = self.generate_fake_data(z_data)
        z_back = z_back.detach()
        z_forw = z_forw.detach()
        lt1, lt2, lt3, lt4 = self.loss_terms(z_data, z_forw, z_back)
        loss = 0.25 * (lt1 + lt2 + lt3 + lt4)
        self.log("valid_loss", loss)
        self.log("valid_lt1", lt1)
        self.log("valid_lt2", lt2)
        self.log("valid_lt3", lt3)
        self.log("valid_lt4", lt4)
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(z_back, z_forw, z_data, loss, idx_epoch)
        return loss

    def visualize(self, z_back, z_forw, z_data, loss, idx_epoch):
        outdir = self.outdir
        fig_dir = os.path.join(outdir, "figs")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        z_forw = z_forw.detach().cpu().numpy()
        z_back = z_back.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        plot_match(self.model, z_back, z_forw, z_data, idx_epoch, loss, fig_dir)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)
        return opt_g

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
