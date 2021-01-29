import os
import torch
import torch.nn as nn
import numpy as np
from torchdyn.models import NeuralDE
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from .plotting import plot_match
from .math import mvrnorm

from .math import MMD, log_eps
from .data import create_dataloader, MyDataset
from .networks import TanhNetOneLayer
from .callbacks import MyCallback


class GenODE(nn.Module):
    """Main model module."""

    def __init__(
        self,
        terminal_loc,
        terminal_std,
        n_hidden: int = 64,
        atol: float = 1e-5,
        rtol: float = 1e-5,
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
        S = self.n_terminal
        assert len(terminal_std) == S, "terminal_std must have length " + S
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
        """Note the minus."""
        z = torch.from_numpy(z).float()
        f = self.ode.defunc(0, z).cpu().detach().numpy()
        return -f

    def fit(
        self,
        z_data,
        batch_size=64,
        n_epochs: int = 100,
        lr: float = 0.005,
        lr_disc: float = 0.005,
        lr_decay: float = 1e-6,
        disc=None,
        plot_freq=0,
    ):
        mmd = MMD(D=self.D, ell2=1.0)
        learner = TrainingSetup(
            self,
            z_data,
            batch_size,
            disc,
            mmd,
            lr,
            lr_disc,
            lr_decay,
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
        disc: nn.Module,
        mmd: nn.Module,
        lr_init: float,
        lr_disc_init: float,
        lr_decay: float,
        plot_freq=0,
        outdir="out",
    ):
        super().__init__()
        self.mode = "mmd" if (disc is None) else "gan"
        num_workers = 0
        ds = MyDataset(z_data)
        train_loader = create_dataloader(ds, batch_size, num_workers, shuffle=True)
        valid_loader = create_dataloader(ds, None, num_workers, shuffle=False)
        self.N = len(train_loader.dataset)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_init = lr_init
        self.lr_disc_init = lr_disc_init
        self.lr_decay = lr_decay
        self.disc = disc
        self.mmd = mmd
        self.plot_freq = plot_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    def training_step(self, data_batch, batch_idx, optimizer_idx=0):
        z_data = data_batch
        N = z_data.size(0)
        z0_draw = self.model.draw_terminal(N=N)
        if self.mode == "mmd":
            G_z = self.model(z0_draw)
            loss = self.mmd(z_data, G_z)
            self.log("train_mmd", loss)
            return loss
        else:
            D_x = self.disc(z_data)
            s2 = torch.ones_like(z_data)
            G_z = z_data + mvrnorm(z_data, s2)
            D_G_z = self.disc(G_z.detach())
            d_loss = -torch.mean(log_eps(D_x) + log_eps(1 - D_G_z))
            return d_loss

    def validation_step(self, data_batch, batch_idx):
        assert batch_idx == 0, "batch_idx should be 0 in validation_step?"
        z_data = data_batch
        N = z_data.shape[0]
        z0_draw = self.model.draw_terminal(N=N)
        z_gen = self.model(z0_draw)
        mmd = self.mmd(z_data, z_gen)
        self.log("valid_mmd", mmd)
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(z_gen, z_data, mmd, idx_epoch)
        return mmd

    def visualize(self, z_gen, z_data, loss, idx_epoch):
        outdir = self.outdir
        fig_dir = os.path.join(outdir, "figs")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        z_gen = z_gen.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        plot_match(self.model, self.disc, z_gen, z_data, idx_epoch, loss, fig_dir)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr_init,
            weight_decay=self.lr_decay,
        )
        if self.mode == "mmd":
            return opt_g
        else:
            opt_d = torch.optim.Adam(
                self.disc.parameters(),
                lr=self.lr_disc_init,
                weight_decay=self.lr_decay,
            )
            return opt_d

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
