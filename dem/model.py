import os
import torch
import torch.nn as nn
import numpy as np
from torchdyn.models import NeuralDE
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from .plotting import plot_match
from .math import accuracy

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
        n_hidden: int = 128,
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
        batch_size=128,
        n_epochs: int = 400,
        lr: float = 0.005,
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
        self.disc = disc
        self.mmd = mmd
        self.plot_freq = plot_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    def generate_fake_data(self, z_data):
        N = z_data.size(0)
        z0_draw = self.model.draw_terminal(N=N)
        z_fake = self.model(z0_draw)
        return z_fake, z0_draw

    def training_step(self, data_batch, batch_idx):
        z_data = data_batch
        z_fake, _ = self.generate_fake_data(z_data)
        if self.mode == "mmd":
            loss = self.mmd(z_data, z_fake)
            return loss
        else:
            D_G_z = self.disc(z_fake)
            loss = -torch.mean(log_eps(D_G_z)) + torch.log(self.mmd(z_data, z_fake))
            return loss

    def validation_step(self, data_batch, batch_idx):
        z_data = data_batch
        D_x = self.disc(z_data)  # classify real data
        z_fake, _ = self.generate_fake_data(z_data)
        D_G_z = self.disc(z_fake.detach())  # classify fake data
        loss = -torch.mean(log_eps(D_G_z)) + torch.log(self.mmd(z_data, z_fake))
        val_real = D_x.detach().cpu().numpy().flatten()
        val_fake = D_G_z.detach().cpu().numpy().flatten()
        acc = accuracy(val_real, val_fake)
        mmd = self.mmd(z_fake, z_data)
        self.log("valid_loss", loss)
        self.log("valid_acc", acc)
        self.log("valid_mmd", mmd)
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(z_fake, z_data, mmd, idx_epoch)
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
        )
        return opt_g

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
