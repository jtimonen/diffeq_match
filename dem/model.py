import os
import torch
import torch.nn as nn
import numpy as np
import torchdiffeq
import torchsde
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from .plotting import (
    plot_state_2d,
    plot_state_3d,
    plot_state_nd,
    plot_sde_2d,
    plot_sde_3d,
    plot_sde_nd,
)

from .discriminator import Discriminator
from .math import log_eps, KDE
from .data import create_dataloader, MyDataset
from .networks import TanhNetTwoLayer
from .callbacks import MyCallback
from pytorch_lightning.callbacks import ModelCheckpoint


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
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        ln = torch.Tensor([np.log(0.2)]).float()
        self.log_noise = nn.Parameter(ln, requires_grad=True)

    @property
    def diffusion_magnitude(self):
        return torch.exp(self.log_noise)

    def forward(self, t, y):
        return self.f(t, y)

    def f(self, t, y):
        return self.net_f(y)

    def g(self, t, y):
        g = self.diffusion_magnitude
        return g * torch.ones_like(y)


class GenModel(nn.Module):
    """Main model module."""

    def __init__(self, z0, n_hidden: int = 32):
        super().__init__()
        self.n_init = z0.shape[0]
        self.D = z0.shape[1]
        self.field = VectorField(self.D, n_hidden)
        self.field_b = Reverser(self.field)
        self.outdir = os.getcwd()
        self.z0 = torch.tensor(z0).float()
        self.kde = KDE()
        print("Created model with D =", self.D, ", n_init =", self.n_init)

    def set_bandwidth(self, z_data):
        self.kde.set_bandwidth(z_data)

    def draw_init(self, N: int):
        idx = np.random.choice(self.n_init, size=N, replace=True)
        return self.z0[idx, :]

    def traj(self, z_init, ts, sde: bool = False, forward: bool = True):
        if forward:
            if sde:
                return torchsde.sdeint(self.field, z_init, ts, method="euler")
            else:
                return torchdiffeq.odeint_adjoint(
                    self.field, z_init, ts, atol=1e-5, rtol=1e-4
                )
        else:
            if sde:
                raise ValueError("Cannot integrate SDE backward!")
            else:
                return torchdiffeq.odeint_adjoint(self.field_b, z_init, ts)

    def forward(self, N: int):
        ts = torch.linspace(0, 1, N).float()
        z_init = self.draw_init(N)
        z_samp = self.traj(z_init, ts, sde=True, forward=True)
        return torch.transpose(z_samp.diagonal(), 0, 1)

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
        self.set_bandwidth(z_data)
        z_data = torch.from_numpy(z_data).float()
        learner = TrainingSetup(
            self,
            z_data,
            batch_size,
            lr,
            plot_freq,
        )
        save_path = learner.outdir

        checkpoint_callback = ModelCheckpoint(
            verbose=True, monitor="valid_loss", mode="min", prefix="mod"
        )

        trainer = Trainer(
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            default_root_dir=save_path,
            callbacks=[MyCallback(), checkpoint_callback],
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
        # self.disc = disc
        # if disc.D != self.model.D:
        #    raise RuntimeError("Discriminator dimension incompatible with model!")
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_init = lr_init
        self.plot_freq = plot_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    # KDE Stuff
    def kde_terms(self, z_data, z_samples):
        loss1 = -torch.mean(self.model.kde(z_samples, z_data))
        loss2 = -torch.mean(self.model.kde(z_data, z_samples))
        return loss1, loss2

    # GAN STUFF
    # def loss_generator(self, z_samples):
    #    """Rather than training G to minimize log(1 − D(G(z))), we can train G to
    #    maximize log D(G(z)). This objective function results in the same fixed point
    #    of the dynamics of G and D but provides much stronger gradients early in
    #    learning. (Goodfellow et al., 2014)
    #    """
    #    G_z = z_samples
    #    D_G_z = self.disc(G_z)  # classify fake data
    #    loss_fake = -torch.mean(log_eps(D_G_z))
    #    return loss_fake
    #
    # def loss_discriminator(self, z_samples, z_data):
    #    """Discriminator loss."""
    #    D_x = self.disc(z_data)  # classify real data
    #    G_z = z_samples
    #    D_G_z = self.disc(G_z.detach())  # classify fake data
    #   loss_real = -torch.mean(log_eps(D_x))
    #   loss_fake = -torch.mean(log_eps(1 - D_G_z))
    #   loss = 0.5 * (loss_real + loss_fake)
    #   return loss

    def training_step(self, data_batch, batch_idx):
        z_data = data_batch
        N = z_data.size(0)
        z_fake = self.model(N)  # generate fake data
        lt1, lt2 = self.kde_terms(z_data, z_fake)
        return 0.5 * (lt1 + lt2)

        # GAN STUFF
        # if optimizer_idx == 0:
        #    loss = self.loss_generator(z_fake)
        # elif optimizer_idx == 1:
        #    loss = self.loss_discriminator(z_fake, z_data)
        # else:
        #    raise RuntimeError("optimizer_idx must be 0 or 1!")
        # return loss

    def validation_step(self, data_batch, batch_idx):
        z_data = data_batch
        N = z_data.size(0)
        z_fake = self.model(N)  # generate fake data
        lt1, lt2 = self.kde_terms(z_data, z_fake)
        loss = 0.5 * (lt1 + lt2)
        self.log("valid_loss", loss)
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(z_fake, z_data, loss, idx_epoch)
                self.sde_viz(z_data, idx_epoch)
        return loss

    @torch.no_grad()
    def visualize(self, z_samp, z_data, loss, idx_epoch):
        fig_dir = os.path.join(self.outdir, "figs")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        z_samp = z_samp.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        if self.model.D == 2:
            plot_state_2d(self.model, z_samp, z_data, idx_epoch, loss, fig_dir)
        elif self.model.D == 3:
            plot_state_3d(self.model, z_samp, z_data, idx_epoch, loss, fig_dir)
        else:
            plot_state_nd(self.model, z_samp, z_data, idx_epoch, loss, None, fig_dir)

    @torch.no_grad()
    def sde_viz(self, z_data, idx_epoch):
        print(" ")
        print("kde_sigma =", self.model.kde.sigma)
        print("diffusion_magnitude =", self.model.field.diffusion_magnitude)
        N_TRAJ = 30  # number of trajectories
        L_TRAJ = 100  # number of points per trajectory
        fig_dir = os.path.join(self.outdir, "figs")
        z_init = self.model.draw_init(N_TRAJ)
        ts = torch.linspace(0, 1, L_TRAJ).float()
        z_traj = torchsde.sdeint(self.model.field, z_init, ts, method="euler")
        z_traj = z_traj.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        if self.model.D == 2:
            plot_sde_2d(z_data, z_traj, idx_epoch, save_dir=fig_dir)
        elif self.model.D == 3:
            plot_sde_3d(z_data, z_traj, idx_epoch, save_dir=fig_dir)
        else:
            plot_sde_nd(z_data, z_traj, idx_epoch, save_dir=fig_dir)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)
        # opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_init)
        return opt_g

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
