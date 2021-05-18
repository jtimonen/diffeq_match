import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adam import Adam

from dem.modules import GenModel, KdeDiscriminator, NeuralDiscriminator
from dem.modules.discriminator import Discriminator
from dem.data.dataloader import create_dataloader
from dem.data.dataset import NumpyDataset
from .setup import TrainingSetup, run_training
from .learner import Learner


def train_model(
    model: GenModel,
    disc: Discriminator,
    data: np.ndarray,
    gen: np.ndarray,
    **training_setup_kwargs
):
    """Train a model.

    :param model: model to be trained
    :param disc: discriminator
    :param data: a numpy array with shape (num_obs, num_dims)
    :param gen: a numpy array with shape (num_gen, num_dims)
    :param training_setup_kwargs: keyword arguments to `TrainingSetup`
    """
    ds = NumpyDataset(data)
    setup = TrainingSetup(ds, **training_setup_kwargs)
    ds_gen = NumpyDataset(gen)
    genloader = create_dataloader(
        ds_gen,
        batch_size=setup.batch_size,
        num_workers=setup.num_workers,
        pin_memory=setup.pin_memory,
        genloader=True,
    )
    if disc is None:
        disc = NeuralDiscriminator(D=model.D)
    occ = GAN(model, disc, setup, genloader)
    trainer = run_training(occ, setup.n_epochs, setup.outdir)
    return occ, trainer


class GAN(Learner):
    def __init__(
        self,
        model: nn.Module,
        disc: Discriminator,
        setup: TrainingSetup,
        genloader,
    ):
        super().__init__(setup)
        self.model = model
        self.disc = disc
        self.lr = setup.lr
        self.lr_disc = setup.lr_disc
        self.n_epochs = setup.n_epochs
        self.genloader = genloader
        self.betas = (setup.b1, setup.b2)

    def configure_optimizers(self):
        opt_g = Adam(self.model.parameters(), lr=self.lr, betas=self.betas)
        opt_d = Adam(self.disc.parameters(), lr=self.lr_disc, betas=self.betas)
        return [opt_g, opt_d], []

    # KDE Stuff
    def kde_terms(self, z_data, z_samples):
        loss1 = -torch.mean(self.model.kde(z_samples, z_data))
        loss2 = -torch.mean(self.model.kde(z_data, z_samples))
        return loss1, loss2

    # GAN STUFF
    def loss_generator(self, z_samples):
        """
        Rather than training G to minimize `log(1 − D(G(z)))`, we can train G to
        maximize `log D(G(z))`. This objective function results in the same fixed point
        of the dynamics of `G` and `D` but provides much stronger gradients early in
        learning. (Goodfellow et al., 2014)
        """
        G_z = z_samples
        D_G_z = self.disc(G_z)  # classify fake data
        loss_fake = -torch.mean(log_eps(D_G_z))
        return loss_fake

    def loss_discriminator(self, z_samples, z_data):
        """Discriminator loss."""
        D_x = self.disc(z_data)  # classify real data
        G_z = z_samples
        D_G_z = self.disc(G_z.detach())  # classify fake data
        loss_real = -torch.mean(log_eps(D_x))
        loss_fake = -torch.mean(log_eps(1 - D_G_z))
        loss = 0.5 * (loss_real + loss_fake)
        return loss

    def training_step(self, data_batch, batch_idx, optim_idx):
        if optim_idx == 0:
            self.generator_step(data_batch, batch_idx)
        if optim_idx == 1:
            self.discriminator_step(data_batch, batch_idx)
        z_data = data_batch
        N = z_data.size(0)
        z_fake = self.model(N)  # generate fake data
        lt1, lt2 = self.kde_terms(z_data, z_fake)
        return 0.5 * (lt1 + lt2)

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
    def check_model(self, z_data):
        z_data = torch.from_numpy(z_data).float()
        N = z_data.size(0)
        z_fake = self.model(N)  # generate fake data
        lt1, lt2 = self.kde_terms(z_data, z_fake)
        loss = 0.5 * (lt1 + lt2)
        print("valid_loss", loss)
        idx_epoch = -1
        self.visualize(z_fake, z_data, loss, idx_epoch)
        self.sde_viz(z_data, idx_epoch)

    @torch.no_grad()
    def generate_traj(self, N: int = 30, z_init=None):
        """Returns tensor of shape (L, N, D)."""
        L = 100
        ts = torch.linspace(0, 1, L).float()
        if z_init is None:
            z_init = self.model.draw_init(N)
        z_traj = self.model.traj(z_init, ts, sde=True, forward=True)
        return z_traj

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
            plot_sde_2d(self.model, z_data, z_traj, idx_epoch, save_dir=fig_dir)
        elif self.model.D == 3:
            plot_sde_3d(self.model, z_data, z_traj, idx_epoch, save_dir=fig_dir)
        else:
            plot_sde_nd(self.model, z_data, z_traj, idx_epoch, save_dir=fig_dir)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)
        # opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr_init)
        return opt_g

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
