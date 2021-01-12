import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import pytorch_lightning as pl
from collections import OrderedDict

from .utils import reshape_traj


class MMDLearner(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dataloader,
        mmd: nn.Module,
        n_draws: int,
        n_timepoints: int,
        lr: float,
        lr_decay: float,
        draw_freq=10,
        outdir="out",
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.lr_decay = lr_decay
        self.mmd = mmd
        self.n_draws = n_draws
        self.n_timepoints = n_timepoints
        self.draw_freq = draw_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    def training_step(self, batch, batch_idx):
        z_data = batch
        z0, t, z_traj = self.model(self.n_draws, self.n_timepoints)
        z = reshape_traj(z_traj)
        loss = self.mmd(z_data, z)
        self.log("train_loss", loss)
        self.log("z0_log_sigma", self.model.z0_log_sigma)
        return loss

    def visualize(self, z_data, loss, idx_epoch):
        outdir = self.outdir
        self.model.visualize(
            z_data, self.n_draws, self.n_timepoints, loss, idx_epoch, outdir
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.lr_decay
        )

    def train_dataloader(self):
        return self.dataloader


class GANLearner(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dataloader,
        disc: nn.Module,
        n_draws: int,
        n_timepoints: int,
        lr: float,
        lr_decay: float,
        draw_freq=10,
        outdir="out",
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.lr_decay = lr_decay
        self.disc = disc
        self.n_draws = n_draws
        self.n_timepoints = n_timepoints
        self.draw_freq = draw_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    def adversarial_loss(self, y_hat, y):
        return func.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        z_data = batch
        z0, t, z_traj = self.model(self.n_draws, self.n_timepoints)
        z = reshape_traj(z_traj)

        # generator
        if optimizer_idx == 0:

            # ground truth result (ie: all fake)
            # put on same device cuz we created this tensor inside training_loop
            valid = torch.ones(z.size(0), 1)
            valid = valid.type_as(z)

            # how we can it make discriminator think it is data
            g_loss = self.adversarial_loss(self.disc(z), valid)
            return g_loss

        # discriminator
        elif optimizer_idx == 1:

            # how well can it label data as data
            valid = torch.ones(z_data.size(0), 1).type_as(z_data)
            real_loss = self.adversarial_loss(self.disc(z_data), valid)

            # how well can it label as fake?
            fake = torch.zeros(z.size(0), 1).type_as(z)

            fake_loss = self.adversarial_loss(self.disc(z.detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            return d_loss
        else:
            raise ValueError("optimizer_idx must be 0 or 1!")

    def visualize(self, z_data, loss, idx_epoch):
        outdir = self.outdir
        self.model.visualize(
            z_data, self.n_draws, self.n_timepoints, loss, idx_epoch, outdir
        )

    def configure_optimizers(self):
        lr = self.lr
        b1 = 0.5
        b2 = 0.999
        opt_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return self.dataloader
