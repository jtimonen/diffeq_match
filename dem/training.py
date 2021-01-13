import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import pytorch_lightning as pl

from .plotting import plot_match


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
        _, _, z_gen = self.model(self.n_draws)
        loss = self.mmd(z_data, z_gen)
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
        lr: float,
        lr_decay: float,
        plot_freq=0,
        outdir="out",
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.lr_decay = lr_decay
        self.disc = disc
        self.plot_freq = plot_freq

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def forward(self, n_draws: int):
        return self.model(n_draws)

    def adversarial_loss(self, y_hat, y):
        return func.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        z_data = batch
        N = z_data.size(0)
        _, _, z_gen = self.model(N)

        # generator
        if optimizer_idx == 0:

            # ground truth result (ie: all fake)
            # put on same device cuz we created this tensor inside training_loop
            valid = torch.ones(N, 1)
            valid = valid.type_as(z_gen)

            # how we can it make discriminator think it is data
            d_gen = self.disc(z_gen)
            g_loss = self.adversarial_loss(d_gen, valid)
            self.log("g_loss", g_loss)
            return g_loss

        # discriminator
        elif optimizer_idx == 1:

            # how well can it label data as data
            valid = torch.ones(N, 1).type_as(z_data)
            d_data = self.disc(z_data)
            real_loss = self.adversarial_loss(d_data, valid)

            # how well can it label as fake?
            fake = torch.zeros(N, 1).type_as(z_gen)
            d_gen = self.disc(z_gen.detach())
            fake_loss = self.adversarial_loss(d_gen, fake)

            # discriminator loss is the average of these
            self.log("real_loss", real_loss)
            self.log("fake_loss", fake_loss)
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss)
            return d_loss
        else:
            raise ValueError("optimizer_idx must be 0 or 1!")

    def visualize(self, z_data, loss, idx_epoch):
        outdir = self.outdir
        fig_dir = os.path.join(outdir, "figs")
        z_data = z_data.detach().cpu().numpy()
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        plot_match(self.model, self.disc, z_data, idx_epoch, loss, fig_dir)

    def configure_optimizers(self):
        lr = self.lr
        b1 = 0.5
        b2 = 0.999
        opt_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return self.dataloader
