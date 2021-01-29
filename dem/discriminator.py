import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer

from .data import create_dataloader, MyDataset
from .plotting import plot_disc
from .math import mvrnorm, log_eps, accuracy
from .networks import LeakyReluNetTwoLayer


class Discriminator(nn.Module):
    """Classifier."""

    def __init__(self, D: int, n_hidden: int = 64):
        super().__init__()
        self.net = LeakyReluNetTwoLayer(D, 1, n_hidden)
        self.D = D

    def forward(self, z):
        z = self.net(z)
        validity = torch.sigmoid(z)
        return validity

    def classify_numpy(self, z):
        z = torch.from_numpy(z).float()
        val = self(z)
        return val.detach().cpu().numpy()

    def fit(
        self,
        z_data,
        batch_size=64,
        n_epochs: int = 100,
        lr: float = 0.005,
        plot_freq=0,
    ):
        learner = DiscriminatorTrainingSetup(
            self,
            z_data,
            lr,
            batch_size,
            plot_freq,
        )
        save_path = learner.outdir
        trainer = Trainer(
            min_epochs=n_epochs,
            max_epochs=n_epochs,
            default_root_dir=save_path,
        )
        trainer.fit(learner)


class DiscriminatorTrainingSetup(pl.LightningModule):
    """Lightning module for discriminator."""

    def __init__(
        self,
        disc: nn.Module,
        z_data,
        lr_init: float,
        batch_size: int,
        plot_freq: int,
        outdir="disc_out",
    ):
        super().__init__()
        ds = MyDataset(z_data)
        num_workers = 0
        train_loader = create_dataloader(ds, batch_size, num_workers, shuffle=True)
        valid_loader = create_dataloader(ds, None, num_workers, shuffle=False)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.N = len(train_loader.dataset)
        self.disc = disc
        self.lr_init = lr_init
        self.plot_freq = plot_freq
        self.sigma = 0.5

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def generate_fake_data(self, z_data):
        s2 = torch.ones_like(z_data)
        z_fake = z_data + self.sigma * mvrnorm(z_data, s2)
        return z_fake

    def training_step(self, data_batch, batch_idx):
        z_data = data_batch
        D_x = self.disc(z_data)  # classify real data
        G_z = self.generate_fake_data(z_data)
        D_G_z = self.disc(G_z.detach())  # classify fake data
        loss_real = -torch.mean(log_eps(D_x))
        loss_fake = -torch.mean(log_eps(1 - D_G_z))
        loss = 0.5 * (loss_real + loss_fake)
        return loss

    def validation_step(self, data_batch, batch_idx):
        z_data = data_batch
        D_x = self.disc(z_data)  # classify real data
        z_fake = self.generate_fake_data(z_data)
        D_G_z = self.disc(z_fake.detach())  # classify fake data
        loss_real = -torch.mean(log_eps(D_x))
        loss_fake = -torch.mean(log_eps(1 - D_G_z))
        loss = 0.5 * (loss_real + loss_fake)
        val_real = D_x.detach().cpu().numpy().flatten()
        val_fake = D_G_z.detach().cpu().numpy().flatten()
        acc = accuracy(val_real, val_fake)
        self.log("valid_loss", loss)
        self.log("valid_acc", acc)
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(z_fake, z_data, loss, idx_epoch)
        return loss

    def visualize(self, z_fake, z_data, loss, idx_epoch):
        outdir = self.outdir
        fig_dir = os.path.join(outdir, "figs")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        z_fake = z_fake.detach().cpu().numpy()
        z_data = z_data.detach().cpu().numpy()
        plot_disc(self.disc, z_fake, z_data, idx_epoch, loss, fig_dir)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.disc.parameters(),
            lr=self.lr_init,
        )
        return opt

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
