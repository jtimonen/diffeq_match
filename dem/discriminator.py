import os
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from .utils import create_grid_around

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .data import create_dataloader, MyDataset
from .plotting import draw_plot
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
        fake_sigma: float = 1.0,
        batch_size: int = 128,
        n_epochs: int = 800,
        lr: float = 0.005,
        plot_freq=100,
    ):
        z_data = torch.from_numpy(z_data).float()
        learner = DiscriminatorTrainingSetup(
            self,
            z_data=z_data,
            fake_sigma=fake_sigma,
            lr_init=lr,
            batch_size=batch_size,
            plot_freq=plot_freq,
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
        fake_sigma: float,
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
        self.fake_sigma = fake_sigma

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        self.outdir = outdir

    def generate_fake_data(self, z_data):
        s2 = torch.ones_like(z_data)
        z_fake = z_data + self.fake_sigma * mvrnorm(z_data, s2)
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
                self.visualize(z_data, loss, acc, idx_epoch)
        return loss

    def visualize(self, z_data, loss, acc, idx_epoch):
        outdir = self.outdir
        fig_dir = os.path.join(outdir, "figs")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        z_fake = None
        z_data = z_data.detach().cpu().numpy()
        if self.disc.D == 2:
            plot_disc_2d(self.disc, z_fake, z_data, idx_epoch, loss, acc, True, fig_dir)
            plot_disc_2d(
                self.disc, z_fake, z_data, idx_epoch, loss, acc, False, fig_dir
            )
        else:
            raise RuntimeError("Plotting implemented only for D=2")

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


def plot_disc_2d(
    disc,
    z_fake,
    z_data,
    idx_epoch,
    loss,
    acc,
    only_contour=False,
    save_dir=".",
    **kwargs
):
    """Visualize discriminator output."""
    epoch_str = "{0:04}".format(idx_epoch)
    loss_str = "{:.5f}".format(loss)
    acc_str = "{:.5f}".format(acc)
    title = "epoch " + epoch_str + ", loss = " + loss_str + ", acc = " + acc_str
    fn_pre = "c_" if only_contour else "d_"
    fn = fn_pre + epoch_str + ".png"
    S = 30
    u = create_grid_around(z_data, S)
    val = disc.classify_numpy(u)
    X = np.reshape(u[:, 0], (S, S))
    Y = np.reshape(u[:, 1], (S, S))
    Z = np.reshape(val, (S, S))

    plt.figure(figsize=(7.0, 6.5))
    plt.contourf(X, Y, Z)
    plt.colorbar()
    if not only_contour:
        plt.scatter(z_data[:, 0], z_data[:, 1], c="red", marker=".", alpha=0.3)
    if z_fake is not None:
        plt.scatter(z_fake[:, 0], z_fake[:, 1], c="orange", marker=".", alpha=0.3)

    plt.title(title)
    x_min = np.min(z_data) * 1.25
    x_max = np.max(z_data) * 1.25
    plt.xlim(x_min, x_max)
    plt.ylim(x_min, x_max)
    draw_plot(fn, save_dir, **kwargs)
