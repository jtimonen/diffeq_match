import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import pytorch_lightning as pl
from .plotting import plot_match


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        valid_loader,
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

    def adversarial_loss(self, y_hat, y):
        return func.binary_cross_entropy(y_hat, y)

    def log_eps(self, x):
        return torch.log(x + 1e-8)

    def training_step(self, data_batch, batch_idx, optimizer_idx=0):
        z_data = data_batch
        N = z_data.size(0)
        z0_draw = self.model.draw_terminal(N=N)
        if self.mode == "mmd":
            G_z = self.model(z0_draw)
            z_data_whole = self.train_loader.dataset.z
            loss = self.mmd(z_data_whole, G_z)
            self.log("train_mmd", loss)
            return torch.log(loss)
        else:
            # generator
            if optimizer_idx == 0:
                G_z = self.model(z0_draw)
                D_G_z = self.disc(G_z)
                g_loss = torch.mean(-self.log_eps(D_G_z))
                return g_loss

            # discriminator
            elif optimizer_idx == 1:
                D_x = self.disc(z_data)
                G_z = self.model(z0_draw)
                D_G_z = self.disc(G_z.detach())
                d_loss = -torch.mean(self.log_eps(D_x) + self.log_eps(1 - D_G_z))
                return d_loss
            else:
                raise ValueError("optimizer_idx must be 0 or 1!")

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
            return [opt_g, opt_d], []

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
