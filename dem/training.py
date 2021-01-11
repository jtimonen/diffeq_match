import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

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
        outdir="out"
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
        self.model.visualize(z_data, self.n_draws, self.n_timepoints, loss, idx_epoch,
                             outdir)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.lr_decay
        )

    def train_dataloader(self):
        return self.dataloader
