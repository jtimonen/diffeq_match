import torch
import torch.nn as nn
import pytorch_lightning as pl
from .utils import reshape_traj


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        dataloader,
        mmd: nn.Module,
        n_draws: int,
        n_timepoints: int,
        lr: float,
        lr_decay: float,
        out_dir=None,
        draw_freq=10,
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.lr_decay = lr_decay
        self.mmd = mmd
        self.n_draws = n_draws
        self.n_timepoints = n_timepoints
        self.out_dir = out_dir
        self.draw_freq = draw_freq

    def forward(self, n_draws: int):
        return self.model(n_draws)

    def training_step(self, batch, batch_idx):
        z_data = batch
        z0, t, z_traj = self.model(self.n_draws, self.n_timepoints)
        z = reshape_traj(z_traj)
        loss = self.mmd(z_data, z)
        self.log("train_loss", loss)
        self.log("z0_log_sigma", self.model.z0_log_sigma)
        idx_epoch = self.current_epoch

        if idx_epoch % self.draw_freq == 0:
            self.visualize(z_data, loss.item(), idx_epoch)

        return loss

    def visualize(self, z_data, loss, idx_epoch):
        self.model.visualize(
            z_data, self.n_draws, self.n_timepoints, loss, self.out_dir, idx_epoch
        )

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.lr_decay
        )

    def train_dataloader(self):
        return self.dataloader
