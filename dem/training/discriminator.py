import torch

from .learner import Learner

import numpy as np
from pytorch_lightning import Trainer
from dem.modules.discriminator import Discriminator
from dem.data.dataset import NumpyDataset
from dem.utils.math import mvrnorm


def train_occ(
    model: Discriminator,
    x: np.ndarray,
    batch_size=128,
    n_epochs: int = 400,
    lr: float = 0.005,
    plot_freq=0,
    outdir="out",
    noise_scale: float = 0.5,
):
    ds = NumpyDataset(x)
    occ = UnaryClassification(
        model=model,
        train_dataset=ds,
        valid_dataset=ds,
        batch_size=batch_size,
        lr=lr,
        plot_freq=plot_freq,
        outdir=outdir,
        noise_scale=noise_scale,
    )
    trainer = Trainer(
        min_epochs=n_epochs, max_epochs=n_epochs, default_root_dir=occ.outdir
    )
    trainer.fit(occ)
    return occ


class UnaryClassification(Learner):
    def __init__(
        self,
        model: Discriminator,
        train_dataset: NumpyDataset,
        valid_dataset: NumpyDataset,
        batch_size: int,
        lr: float,
        plot_freq: int,
        outdir,
        noise_scale: float,
    ):
        super().__init__(
            train_dataset, valid_dataset, batch_size, None, plot_freq, outdir
        )
        self.model = model
        self.lr = lr
        self.noise_scale = noise_scale

    @property
    def is_kde(self):
        return hasattr(self.model, "kde")

    def generate_noisy_data(self, x: torch.Tensor):
        scale = self.noise_scale * torch.ones_like(x)
        x_noisy = mvrnorm(x, scale)
        return x_noisy

    def loss(self, x_real: torch.Tensor):
        x_noisy = self.generate_noisy_data(x_real)  # float32 tensor with size [B, D]
        if self.is_kde:
            self.model.set_data(x0=x_noisy, x1=x_real)
        log_p1_real = self.model(x_real, log=True)  # float32 tensor with size [B]
        log_p1_noisy = self.model(x_noisy, log=True)  # float32 tensor with size [B]
        val = -log_p1_real - torch.expm1(-log_p1_noisy)
        return val.mean()

    def training_step(self, data_batch, batch_idx):
        return self.loss(data_batch)

    def validation_step(self, data_batch, batch_idx):
        loss = self.loss(data_batch)
        loss_str = loss.item()

        pf = self.plot_freq
        if (pf > 0) and (self.current_epoch % pf == 0):
            self.visualize(data_batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def visualize(self, *args):
        if self.model.D == 2:
            self.visualize_2d()
        else:
            pass

    def visualize_2d(self):
        x = self.valid
