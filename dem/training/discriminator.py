import torch
import os

from .learner import Learner
from .setup import TrainingSetup, run_training

import numpy as np
from dem.modules.discriminator import Discriminator
from dem.data.dataset import NumpyDataset
from dem.utils.math import mvrnorm
from dem.utils.utils import accuracy
from dem.plotting import plot_disc_2d


def train_occ(
    model: Discriminator,
    x: np.ndarray,
    noise_scale: float = 1.0,
    **training_setup_kwargs
):
    ds = NumpyDataset(x)
    setup = TrainingSetup(ds, None, **training_setup_kwargs)
    occ = UnaryKDEClassification(
        model=model,
        setup=setup,
        noise_scale=noise_scale,
    )
    return run_training(occ, setup.n_epochs, setup.outdir)


class UnaryKDEClassification(Learner):
    def __init__(
        self,
        model: Discriminator,
        setup: TrainingSetup,
        noise_scale: float,
    ):
        super().__init__(setup)
        self.model = model
        self.lr = setup.lr
        self.n_epochs = setup.n_epochs
        self.noise_scale = noise_scale
        self.plot_contour = True
        self.plot_points = True

    def generate_noisy_data(self, x: torch.Tensor):
        scale = self.noise_scale * torch.ones_like(x)
        x_noisy = mvrnorm(x, scale)
        return x_noisy

    def create_x(self, data_batch):
        x_real = data_batch
        x_noisy = self.generate_noisy_data(x_real)  # float32 tensor with size [B, D]
        return x_real, x_noisy

    @staticmethod
    def target_labels(y_pred):
        N = int(len(y_pred) / 2)
        y_target = np.array(N * [1] + N * [0], dtype=np.uint8)
        return y_target

    def forward(self, data_batch):
        x_real, x_noisy = self.create_x(data_batch)
        self.model.set_data(x0=x_noisy, x1=x_real)
        log_p1_real = self.model(x_real, log=True)  # float32 tensor with size [B]
        log_p1_noisy = self.model(x_noisy, log=True)  # float32 tensor with size [B]
        loss = log_p1_real + torch.expm1(-log_p1_noisy)
        x = torch.vstack((x_real, x_noisy)).cpu().detach().numpy()
        y = torch.cat((log_p1_real, log_p1_noisy))
        y = torch.exp(y).cpu().detach().numpy()
        return x, y, loss.mean()

    def training_step(self, data_batch, batch_idx):
        _, _, loss = self.forward(data_batch)
        return loss

    def validation_step(self, data_batch, batch_idx):
        x, y_pred, loss = self.forward(data_batch)
        y_target = self.target_labels(y_pred)
        acc = accuracy(y_target, y_pred, prob=True)
        self.log("valid_loss", loss)
        self.log("valid_accuracy", acc)
        self.log("bandwidth", self.model.kde.bw)
        pf = self.plot_freq
        if (pf > 0) and (self.current_epoch % pf == 0):
            self.visualize(x, y_pred)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def visualize(self, x, y_pred):
        fn = self.create_figure_name()
        y_true = self.target_labels(y_pred)
        title = self.epoch_str() + (", bw = %1.4f" % self.model.kde.bw)
        fig_dir = os.path.join(self.outdir, "figs")
        if self.model.D == 2:
            plot_disc_2d(
                self.model,
                x,
                y_true,
                save_name=fn,
                save_dir=fig_dir,
                title=title,
                contour=self.plot_contour,
                points=self.plot_points,
            )
        else:
            pass
