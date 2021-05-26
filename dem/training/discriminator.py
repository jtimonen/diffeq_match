import os
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from .learner import Learner
from .setup import TrainingSetup, run_training
from dem.modules.discriminator import Discriminator, KdeDiscriminator
from dem.data.dataset import NumpyDataset
from dem.utils.math import mvrnorm
from dem.utils.utils import tensor_to_numpy
from dem.utils.classification import accuracy, create_classification
from dem.plotting import plot_disc_2d


def train_occ(
    discriminator: Discriminator,
    x: np.ndarray,
    noise_scale: float = 1.0,
    **training_setup_kwargs
):
    """Train a binary classifier using data from only one class.

    :param discriminator: The discriminator to train.
    :param x: Numpy array of shape (N, D), containing N observations belonging
    to class 1.
    :param noise_scale: Standard deviation of noise added to x to generate fake data.
    """
    ds = NumpyDataset(x)
    setup = TrainingSetup(ds, **training_setup_kwargs)
    occ = UnaryClassification(discriminator, setup, noise_scale)
    trainer = run_training(occ, setup.n_epochs, setup.outdir)
    return occ, trainer


class UnaryClassification(Learner):
    def __init__(
        self,
        discriminator: Discriminator,
        setup: TrainingSetup,
        noise_scale: float,
    ):
        super().__init__(setup)  # set dataloaders etc
        self.discriminator = discriminator
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

    def forward(self, data_batch):
        x_real, x_noisy = self.create_x(data_batch)
        x, y_target = create_classification(x_real, x_noisy)
        y_pred = self.discriminator(x)  # classify
        loss = binary_cross_entropy(y_pred, y_target, reduction="mean")
        return x, y_target, y_pred, loss

    def update_kde(self):
        data = self.whole_trainset()
        x_real, x_noisy = self.create_x(data)
        self.discriminator.update(x0=x_noisy, x1=x_real)

    def training_step(self, data_batch, batch_idx):
        _, _, _, loss = self.forward(data_batch)
        return loss

    def validation_step(self, data_batch, batch_idx):
        x, y_target, y_pred, loss = self.forward(data_batch)
        y_target = tensor_to_numpy(y_target).astype(np.uint8)
        y_pred = tensor_to_numpy(y_pred)
        x = tensor_to_numpy(x)
        acc = accuracy(y_target, y_pred, prob=True)
        self.log_metrics(loss.item(), acc)
        pf = self.plot_freq
        if (pf > 0) and (self.current_epoch % pf == 0):
            self.visualize(x, y_target)
        return loss

    def log_metrics(self, loss: float, acc: float):
        self.log("valid_loss", loss)
        self.log("valid_accuracy", acc)
        if self.involves_kde:
            self.log("bandwidth", self.discriminator.kde.bw)

    def configure_optimizers(self):
        return torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

    def visualize(self, x, y_target):
        fn = self.create_figure_name()
        title = self.epoch_str()
        if self.involves_kde:
            title = title + (", bw = %1.4f" % self.discriminator.kde.bw)
        fig_dir = os.path.join(self.outdir, "figs")
        if self.discriminator.D == 2:
            plot_disc_2d(
                self.discriminator,
                x,
                y_target,
                save_name=fn,
                save_dir=fig_dir,
                title=title,
                contour=self.plot_contour,
                points=self.plot_points,
            )
        else:
            pass
