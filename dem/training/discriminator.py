import torch

from .learner import Learner

import numpy as np
from pytorch_lightning import Trainer
from .callbacks import MyCallback
from dem.modules.discriminator import Discriminator
from dem.data.dataset import ClassificationDataset


def train_discriminator(
    model: Discriminator,
    x: np.ndarray,
    labels: np.ndarray,
    batch_size=128,
    n_epochs: int = 400,
    lr: float = 0.005,
    plot_freq=0,
    outdir="out",
):
    ds = ClassificationDataset(x, labels)
    learner = DiscriminatorLearner(
        model=model,
        train_dataset=ds,
        valid_dataset=ds,
        batch_size=batch_size,
        lr=lr,
        plot_freq=plot_freq,
        outdir=outdir,
    )
    trainer = Trainer(
        min_epochs=n_epochs, max_epochs=n_epochs, default_root_dir=learner.outdir
    )
    trainer.fit(learner)
    return learner


class DiscriminatorLearner(Learner):
    def __init__(
        self,
        model: Discriminator,
        train_dataset: ClassificationDataset,
        valid_dataset: ClassificationDataset,
        batch_size: int,
        lr: float,
        plot_freq: int,
        outdir,
    ):
        super().__init__(
            train_dataset, valid_dataset, batch_size, None, plot_freq, outdir
        )
        self.model = model
        self.lr = lr

    def training_step(self, x, labels, batch_idx):
        y_pred = self.model(x, labels)
        print(y_pred)

    def validation_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def visualize(self, *args):
        return 0
