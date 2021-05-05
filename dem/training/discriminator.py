import numpy as np
import pytorch_lightning as pl
from dem.data.dataloader import create_dataloader

import torch
import torch.nn as nn

from pytorch_lightning import Trainer

from dem.data.dataset import ClassificationDataset


def train_discriminator(
    model: nn.Module,
    z_data,
    batch_size=128,
    n_epochs: int = 400,
    lr: float = 0.005,
    plot_freq=0,
):
    z_data = torch.from_numpy(z_data).float()
    learner = TrainingSetup(
        self,
        z_data,
        batch_size,
        lr,
        plot_freq,
    )
    save_path = learner.outdir

    trainer = Trainer(
        min_epochs=n_epochs,
        max_epochs=n_epochs,
        default_root_dir=save_path,
        callbacks=[MyCallback(), checkpoint_callback],
    )
    trainer.fit(learner)
