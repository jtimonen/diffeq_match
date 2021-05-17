from pytorch_lightning import Trainer, LightningModule
from torch.utils.data.dataset import random_split


def run_training(module: LightningModule, n_epochs: int, outdir):
    trainer = Trainer(max_epochs=n_epochs, min_epochs=n_epochs, default_root_dir=outdir)
    if n_epochs > 0:
        trainer.fit(module)
    return trainer


def create_split(dataset, p_valid):
    if p_valid is None:
        train_dataset = dataset
        valid_dataset = dataset
    else:
        N = len(dataset)
        N_valid = int(p_valid * N)
        split = random_split(dataset, lengths=[N - N_valid, N_valid])
        train_dataset = split[0]
        valid_dataset = split[1]
    return train_dataset, valid_dataset


class TrainingSetup:
    def __init__(
        self,
        dataset,
        lr: float = 0.005,
        n_epochs: int = 400,
        outdir="out",
        batch_size: int = 64,
        num_workers: int = 0,
        plot_freq: int = 0,
        lr_disc=None,
        p_valid=None,
        pin_memory=False,
    ):
        if lr_disc is None:
            lr_disc = lr
        self.train_dataset, self.valid_dataset = create_split(dataset, p_valid)
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_disc = lr_disc
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.outdir = outdir
        self.plot_freq = plot_freq
        self.pin_memory = pin_memory

    def __repr__(self):
        desc = "<TrainingSetup (lr=%1.4f, n_epochs=%d)>" % (self.lr, self.n_epochs)
        return desc
