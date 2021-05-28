from pytorch_lightning import Trainer, LightningModule
from torch.utils.data.dataset import random_split
import warnings


def run_training(module: LightningModule, n_epochs: int, outdir):
    trainer = Trainer(max_epochs=n_epochs, min_epochs=n_epochs, default_root_dir=outdir)
    ver = trainer.logger.version
    module.set_version(ver)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore", category=UserWarning, message="The dataloader,"
        )
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
            message="you passed in a val_dataloader but have no validation_step",
        )
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
        lr: float = 0.001,
        n_epochs: int = 400,
        outdir="out",
        batch_size: int = 256,
        num_workers: int = 0,
        plot_freq: int = 0,
        lr_disc=None,
        p_valid=None,
        pin_memory=False,
        b1: float = 0.9,
        b2: float = 0.999,
        weight_decay: float = 1e-5,
    ):
        if lr_disc is None:
            lr_disc = lr
        self.train_dataset, self.valid_dataset = create_split(dataset, p_valid)
        L = len(self.train_dataset)
        if batch_size > L:
            msg = (
                "Setting batch_size to training data set size (%d), "
                "because specified value (%d) was larger" % (L, batch_size)
            )
            batch_size = L
            print(msg)
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_disc = lr_disc
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.outdir = outdir
        self.plot_freq = plot_freq
        self.pin_memory = pin_memory
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay

    def __repr__(self):
        desc = "* TrainingSetup: lr=%s, n_epochs=%d, batch_size=%d, weight_decay=%s" % (
            "{:.2e}".format(self.lr),
            self.n_epochs,
            self.batch_size,
            "{:.2e}".format(self.weight_decay),
        )
        return desc
