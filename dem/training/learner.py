import os
import abc
import pytorch_lightning as pl
from dem.data.dataloader import create_dataloader
from .setup import TrainingSetup
from dem.utils import read_logged_scalar, read_logged_events


class Learner(pl.LightningModule, abc.ABC):
    def __init__(self, setup: TrainingSetup):
        super().__init__()
        train_loader = create_dataloader(
            setup.train_dataset,
            shuffle=True,
            batch_size=setup.batch_size,
            num_workers=setup.num_workers,
            pin_memory=setup.pin_memory,
        )
        valid_loader = create_dataloader(
            setup.valid_dataset,
            shuffle=False,
            batch_size=None,
            num_workers=setup.num_workers,
            pin_memory=setup.pin_memory,
        )
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.plot_freq = setup.plot_freq
        self.outdir = setup.outdir
        self.set_outdir(setup.outdir)

    @property
    def num_train(self):
        return len(self.train_loader.dataset)

    @property
    def num_valid(self):
        return len(self.valid_loader.dataset)

    def set_outdir(self, path):
        """Set output directory."""
        if not os.path.isdir(path):
            os.mkdir(path)
            print("Created output directory " + path)
        else:
            raise RuntimeError("Output directory (" + path + ") exists!")
        self.outdir = path
        print("Set output directory to " + path)

    def configure_optimizers(self):
        raise NotImplementedError

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def epoch_str(self, digits: int = 4):
        return str(self.current_epoch).zfill(digits)

    def create_figure_name(self, prefix: str = "fig", ext: str = ".png"):
        fn = prefix + self.epoch_str() + ext
        return fn

    def read_logged_scalar(self, name="valid_loss", version: int = 0):
        """Read a logged scalar."""
        df = read_logged_scalar(name=name, parent_dir=self.outdir, version=version)
        return df

    def read_logged_events(self, version: int = 0):
        """Read the event accumulator."""
        ea = read_logged_events(parent_dir=self.outdir, version=version)
        return ea
