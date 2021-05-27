import os
import abc
import pytorch_lightning as pl
import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from dem.data.dataloader import create_dataloader
from dem.utils import read_logged_scalar, read_logged_events
from dem.modules import GenerativeModel
from dem.modules.discriminator import Discriminator
from dem.utils.utils import tensor_to_numpy
from dem.training.setup import TrainingSetup
from dem.plotting.discriminator import plot_disc_2d
from dem.plotting.training import plot_gan_progress


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

    def whole_trainset(self):
        batches = [batch for batch in self.train_loader]
        return torch.cat(batches, dim=0)

    def whole_validset(self):
        batches = [batch for batch in self.valid_loader]
        return torch.cat(batches, dim=0)

    def is_plot_epoch(self):
        return self.current_epoch % self.plot_freq == 0

    @property
    def involves_kde(self):
        return self.discriminator.is_kde

    def on_fit_start(self) -> None:
        print("Training started.")
        if self.involves_kde:
            self.update_kde()

    def on_epoch_start(self) -> None:
        if self.involves_kde:
            self.update_kde()

    def on_fit_end(self) -> None:
        print("Training done.")
        if self.involves_kde:
            self.update_kde()

    def set_outdir(self, path):
        """Set output directory."""
        if not os.path.isdir(path):
            os.mkdir(path)
            print("Created output directory '" + path + "'")
        else:
            raise RuntimeError("Output directory '" + path + "' already exists!")
        self.outdir = path
        print("Set output directory to '" + path + "'")

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


class AdversarialLearner(Learner):
    def __init__(
        self,
        model: GenerativeModel,
        discriminator: Discriminator,
        setup: TrainingSetup,
    ):
        super().__init__(setup)
        self.model = model
        self.discriminator = discriminator
        self.lr = setup.lr
        self.lr_disc = setup.lr_disc
        self.n_epochs = setup.n_epochs
        self.betas = (setup.b1, setup.b2)
        self.setup_desc = setup.__repr__()

    def __repr__(self):
        desc = type(self).__name__
        desc += "\n  " + self.model.__repr__()
        desc += "\n  " + self.discriminator.__repr__()
        desc += "\n  " + self.setup_desc
        return desc

    def configure_optimizers(self):
        raise NotImplementedError

    def loss_generator(self, gen_batch):
        """
        Rather than training G to minimize `log(1 − D(G(z)))`, we can train G to
        maximize `log D(G(z))`. This objective function results in the same fixed point
        of the dynamics of `G` and `D` but provides much stronger gradients early in
        learning. (Goodfellow et al., 2014)
        """
        target_fool = torch.ones(gen_batch.size(0), 1).type_as(gen_batch)
        return binary_cross_entropy(self.discriminator(gen_batch), target_fool)

    def loss_discriminator(self, data_batch, gen_batch):
        """Discriminator loss."""
        target_data = torch.ones(data_batch.size(0), 1).type_as(data_batch)
        target_gen = torch.zeros(gen_batch.size(0), 1).type_as(gen_batch)
        d_data = self.discriminator(data_batch)
        loss_data = binary_cross_entropy(d_data, target_data)
        d_gen = self.discriminator(gen_batch.detach())
        loss_gen = binary_cross_entropy(d_gen, target_gen)
        return 0.5 * (loss_data + loss_gen)

    def accuracy(self, data_batch, gen_batch):
        d_data = self.discriminator(data_batch)
        d_gen = self.discriminator(gen_batch)
        d_data = tensor_to_numpy(d_data).ravel()
        d_gen = tensor_to_numpy(d_gen).ravel()
        n_correct = sum(d_data > 0.5) + sum(d_gen <= 0.5)
        return n_correct / (len(d_data) + len(d_gen))

    @torch.no_grad()
    def update_kde(self):
        data = self.whole_trainset()
        gen = self.model(N=data.shape[0], like=data)
        self.discriminator.update(x0=gen, x1=data)

    def on_epoch_end(self):
        g_loss, d_loss, acc = self.validate_model()
        self.log("g_loss", g_loss)
        self.log("d_loss", d_loss)
        self.log("accuracy", acc)
        if self.involves_kde:
            self.log("kde_bandwidth", self.discriminator.kde.bw)

    def on_fit_end(self) -> None:
        print("Training done.")
        if self.involves_kde:
            self.update_kde()
        force_plot = self.plot_freq > 0
        self.validate_model(force_plot=force_plot)

    def validate_model(self, force_plot=False):
        data = self.whole_validset()
        gen = self.model(N=data.shape[0], like=data)
        g_loss = self.loss_generator(gen)
        d_loss = self.loss_discriminator(data, gen)
        acc = self.accuracy(data, gen)
        if self.is_plot_epoch() or force_plot:
            self.visualize(data, gen, g_loss, d_loss, acc)
        return g_loss, d_loss, acc

    @torch.no_grad()
    def visualize(self, data, gen, g_loss, d_loss, acc):
        data = tensor_to_numpy(data)
        N = data.shape[0]
        fn = self.create_figure_name(prefix="model")
        self.model.visualize(
            N=N, data=data, epoch=self.current_epoch, save_name=fn, save_dir=self.outdir
        )
        self.visualize_disc(data, gen, acc)
        if self.current_epoch > 0:
            self.visualize_training()

    def visualize_disc(self, data, gen, acc):
        title = "epoch = %d, accuracy=%1.4f" % (self.current_epoch, acc)
        x = np.vstack((gen, data))
        y_target = np.array(gen.shape[0] * [0] + data.shape[0] * [1])
        fn = self.create_figure_name(prefix="disc")
        sd = self.outdir
        if self.model.D == 2:
            plot_disc_2d(
                self.discriminator,
                x,
                y_target,
                save_name=fn,
                save_dir=sd,
                title=title,
                contour=True,
                points=True,
                gan_mode=True,
            )

    def visualize_training(self, version: int = 0):
        try:
            g_loss = self.read_logged_scalar(name="g_loss", version=version)
            d_loss = self.read_logged_scalar(name="d_loss", version=version)
            acc = self.read_logged_scalar(name="accuracy", version=version)
            if self.involves_kde:
                bw = self.read_logged_scalar(name="kde_bandwidth", version=version)
            else:
                bw = None
        except FileNotFoundError:
            print("Unable to read logged scalars. FileNotFoundError caught.")
            return None
        fn = "progress.png"
        plot_gan_progress(g_loss, d_loss, acc, bw, save_name=fn, save_dir=self.outdir)
