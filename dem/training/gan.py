import os

import numpy as np
import torch
from torch.optim.adam import Adam
from torch.nn.functional import binary_cross_entropy

from dem.modules import GenerativeModel, KdeDiscriminator, NeuralDiscriminator
from dem.modules.discriminator import Discriminator
from dem.data.dataset import NumpyDataset
from dem.utils.utils import tensor_to_numpy
from .setup import TrainingSetup, run_training
from .learner import Learner
from dem.plotting.discriminator import plot_disc_2d
from dem.plotting.training import plot_gan_progress


def train_model(
    model: GenerativeModel,
    disc: Discriminator,
    data: np.ndarray,
    verbose: bool = True,
    **training_setup_kwargs
):
    """Train a model.

    :param model: model to be trained
    :param disc: discriminator
    :param data: a numpy array with shape (num_obs, num_dims)
    :param verbose: print more information?
    :param training_setup_kwargs: keyword arguments to `TrainingSetup`
    """
    ds = NumpyDataset(data)
    setup = TrainingSetup(ds, **training_setup_kwargs)
    if disc is None:
        disc = NeuralDiscriminator(D=model.D)
    occ = GAN(model, disc, setup)
    if verbose:
        print("")
        print(model)
        print(disc)
        print(setup)
        print("")
    trainer = run_training(occ, setup.n_epochs, setup.outdir)
    return occ, trainer


class GAN(Learner):
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

    def configure_optimizers(self):
        pars_g = self.model.parameters()
        pars_d = self.discriminator.parameters()
        opt_g = Adam(pars_g, lr=self.lr, betas=self.betas)
        opt_d = Adam(pars_d, lr=self.lr_disc, betas=self.betas)
        return [opt_g, opt_d], []

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

    def training_step(self, data_batch, batch_idx, optimizer_idx):
        """Perform a training step."""
        N = data_batch.shape[0]
        if optimizer_idx == 0:
            gen_batch = self.model(N=N, like=data_batch)
            loss = self.loss_generator(gen_batch)
        elif optimizer_idx == 1:
            gen_batch = self.model(N=N, like=data_batch)
            loss = self.loss_discriminator(data_batch, gen_batch)
        else:
            raise RuntimeError("invalid optimizer_idx!")
        return loss

    def on_epoch_end(self):
        validation_data = next(iter(self.valid_loader))
        N = validation_data.shape[0]
        gen = self.model(N=N, like=validation_data)
        g_loss = self.loss_generator(gen)
        d_loss = self.loss_discriminator(validation_data, gen)
        acc = self.accuracy(validation_data, gen)
        self.log("g_loss", g_loss.item())
        self.log("d_loss", d_loss.item())
        self.log("accuracy", acc)
        pf = self.plot_freq
        if pf > 0:
            if self.current_epoch % pf == 0:
                self.visualize(validation_data, gen, g_loss, d_loss, acc)
        return g_loss

    @torch.no_grad()
    def visualize(self, data, gen, g_loss, d_loss, acc):
        data = tensor_to_numpy(data)
        N = data.shape[0]
        fn = self.create_figure_name(prefix="model")
        self.model.visualize(N=N, data=data, save_name=fn, save_dir=self.outdir)
        self.visualize_disc(data, gen, acc)
        if self.current_epoch > 0:
            self.visualize_training()

    def visualize_disc(self, data, gen, acc):
        title = "Accuracy=%1.4f" % acc
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
            )

    def visualize_training(self, version: int = 0):
        try:
            g_loss = self.read_logged_scalar(name="g_loss", version=version)
            d_loss = self.read_logged_scalar(name="d_loss", version=version)
            acc = self.read_logged_scalar(name="accuracy", version=version)
        except FileNotFoundError:
            print("Unable to read logged scalars. FileNotFoundError caught.")
            return None
        fn = self.create_figure_name(prefix="progress")
        plot_gan_progress(g_loss, d_loss, acc, save_name=fn, save_dir=self.outdir)
