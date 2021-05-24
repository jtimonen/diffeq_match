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


def train_model(
    model: GenerativeModel,
    disc: Discriminator,
    data: np.ndarray,
    **training_setup_kwargs
):
    """Train a model.

    :param model: model to be trained
    :param disc: discriminator
    :param data: a numpy array with shape (num_obs, num_dims)
    :param training_setup_kwargs: keyword arguments to `TrainingSetup`
    """
    ds = NumpyDataset(data)
    setup = TrainingSetup(ds, **training_setup_kwargs)
    if disc is None:
        disc = NeuralDiscriminator(D=model.D)
    occ = GAN(model, disc, setup)
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

    def loss_discriminator(self, gen_batch, data_batch):
        """Discriminator loss."""
        target_data = torch.ones(gen_batch.size(0), 1).type_as(data_batch)
        target_gen = torch.zeros(gen_batch.size(0), 1).type_as(gen_batch)
        d_data = self.discriminator(data_batch)
        d_gen = self.discriminator(gen_batch).detach()
        loss_data = binary_cross_entropy(d_data, target_data)
        loss_gen = binary_cross_entropy(d_gen, target_gen)
        return 0.5 * (loss_data + loss_gen)

    def training_step(self, data_batch, batch_idx, optimizer_idx):
        """Perform a training step."""
        N = data_batch.shape[0]
        gen_batch = self.model(N=N, like=data_batch)
        if optimizer_idx == 0:
            loss = self.loss_generator(gen_batch)
        elif optimizer_idx == 1:
            loss = self.loss_discriminator(data_batch, gen_batch)
        else:
            raise RuntimeError("invalid optimizer_idx!")
        return loss

    def on_epoch_end(self):
        validation_data = next(iter(self.valid_loader))
        N = validation_data.shape[0]
        gen = self.model(N=N, like=validation_data)
        g_loss = self.loss_generator(gen)
        d_loss = self.loss_discriminator(gen, validation_data)
        self.log("g_loss", g_loss.item())
        self.log("d_loss", d_loss.item())
        idx_epoch = self.current_epoch
        pf = self.plot_freq
        if pf > 0:
            if idx_epoch % pf == 0:
                self.visualize(gen, validation_data, g_loss, d_loss, idx_epoch)
        return g_loss

    @torch.no_grad()
    def visualize(self, gen, data, g_loss, d_loss, idx_epoch):
        gen = tensor_to_numpy(gen)
        data = tensor_to_numpy(data)
