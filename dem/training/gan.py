import numpy as np
from torch.optim.adam import Adam

from dem.modules import GenerativeModel, NeuralDiscriminator
from dem.modules.discriminator import Discriminator
from dem.data.dataset import NumpyDataset
from dem.utils.utils import num_trainable_params
from .setup import TrainingSetup, run_training
from .learner import AdversarialLearner


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
    if num_trainable_params(disc) > 0:
        learner = GAN(model, disc, setup)
    else:
        learner = GANFixedDiscriminator(model, disc, setup)
    if verbose:
        print("\n", learner, "\n")
    trainer = run_training(learner, setup.n_epochs, setup.outdir)
    return learner, trainer


class GAN(AdversarialLearner):
    def __init__(
        self,
        model: GenerativeModel,
        discriminator: Discriminator,
        setup: TrainingSetup,
    ):
        super().__init__(model, discriminator, setup)

    def configure_optimizers(self):
        pars_g = self.model.parameters()
        opt_g = Adam(
            pars_g, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )
        pars_d = self.discriminator.parameters()
        opt_d = Adam(
            pars_d, lr=self.lr_disc, betas=self.betas, weight_decay=self.weight_decay
        )
        return [opt_g, opt_d], []

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


class GANFixedDiscriminator(AdversarialLearner):
    def __init__(
        self,
        model: GenerativeModel,
        discriminator: Discriminator,
        setup: TrainingSetup,
    ):
        super().__init__(model, discriminator, setup)
        num_pars = num_trainable_params(discriminator)
        assert (
            num_pars == 0
        ), "Discriminator has trainable parameters, but it should be fixed."

    def configure_optimizers(self):
        pars_g = self.model.parameters()
        return Adam(
            pars_g, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
        )

    def training_step(self, data_batch, batch_idx):
        """Perform a training step."""
        N = data_batch.shape[0]
        gen_batch = self.model(N=N, like=data_batch)
        return self.loss_generator(gen_batch)
