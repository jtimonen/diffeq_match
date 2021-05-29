import numpy as np

from dem.modules import GenerativeModel
from dem.modules.discriminator import Discriminator
from dem.data.dataset import NumpyDataset
from dem.utils.utils import num_trainable_params
from .setup import TrainingSetup, run_training
from dem.training.gan import GAN, WGAN, GANFixedDiscriminator


def train_model(
    model: GenerativeModel,
    disc: Discriminator,
    data: np.ndarray,
    verbose: bool = True,
    **training_setup_kwargs
):
    """Train a model.

    :param model: model to be trained
    :param disc: discriminator or critic
    :param data: a numpy array with shape (num_obs, num_dims)
    :param verbose: print more information?
    :param training_setup_kwargs: keyword arguments to `TrainingSetup`
    """
    ds = NumpyDataset(data)
    setup = TrainingSetup(ds, **training_setup_kwargs)
    if disc is None:
        raise ValueError("Discriminator cannot be None! See create_discriminator().")
    if num_trainable_params(disc) > 0:
        if disc.is_critic:
            learner = WGAN(model, disc, setup)
        else:
            learner = GAN(model, disc, setup)
    else:
        learner = GANFixedDiscriminator(model, disc, setup)
    if verbose:
        print("\n", learner, "\n")
    trainer = run_training(learner, setup.n_epochs, setup.outdir)
    return learner, trainer
