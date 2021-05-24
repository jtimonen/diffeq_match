import os

import dem
import numpy as np
import torch

D = 2
N_gen = 13
z = np.random.normal(size=(100, D))
z0 = 1 + 0.3 * np.random.normal(size=(N_gen, D))
generator = dem.create_model(init=z0)
disc = dem.create_discriminator(D=D)


def test_gan_creation():
    gan, trainer = dem.train_model(
        model=generator, disc=disc, data=z, n_epochs=0, outdir="tests/out/gan1"
    )
    x, traj, _ = gan.model.forward_numpy(N=64)
    a = x[-1]
    assert a.shape == (64, 2)
    assert a.dtype == np.float32
    num_unique_rows = len(np.unique(x[0].sum(1)))
    assert num_unique_rows == N_gen
    os.rmdir("tests/out/gan1")


def test_gan_training():
    gan, trainer = dem.train_model(
        model=generator, disc=disc, data=z, n_epochs=4, outdir="tests/out/gan2"
    )
    a = gan.model(N=64)
    assert a.shape == (64, 2)
    assert a.dtype == torch.float32
