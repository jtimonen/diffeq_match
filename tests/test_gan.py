import os

import dem
import numpy as np
import torch

D = 2
N_gen = 12
z = np.random.normal(size=(100, D))
z0 = 1 + 0.3 * np.random.normal(size=(N_gen, D))
generator = dem.create_model(init=z0)
disc = dem.create_discriminator(D=D)


def test_gan_creation():
    gan, trainer = dem.train_model(
        model=generator, disc=disc, data=z, gen=z0, n_epochs=0, outdir="test_gan"
    )
    a = next(iter(gan.genloader))
    assert a.shape == (64, 2)
    assert a.dtype == torch.float32
    b = a.numpy()
    num_unique_rows = len(np.unique(b.sum(1)))
    assert num_unique_rows == N_gen
    os.rmdir("test_gan")
