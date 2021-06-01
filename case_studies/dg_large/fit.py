#!/usr/bin/python
import sys
import numpy as np
import os
import dem


# Get data
def run_fit(idx: int):

    # Read data
    n_latent = idx + 1
    folder_name = "latent_" + str(n_latent)
    outdir = "dem_" + str(n_latent)
    latent = np.loadtxt(os.path.join(folder_name, "latent.txt"))
    colors = np.loadtxt(
        os.path.join(folder_name, "colors.txt"), dtype=np.str, comments=None
    )
    types = np.loadtxt(
        os.path.join(folder_name, "types.txt"), dtype=np.str, comments=None
    )
    idx_init = np.where(types == "nIPC")[0]
    z_init = np.mean(latent[idx_init, :], axis=0, keepdims=True)
    print("init=", z_init)
    s1 = dem.Stage(sigma=0.05, sde=True)
    disc = dem.create_discriminator(D=n_latent, fixed_kde=True)
    model = dem.create_model(init=z_init, stages=[s1])
    print(model)
    dem.train_model(
        model,
        disc,
        data=latent,
        plot_freq=1,
        lr=0.001,
        batch_size=1000,
        outdir=outdir,
        n_epochs=20,
    )


if __name__ == "__main__":
    print("Commandline arguments", str(sys.argv))
    idx = int(sys.argv[1])
    print("idx = ", idx)
    run_fit(idx)
