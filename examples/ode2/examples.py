#!/usr/bin/python
import sys
import dem
import numpy as np


def run_experiment(
    idx: int,
    idx_sim: int,
    N=3000,
    n_epochs=1800,
    lr=0.00025,
    batch_size=256,
    wgan=False,
    kde=False,
    fixed_kde=False,
):

    pf = int(n_epochs / 100)
    pf = max(pf, 1)

    # Simulate data
    if idx_sim == 1:
        prefix = "curve_2d"
        z_data, t_data = dem.sim(1, N, sigma=0.1)
    elif idx_sim == 2:
        prefix = "spiral_2d"
        z_data, t_data = dem.sim(2, N, omega=2 * 3.14, decay=2.0)
    elif idx_sim == 3:
        prefix = "bifur_2d"
        z_data, t_data = dem.sim(3, N, 0.06)
    elif idx_sim == 4:
        prefix = "fork_2d"
        z_data, t_data = dem.sim(4, N, 0.1)
    elif idx_sim == 5:
        prefix = "fork_3d"
        sigma = 0.1
        z_data, t_data = dem.sim(4, N, sigma)
        z3 = 0.5 * np.cos(0.7 * np.pi * t_data).reshape(-1, 1)
        z3 = z3 + sigma * np.random.normal(size=z3.shape)
        z_data = np.hstack((z_data, z3))
    else:
        raise ValueError("Unknown idx_sim (%d)" % idx)
    fn = "sim_%s_%d.png" % (prefix, idx)
    outdir = "%s_%d" % (prefix, idx)
    print(" ===== THIS IS EXPERIMENT NAMED <" + outdir + "> ===== ")
    dem.plot_sim(z_data, t_data, save_name=fn)
    z_init = z_data[(N-100):N, :]

    # Create model and discriminator
    s1 = dem.Stage(backwards=True)
    s2 = dem.Stage(sigma=0.05)
    stages = [s1, s2]
    model = dem.create_model(init=z_init, stages=stages)
    disc = dem.create_discriminator(
        D=model.D, critic=wgan, kde=kde, fixed_kde=fixed_kde
    )

    # Training
    dem.train_model(
        model,
        disc,
        z_data,
        plot_freq=pf,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        outdir=outdir,
    )


if __name__ == "__main__":
    print("Command line arguments:", str(sys.argv))
    if len(sys.argv) < 3:
        raise ValueError(
            "Must give at least two command line arguments (idx, " "idx_sim)!"
        )
    idx = int(sys.argv[1])
    idx_sim = int(sys.argv[2])
    if idx == 0:
        run_experiment(idx, idx_sim)
    if idx == 1:
        run_experiment(idx, idx_sim, kde=True)
    elif idx == 2:
        run_experiment(idx, idx_sim, fixed_kde=True)
    elif idx == 3:
        run_experiment(idx, idx_sim, wgan=True, n_epochs=10000)
    else:
        raise ValueError("Unknown idx (%d)" % idx)
