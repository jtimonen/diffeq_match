#!/usr/bin/python
import sys
import dem


def run_experiment(
    idx: int,
    idx_sim: int,
    N=3000,
    n_epochs=600,
    lr=0.0001,
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
    elif idx_sim == 1:
        prefix = "spiral_2d"
        z_data, t_data = dem.sim(2, N, omega=2 * 3.14, decay=2.0)
    elif idx_sim == 2:
        prefix = "bifur_2d"
        z_data, t_data = dem.sim(3, N, 0.06)
    elif idx_sim == 3:
        prefix = "fork_2d"
        z_data, t_data = dem.sim(4, N, 0.1)
    else:
        raise ValueError("Unknown idx_sim (%d)" % idx)
    fn = "sim_%s_%d.png" % (prefix, idx)
    outdir = "%s_%d" % (prefix, idx)
    dem.plot_sim(z_data, t_data, save_name=fn)
    z_init = z_data[0:100, :]

    # Create model and discriminator
    model = dem.create_model(init=z_init)
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
    if len(sys.argv) < 2:
        raise ValueError("Must give at least one command line argument (idx)!")
    idx = int(sys.argv[1])
    if idx == 0:
        run_experiment(idx)
    if idx == 1:
        run_experiment(idx, kde=True)
    elif idx == 2:
        run_experiment(idx, fixed_kde=True)
    elif idx == 3:
        run_experiment(idx, wgan=True, n_epochs=6000)
    else:
        raise ValueError("Unknown idx (%d)" % idx)
