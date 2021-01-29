from dem import sim, plot_sim
from dem import GenODE, Discriminator
import torch

N = 1000

z_data, _ = sim(2, N, omega=2 * 3.14, decay=2.0)

t_loc = [[1.0, 0.0]]
t_std = [0.05]
model = GenODE(t_loc, t_std)

zz = torch.from_numpy(z_data).float()

# Create and fit discriminator
disc = Discriminator(D=model.D)
disc.fit(zz, plot_freq=100, n_epochs=1000, lr=0.005)

# Create and fit model
model.fit(zz, plot_freq=10, n_epochs=500, lr=0.005, disc=disc, batch_size=128)
