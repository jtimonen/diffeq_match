from dem import sim, plot_sim
from dem import GenODE, Discriminator

import torch

N = 1000
z_data, t_data = sim(1, N, sigma=0.1)
# plot_sim(z_data, t_data)

t_loc = [[-1.5, 0.0]]
i_loc = [[1.5, 0.2]]
t_std = [0.07]
i_std = t_std

model = GenODE(i_loc, i_std, t_loc, t_std, n_hidden=24)

zz = torch.from_numpy(z_data).float()

# Create and fit discriminator
disc = Discriminator(D=model.D)
disc.fit(zz, plot_freq=50, n_epochs=300, lr=0.005)

# Create and fit model
model.fit(zz, plot_freq=10, n_epochs=500, lr=0.005, disc=disc, batch_size=64)
