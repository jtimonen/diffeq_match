from dem import sim, plot_sim
from dem import GenODE, Discriminator

import torch

N = 1000
z_data, t_data = sim(1, N, sigma=0.1)
# plot_sim(z_data, t_data)

t_loc = [[-1.5, 0.0]]
t_std = [0.07]
model = GenODE(t_loc, t_std, n_hidden=128)

zz = torch.from_numpy(z_data).float()
disc = Discriminator(D=model.D)

disc.fit(zz, plot_freq=10, n_epochs=300)
