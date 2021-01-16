from dem import sim, plot_sim
from dem import GenODE

import torch

N = 500
z_data, t_data = sim(2, N)
# plot_sim(z_data, t_data)

t_loc = [[-1.0, 0.0]]
t_std = [0.05]
model = GenODE(t_loc, t_std)

zz = torch.from_numpy(z_data).float()

# model.fit_gan(zz, n_epochs=600, plot_freq=10, lr=0.001)
model.fit(zz, n_epochs=300, plot_freq=10, lr=0.005)
