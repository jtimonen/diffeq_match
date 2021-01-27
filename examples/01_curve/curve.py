from dem import sim, plot_sim
from dem import GenODE

import torch

N = 1000
z_data, t_data = sim(1, N, sigma=0.1)
# plot_sim(z_data, t_data)

t_loc = [[-1.7, -0.15]]
t_std = [0.05]
model = GenODE(t_loc, t_std)

zz = torch.from_numpy(z_data).float()

model.fit(zz, n_epochs=500, lr=0.005, lr_disc=0.005, plot_freq=10, mode="gan")
