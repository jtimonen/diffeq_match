from dem import sim, plot_sim
from dem import GenODE

import torch

N = 300
z_data, t_data = sim(2, N)

z0 = [-1.0, 0.0]
model = GenODE(2, z0)

zz = torch.from_numpy(z_data).float()

model.fit_mmd(zz, n_epochs=200, n_draws=50, plot_freq=10)
