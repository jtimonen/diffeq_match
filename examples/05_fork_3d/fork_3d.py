from dem import sim, plot_sim
from dem import GenODE
import torch
import numpy as np

N = 1200

sigma = 0.1
z_data, t_data = sim(4, N, sigma)
z3 = 0.5 * np.cos(0.7 * np.pi * t_data).reshape(-1, 1)
z3 = z3 + sigma * np.random.normal(size=z3.shape)
z_data = np.hstack((z_data, z3))

i_loc = [[0.0, 0.0, 0.0]]
i_std = 0.05
t_loc = [[1.5, 1.5, -0.4], [1.5, -1.5, -0.4]]
t_std = [0.05, 0.05]

# plot_sim(z_data, t_data)

model = GenODE(i_loc, i_std, t_loc, t_std, n_hidden=64, sigma=0.05)
zz = torch.from_numpy(z_data).float()

# Create and fit model
model.fit(zz, plot_freq=10, n_epochs=1000, lr=0.005, batch_size=256)
