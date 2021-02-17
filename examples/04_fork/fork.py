from dem import sim, plot_sim
from dem import GenODE, Discriminator
import torch
from dem.plotting import plot_kde

N = 1200

z_data, t_data = sim(4, N, 0.1)

i_loc = [[0.0, 0.0]]
i_std = 0.05
t_loc = [[1.5, 1.5], [1.5, -1.5]]
t_std = [0.05, 0.05]

# plot_sim(z_data, t_data)

model = GenODE(i_loc, i_std, t_loc, t_std, n_hidden=64, sigma=0.05)
zz = torch.from_numpy(z_data).float()

# Create and fit model
model.fit(zz, plot_freq=5, n_epochs=1000, lr=0.005, batch_size=256)
