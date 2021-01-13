from dem import sim, plot_sim
from dem import GenODE
import torch

N = 1000

z_data, t_data = sim(3, N, 0.06)
# plot_sim(z_data, t_data)

model = GenODE(2, [-0.7, -1.0], n_hidden=256)

zz = torch.from_numpy(z_data).float()

model.fit_gan(zz, n_epochs=5000, lr=0.005, plot_freq=10)
