from dem import sim, plot_sim
from dem import GenODE
import torch

N = 1000

z_data, t_data = sim(3, N, 0.06)

loc = [[1.0, 0.5], [1.0, -1.0]]
std = [0.05, 0.05]

model = GenODE(loc, std, n_hidden=256)

zz = torch.from_numpy(z_data).float()

model.fit(zz, n_epochs=1000, lr=0.01, plot_freq=1, mode="gan")
