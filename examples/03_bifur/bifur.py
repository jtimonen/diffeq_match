from dem import sim, plot_sim
from dem import GenODE
from dem.pretraining import gdist
import torch

N = 500

z_data, t_data = sim(3, N, 0.06)
# plot_sim(z_data, t_data)

model = GenODE(2, [-0.7, -1.0], n_hidden=256)
print(model)
gd = gdist(z_data)
print(gd.shape)
gd = gd[0, :]
plot_sim(z_data, c=gd)

zz = torch.from_numpy(z_data).float()

# model.fit(zz, n_epochs=500, n_draws=100, n_timepoints=30, lr=0.01, mmd_ell=1.0,
# plot_freq = 10)
