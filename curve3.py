from dem.sim import sim, plot_sim
from dem.model import GenODE
import torch

N = 500

z_data, t_data = sim(3, N, 0.06)
#plot_sim(z_data, t_data)

model = GenODE(2, [-0.7, -1.0], n_hidden=256)

zz = torch.from_numpy(z_data).float()

model.fit(zz, n_epochs=500, n_draws=100, n_timepoints=30, lr=0.01,
          mmd_ell=1.0)
