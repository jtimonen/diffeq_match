from dem import sim, plot_sim
from dem import GenODE
from dem.pretraining import gdist

import torch

N = 300
z_data, t_data = sim(2, N)


model = GenODE(2, [-1.0, 0.0])
print(model)
gd = gdist(z_data)
print(gd.shape)
gd = gd[0, :]
plot_sim(z_data, c=gd)

zz = torch.from_numpy(z_data).float()


# model.fit(zz, n_epochs=300, n_draws=50, plot_freq=10)
