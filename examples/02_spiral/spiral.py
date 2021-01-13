from dem import sim, plot_sim
from dem import GenODE
import torch

N = 500

z_data, _ = sim(2, N, omega=2 * 3.14, decay=2.0)

z0 = [1.0, 0.0]
model = GenODE(2, z0)

zz = torch.from_numpy(z_data).float()

model.fit_gan(zz, n_epochs=500, lr=0.005, plot_freq=5)
