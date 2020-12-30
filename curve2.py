from dem.sim import sim
from dem.model import GenODE
import torch

N = 300
z_data, t_data = sim(2, N)
# plot_sim(z, t)

# z, t = sim(2, N)
# plot_sim(z, t)

# z, t = sim(2, N, omega=2 * 3.14, decay=1.0)
# plot_sim(z, t)

model = GenODE(2, [-1.0, 0.0])

zz = torch.from_numpy(z_data).float()

model.fit(zz, n_epochs=200, n_draws=30)

