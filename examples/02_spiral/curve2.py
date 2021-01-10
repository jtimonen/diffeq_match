from dem.sim import sim
from dem.model import GenODE
import torch

N = 500

z_data, _ = sim(2, N, omega=2 * 3.14, decay=1.2)

model = GenODE(2, [1.0, 0.0])

zz = torch.from_numpy(z_data).float()

model.fit(zz, n_epochs=500, n_draws=30, n_timepoints=30, lr=0.01)
