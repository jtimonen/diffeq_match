from dem.sim import sim
from dem.model import GenODE
import torch

N = 300
z_data, t_data = sim(2, N)


model = GenODE(2, [-1.0, 0.0])
print(model)

zz = torch.from_numpy(z_data).float()

model.fit(zz, n_epochs=400, n_draws=30)

