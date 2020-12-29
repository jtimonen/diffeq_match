from dem.sim import sim, plot_sim
from dem.model import GenODE
import torch

N = 100
z, t = sim(1, N)
# plot_sim(z, t)

# z, t = sim(2, N)
# plot_sim(z, t)

# z, t = sim(2, N, omega=2 * 3.14, decay=1.0)
# plot_sim(z, t)

a = GenODE(2, [-1.0, 0.0])
# print(a)

z0, t = a.draw_z0t0(N=N)

zf = a(z0, t)

zz = torch.from_numpy(z).float()
mmd = a.loss(zz, zf)

a.plot_forward(zz, zf)
