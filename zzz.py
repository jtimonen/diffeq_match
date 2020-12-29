from dem.sim import sim, plot_sim
from dem.model import GenODE

z, t = sim(1, 300)
# plot_sim(z, t)

# z, t = sim(2, 300)
# plot_sim(z, t)

# z, t = sim(2, 500, omega=2 * 3.14, decay=1.0)
# plot_sim(z, t)

a = GenODE(2, [-1.0, 0.0])
# print(a)

z0, t = a.draw_z0t0(N=20)
print(z0)
print(t)
