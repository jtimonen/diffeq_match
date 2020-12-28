from dem.sim import sim, plot_sim

z, t = sim(1, 300)
plot_sim(z, t)

z, t = sim(2, 300)
plot_sim(z, t)

z, t = sim(2, 500, omega=2 * 3.14, decay=0.85)
plot_sim(z, t)
