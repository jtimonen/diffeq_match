import hdviz


def determine_line_prop(sde, backwards, stage_idx):
    if backwards and sde:
        raise RuntimeError("sde and backwards cannot be True at the same time!")
    if not sde:
        color = "blue" if backwards else "black"
        style = "dotted" if (stage_idx % 2 == 1) else "solid"
    else:
        color = "red"
        style = "solid"
    return color, style


def plot_simulation(
    sim, data, save_name, save_dir, point_alpha, ode_alpha, sde_alpha, title, **kwargs
):
    pltr = hdviz.create_plotter(sim.D)
    if data is not None:
        pltr.add_pointset(data, label="data", alpha=point_alpha, marker="x")
    pltr.add_pointset(sim.x_init, label="init", alpha=point_alpha, marker="x")
    for s in range(0, sim.num_stages):
        label = "stage " + str(s + 1)
        pltr.add_pointset(sim.x_stages[s], label=label, alpha=point_alpha, marker="x")
        ot = sim.traj_ode[s]
        st = sim.traj_sde[s]
        rev = sim.stages[s].backwards
        if ot is not None:
            color, style = determine_line_prop(False, rev, s)
            tlab = label + ", ODE, back=" + str(rev)
            pltr.add_lineset(ot, label=tlab, color=color, alpha=ode_alpha, style=style)
        if st is not None:
            color, style = determine_line_prop(True, rev, s)
            tlab = label + ", SDE, back=" + str(rev)
            pltr.add_lineset(st, label=tlab, color=color, alpha=sde_alpha, style=style)
    pltr.plot(title=title)
    hdviz.draw_plot(save_name, save_dir, **kwargs)
