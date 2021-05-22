import hdviz


def plot_model_state(x_all, traj_all, data, save_name, save_dir, **kwargs):
    num_x = len(x_all)
    D = x_all[0].shape[1]
    pltr = hdviz.create_plotter(D)
    pltr.add_pointset(x_all[0], label="init")
    for s in range(1, num_x):
        label = "stage " + str(s)
        pltr.add_pointset(x_all[s], label=label)
    for s in range(0, num_x - 1):
        label = "stage " + str(s)
        pltr.add_lineset(traj_all[s], label=label)
    pltr.plot()
    hdviz.draw_plot(save_name, save_dir, **kwargs)
