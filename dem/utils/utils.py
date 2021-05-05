def reshape_traj(z_traj):
    n_timepoints = z_traj.shape[0]
    n_samples = z_traj.shape[1]
    n_dimensions = z_traj.shape[2]
    return z_traj.view(n_timepoints * n_samples, n_dimensions)
