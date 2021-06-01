import dem
import numpy as np
from dem.modules.networks import ConstantLinear


def test_dynamics_creation():
    dyn = dem.create_dynamics(2)
    z = np.random.normal(size=(10, 2))
    ts = np.linspace(0, 10, 30)
    y_traj = dyn.traj_numpy(z, ts)
    y_back = dyn.traj_numpy(z, ts, backward=True)
    assert y_traj.shape == (10, 30, 2)
    assert y_back.shape == (10, 30, 2)


def test_ode_traj():
    A = np.array([[-0.5, -1.5], [2.0, -0.5]], dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    net_f = ConstantLinear(A, b)
    dyn = dem.DynamicModel(net_f=net_f)
    z = 1 + 0.1 * np.random.normal(size=(10, 2))
    ts = np.linspace(0, 8, 100)
    y_traj = dyn.traj_numpy(z, ts)
    y_back = dyn.traj_numpy(z, ts, backward=True)
    dem.visualize(
        points=y_traj[:, 1, :],
        trajectories=y_traj,
        save_dir="tests/out",
        save_name="traj.png",
        xlim=[-1.0, 1.5],
    )
    dem.visualize(
        points=y_back[:, 1, :],
        trajectories=y_back,
        save_dir="tests/out",
        save_name="back.png",
        lines_kwargs=dict(linewidth=3),
    )
