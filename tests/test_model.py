import dem
import numpy as np
from dem.modules.networks import ConstantLinear


def test_model_creation():
    model = dem.create_model(D=2)
    z = np.random.normal(size=(10, 2))
    ts = np.linspace(0, 10, 30)
    y_traj = model.traj_numpy(z, ts)
    y_back = model.traj_numpy(z, ts, backward=True)
    assert y_traj.shape == (10, 30, 2)
    assert y_back.shape == (10, 30, 2)


def test_ode_traj():
    A = np.array([[-0.5, -1.5], [2.0, -0.5]], dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    net_f = ConstantLinear(A, b)
    model = dem.GenModel(net_f)
    z = 1 + 0.1 * np.random.normal(size=(10, 2))
    ts = np.linspace(0, 8, 100)
    y_traj = model.traj_numpy(z, ts)
    y_back = model.traj_numpy(z, ts, backward=True)
    dem.visualize(
        trajectories=y_traj, save_dir="test_out", save_name="traj.png", xlim=[-1.0, 1.5]
    )
    dem.visualize(
        trajectories=y_back,
        save_dir="test_out",
        save_name="back.png",
    )
