import dem
import numpy as np


def test_model_creation():
    model = dem.create_model(D=2)
    z = np.random.normal(size=(10, 2))
    ts = np.linspace(0, 10, 30)
    y_traj = model.traj_numpy(z, ts)
    y_back = model.traj_numpy(z, ts, backward=True)
    assert y_traj.shape == (30, 10, 2)
    assert y_back.shape == (30, 10, 2)
