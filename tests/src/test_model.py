import dem
from dem import Stage
import numpy as np
from dem.modules.networks import ConstantLinear


def test_default_model_creation():
    gm = dem.create_model(None)
    assert gm.num_stages == 1, "default number of stages should be 1"


def test_stages_model_creation():
    s1 = Stage(backwards=True, uniform=False, t_max=1.0, sde=False)
    s2 = Stage(backwards=False, uniform=True, t_max=1.0, sde=True)
    stages = [s1, s2]
    gm = dem.create_model(None, stages=stages, n_hidden=16)
    x_forw = gm(10)
    assert gm.num_stages == 2, "number of stages should be 2"
    assert x_forw.shape == (10, 2), "incorrect output tensor shape"


def test_model_visualization():
    gm = dem.create_model(None)
    gm.visualize(9, save_name="model1_viz.png", save_dir="tests/out")
    A = np.array([[-0.5, -1.5], [2.0, -0.5]], dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    s1 = Stage(backwards=True, uniform=False, t_max=1.0, sde=False)
    s2 = Stage(backwards=False, uniform=True, t_max=1.0, sde=True)
    stages = [s1, s2]
    net_f = ConstantLinear(A, b)
    init = 1.0 + 0.2 * np.random.normal(size=(10, 2))
    gm = dem.create_model(init=init, net_f=net_f, stages=stages)
    gm.visualize(100, save_name="model2_viz.png", save_dir="tests/out")
