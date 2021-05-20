import dem
from dem import Stage


def test_default_model_creation():
    gm = dem.create_model(None)
    assert gm.num_stages == 1, "default number of stages should be 1"


def test_stages_model_creation():
    s1 = Stage(backwards=True, uniform=False, t_max=1.0, sde=False)
    s2 = Stage(backwards=False, uniform=True, t_max=1.0, sde=True)
    stages = [s1, s2]
    gm = dem.create_model(None, stages=stages, n_hidden=16)
    assert gm.num_stages == 2, "number of stages should be 2"
