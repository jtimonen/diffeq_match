import dem


def test_model_creation():
    gm = dem.create_model(None)
    assert gm.num_stages == 1, "default number of stages should be 1"
