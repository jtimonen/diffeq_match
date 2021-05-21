import dem


def test_session_info():
    assert dem.session_info() is None, "session_info() should return None if not quiet"
    si = dem.session_info(quiet=True)
    assert len(si) > 30, "session_info() should return a long description"
