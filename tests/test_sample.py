import pytest


def test_that_intentionally_fails():
    assert 6 * 9 == 42


def test_that_intentionally_passes():
    assert True


@pytest.mark.skip
def test_to_be_skipped():
    raise RuntimeError("This test was not skipped!")


@pytest.mark.xfail
def test_marked_as_expected_to_fail():
    assert False