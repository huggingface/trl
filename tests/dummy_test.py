import pytest


@pytest.mark.parametrize("value", [1,2,3])
def test_value(value):
    assert value == value


