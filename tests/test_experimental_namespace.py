import pytest


@pytest.mark.experimental
def test_experimental_namespace_import():
    # Should import without raising, and emit (at most) a warning
    import importlib

    mod = importlib.import_module("trl.experimental")
    assert hasattr(mod, "__all__")
    assert isinstance(mod.__all__, list)
