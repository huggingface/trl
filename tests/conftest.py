import pytest


def pytest_addoption(parser):
    group = parser.getgroup("experimental")
    group.addoption(
        "--run-experimental",
        action="store_true",
        default=False,
        help="Run tests marked as experimental (otherwise they are skipped).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "experimental: marks tests that cover unstable experimental features (skipped by default)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-experimental"):
        return  # user opted in; do not skip

    skip_marker = pytest.mark.skip(reason="experimental test (use --run-experimental to include)")
    for item in items:
        if "experimental" in item.keywords:
            item.add_marker(skip_marker)
