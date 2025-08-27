from pathlib import Path

import matplotlib
import pytest
from matplotlib.figure import Figure


def pytest_configure(config):  # noqa: ARG001
    matplotlib.use("Agg")


@pytest.fixture()
def test_data_dir():
    return Path(__file__).parent / "tests" / "data"


@pytest.fixture(params=["idefix_planet3d", "fargo3d_planet2d"])
def planet_simulation_dir(test_data_dir, request):
    return test_data_dir / request.param


@pytest.fixture(params=["idefix_rwi", "idefix_planet3d", "fargo3d_planet2d"])
def simulation_dir(test_data_dir, request):
    return test_data_dir / request.param


@pytest.fixture()
def temp_figure_and_axis():
    fig = Figure()
    ax = fig.add_subplot()
    yield (fig, ax)
