import os

import numpy.testing as npt
import pytest
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from nonos._approx import closest_index
from nonos.api import GasDataSet, compute, from_data


def test_plot_planet_corotation(test_data_dir):
    os.chdir(test_data_dir / "idefix_planet3d")

    ds = GasDataSet(43, geometry="polar")
    field = ds["RHO"].radial_at_r().vertical_at_midplane()
    azimfield = field.map("phi").data
    assert closest_index(azimfield, azimfield.max()) != 0

    azimfieldPlanet = field.map("phi", rotate_with="planet0.dat").data
    assert closest_index(azimfieldPlanet, azimfieldPlanet.max()) == 0


def test_unit_conversion(test_data_dir):
    os.chdir(test_data_dir / "idefix_planet3d")

    ds = GasDataSet(43, geometry="polar")
    fig = Figure()
    ax = fig.add_subplot()

    plotfield10 = (
        ds["RHO"]
        .vertical_at_midplane()
        .map("R", "phi")
        .plot(fig, ax, unit_conversion=10)
    )
    plotfield = ds["RHO"].vertical_at_midplane().map("R", "phi").plot(fig, ax)

    npt.assert_allclose(plotfield10.get_array(), 10 * plotfield.get_array())


def test_vmin_vmax_api(test_data_dir):
    ds = GasDataSet(1, directory=test_data_dir / "idefix_rwi", geometry="polar")
    fig = Figure()
    ax = fig.add_subplot()
    p = ds["VX1"].vertical_at_midplane().map("R", "phi")

    # check that no warning is emitted from matplotlib
    im1 = p.plot(fig, ax, vmin=-1, vmax=1, norm=SymLogNorm(linthresh=0.1, base=10))
    im2 = p.plot(fig, ax, norm=SymLogNorm(linthresh=0.1, base=10, vmin=-1, vmax=1))

    npt.assert_array_equal(im1.get_array(), im2.get_array())


def test_compute_from_data(test_data_dir):
    ne = pytest.importorskip("numexpr")

    directory = test_data_dir / "idefix_planet3d"
    os.chdir(directory)

    ds = GasDataSet(43, geometry="polar")

    rhovpfield = ds["RHO"].vertical_projection()
    vx2vpfield = ds["VX2"].vertical_projection()

    rhovp = rhovpfield.data
    vx2vp = vx2vpfield.data

    with pytest.deprecated_call():
        rhovx2_from_data = from_data(
            field="RHOVX2",
            data=rhovp * vx2vp,
            coords=rhovpfield.coords,
            on=rhovpfield.on,
            operation=rhovpfield.operation,
            directory=directory,
        )

    datane = ne.evaluate("rhovp*vx2vp")
    with pytest.deprecated_call():
        rhovx2_compute = compute(
            field="RHOVX2",
            data=datane,
            ref=rhovpfield,
        )

    npt.assert_array_equal(rhovx2_from_data.data, rhovx2_compute.data)

    rhovx2_replace = ds["RHO"].replace(name="RHOVX2", data=datane)
    npt.assert_array_equal(rhovx2_compute.data, rhovx2_replace.data)


def test_corotation_api_float(test_data_dir):
    os.chdir(test_data_dir / "idefix_newvtk_planet2d")

    ds = GasDataSet(23)
    case1 = ds["RHO"].map("x", "y", rotate_with="planet0.dat")
    ds = GasDataSet(23)
    case2 = ds["RHO"].map("x", "y", rotate_by=1.89628895460529)

    npt.assert_array_equal(case1.data, case2.data)


@pytest.mark.parametrize(
    "map_args",
    [("R", "phi"), ("phi", "R")],
)
def test_reg(test_data_dir, map_args):
    ds = GasDataSet(23, directory=test_data_dir / "idefix_newvtk_planet2d")
    fig = Figure()
    ax = fig.add_subplot()
    ds["RHO"].map(*map_args).plot(fig, ax)
