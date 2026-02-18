from copy import deepcopy
from importlib.util import find_spec

import inifix
import matplotlib as mpl
import matplotlib.style
import numpy as np
import pytest
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from nonos._geometry import Coordinates
from nonos.api import Field, GasDataSet, NonosLick
from nonos.loaders import Loader
from nonos.styling import set_mpl_style


@pytest.fixture
def tmp_mpl_state():
    # reset matplotlib's state when the test is over
    style = deepcopy(mpl.rcParams)
    yield
    for k in set(mpl.rcParams.keys()) - set(style.keys()):  # pragma: no cover
        mpl.rcParams.pop(k)
    mpl.rcParams.update(style)


@pytest.mark.usefixtures("tmp_mpl_state")
@pytest.mark.parametrize("scaling", [0.5, 1.0, 2.0])
@pytest.mark.mpl_image_compare(style="default")
def test_set_mpl_style(scaling):
    with pytest.deprecated_call():
        set_mpl_style(scaling)

    fig = Figure()
    ax = fig.add_subplot()

    x = np.linspace(0, 2 * np.pi)
    for phase in np.linspace(0, np.pi / 2, 5):
        y = np.sin(x + phase)
        ax.plot(x, y)

    ax.set(
        title="nonos style",
        xlabel="$x$ axis",
        ylabel="$y$ axis",
    )
    ax.annotate(f"{scaling=}", (0.05, 0.1), xycoords="axes fraction", fontsize=15)

    return fig


@pytest.mark.mpl_image_compare()
def test_2D_polar_plane(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(23, directory=test_data_dir / "idefix_newvtk_planet2d")
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(fig, ax, title="vr")
    return fig


@pytest.mark.mpl_image_compare()
def test_symlog(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(23, directory=test_data_dir / "idefix_newvtk_planet2d")
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(
        fig,
        ax,
        title="vr",
        norm=SymLogNorm(vmin=-0.05, vmax=0.05, linthresh=1e-4, base=10),
    )
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_aa_xz(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    fig, ax = temp_figure_and_axis

    ds["RHO"].azimuthal_average().map("x", "z").plot(fig, ax, log=True, title="rho")
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_vm_phiR(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    fig, ax = temp_figure_and_axis

    ds["RHO"].vertical_at_midplane().map("phi", "R").plot(
        fig, ax, log=True, title="rho"
    )
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_vm_xy(test_data_dir, temp_figure_and_axis):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    fig, ax = temp_figure_and_axis

    ds["RHO"].vertical_at_midplane().map("x", "y").plot(fig, ax, log=True, title="rho")
    return fig


@pytest.mark.skipif(find_spec("lick") is None, reason="lick is not installed")
@pytest.mark.parametrize("method", ["nearest", "linear"])
@pytest.mark.mpl_image_compare()
def test_nonoslick_method(method, tmp_path):
    root_size = 2
    fake_grid = {
        "geometry": "cartesian",
        "x1": np.linspace(0, 1, root_size + 1),
        "x2": np.linspace(0, 1, root_size + 1),
        "x3": np.array([1.0]),
    }
    fake_coords = Coordinates(**fake_grid)

    xxmed = fake_coords.get_axis_array_med("x")
    yymed = fake_coords.get_axis_array_med("y")
    xxedge = fake_coords.get_axis_array("x")
    yyedge = fake_coords.get_axis_array("y")

    rng = np.random.default_rng(seed=0)
    fake_Vx = rng.normal(0, 1, size=root_size**2).reshape(root_size, root_size)
    fake_Vy = rng.normal(0, 1, size=root_size**2).reshape(root_size, root_size)
    fake_F = rng.normal(0, 1, size=root_size**2).reshape(root_size, root_size)

    # TODO : mandatory for now to have a idefix.ini file, but should be removed in the future
    data = {
        "Output": {"vtk": 1},
        "Hydro": {},
    }
    with open(tmp_path / "idefix.ini", "wb") as fh:
        inifix.dump(data, fh)

    loader = Loader.resolve(directory=tmp_path)
    Vx, Vy, F = [
        Field(
            name=name,
            data=arr,
            coordinates=fake_coords,
            native_geometry=fake_coords.geometry,
            snapshot_uid=0,
            loader=loader,
        )
        for name, arr in [("Vx", fake_Vx), ("Vy", fake_Vy), ("F", fake_F)]
    ]
    lick = NonosLick(
        xxmed,
        yymed,
        Vx,
        Vy,
        F,
        xmin=xxedge.min(),
        xmax=xxedge.max(),
        ymin=yyedge.min(),
        ymax=yyedge.max(),
        niter_lic=1,
        size_interpolated=50 * root_size,
        method=method,
    )
    fig = Figure()
    ax = fig.add_subplot()
    lick.plot(
        fig, ax, title="F", density_streamlines=1, color_streamlines="w", cmap="inferno"
    )
    ax.set(
        title=f"{method=}",
        aspect="equal",
        xlim=(xxedge.min(), xxedge.max()),
        ylim=(yyedge.min(), yyedge.max()),
    )
    return fig
