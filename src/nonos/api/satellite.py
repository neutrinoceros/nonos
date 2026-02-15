import os
import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Literal, TypeAlias

import numpy as np
from packaging.version import Version

from nonos._geometry import Coordinates
from nonos._types import D2, D, F, FArray, FArray2D, StrDict
from nonos.api.analysis import GasField, Plotable
from nonos.loaders import Recipe, loader_from, recipe_from

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def file_analysis(
    filename: os.PathLike[str],
    *,
    inifile: os.PathLike[str] | None = None,
    code: str | None = None,
    directory: os.PathLike[str] | None = None,
    norb: int | None = None,
) -> "FArray2D":
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)

    columns = np.loadtxt(directory / filename, dtype="float64").T
    if norb is None:
        return columns

    loader = loader_from(
        code=code,
        parameter_file=inifile,
        directory=directory,
    )
    recipe = recipe_from(code=code, parameter_file=inifile, directory=directory)
    ini = loader.load_ini_file().meta

    if recipe is Recipe.IDEFIX_VTK and "analysis" in ini["Output"]:
        analysis = ini["Output"]["analysis"]
        rpini = ini["Planet"]["dpl"]
        Ntmean = round(norb * 2 * np.pi * pow(rpini, 1.5) / analysis)
        if find_spec("scipy") is not None:
            from scipy.ndimage import uniform_filter1d

            for i, column in enumerate(columns):
                columns[i] = uniform_filter1d(column, Ntmean)
        else:
            # fallback to numpy if scipy isn't available (less performant)
            for i, column in enumerate(columns):
                columns[i] = np.convolve(column, Ntmean, mode="valid")
    else:
        raise NotImplementedError(
            f"moving average on {norb} orbits is not implemented for the recipe {recipe}"
        )
    return columns


InterpMethod: TypeAlias = Literal["nearest", "linear", "cubic"]


class NonosLick(Generic[F]):
    def __init__(
        self,
        x: FArray2D[F],
        y: FArray2D[F],
        lx: GasField[D2, F],
        ly: GasField[D2, F],
        field: GasField[D2, F],
        *,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        size_interpolated: int = 1000,
        niter_lic: int = 6,
        kernel_length: int = 101,
        method: InterpMethod = "linear",
        method_background: InterpMethod = "nearest",
        light_source: bool = True,
    ) -> None:
        if find_spec("lick") is None:
            raise RuntimeError(
                "NonosLick cannot be instantiated because lick is not installed"
            )

        lick_box_kwargs: StrDict
        if Version(version("lick")) >= Version("0.10.0dev0"):
            from lick import lick_box  # type: ignore[attr-defined]

            lick_box_kwargs = {
                "kernel": np.sin(np.linspace(0, np.pi, kernel_length, endpoint=False)),
                "post_lic": "north-west-light-source" if light_source else None,
                "indexing": "xy",
            }
        else:
            from lick.lick import lick_box  # type: ignore[no-redef]

            lick_box_kwargs = {
                "kernel_length": kernel_length,
                "light_source": light_source,
            }

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # (x,y) are 2D meshgrids at cell centers
        self.X: FArray2D[F]
        self.Y: FArray2D[F]
        self.LINE1: FArray2D[F]
        self.LINE2: FArray2D[F]
        self.F: FArray2D[F]
        self.lick: FArray2D[F]
        self.X, self.Y, self.LINE1, self.LINE2, self.F, self.lick = lick_box(
            x,
            y,
            lx.data,
            ly.data,
            field.data,
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            size_interpolated=size_interpolated,
            niter_lic=niter_lic,
            method=method,
            method_background=method_background,
            **lick_box_kwargs,
        )

    def plot(
        self,
        fig: "Figure",
        ax: "Axes",
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        alpha: float = 0.45,
        log: bool = False,
        cmap: str | None = None,
        title: str | None = None,
        density_streamlines: float | None = None,
        color_streamlines: str = "black",
    ) -> "Artist":
        im = Plotable(
            abscissa=("x", self.X),
            ordinate=("y", self.Y),
            field=("background", self.F),
        ).plot(
            fig,
            ax,
            vmin=vmin,
            vmax=vmax,
            log=log,
            cmap=cmap,
            dpi=500,
            title=title,
            shading="nearest",
            rasterized=True,
        )
        Plotable(
            abscissa=("x", self.X),
            ordinate=("y", self.Y),
            field=("lick", self.lick),
        ).plot(
            fig,
            ax,
            log=False,
            cmap="binary_r",
            dpi=500,
            alpha=alpha,
            shading="nearest",
            rasterized=True,
        )
        if density_streamlines is not None:
            ax.streamplot(
                self.X,
                self.Y,
                self.LINE1,
                self.LINE2,
                density=density_streamlines,
                arrowstyle="->",
                linewidth=0.5,
                color=color_streamlines,
                # color=np.log10(self.F*np.sqrt(self.LINE1**2+self.LINE2**2)),#/np.max(np.log10(self.F*np.sqrt(self.LINE1**2+self.LINE2**2))),
                # cmap=cb.cbmap("binary_r"),
            )
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        return im


@deprecated(
    "nonos.api.satellite.compute is deprecated since v0.20.0 "
    "and may be removed in a future version. "
    "Use GasField.replace instead."
)
def compute(
    field: str,
    data: FArray[D, F],
    ref: GasField[D, F],
) -> GasField[D, F]:
    return ref.replace(name=field, data=data)


@deprecated(
    "nonos.api.satellite.from_data is deprecated since v0.11.0 "
    "and may be removed in a future version. "
    "Use GasField.replace instead."
)
def from_data(
    *,
    field: str,
    data: FArray[D, F],
    coords: Coordinates[F],
    on: int,
    operation: str,
    inifile: os.PathLike[str] | None = None,
    code: str | None = None,
    directory: os.PathLike[str] | None = None,
) -> GasField[D, F]:  # pragma: no cover
    return GasField._legacy_init(
        field,
        data,
        coords,
        coords.geometry,
        on,
        operation=operation,
        inifile=inifile,
        code=code,
        directory=directory,
    )


def from_file(
    *,
    field: str,
    operation: str,
    on: int,
    directory: os.PathLike[str] | None = None,
) -> GasField:
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)
    repout = field.lower()
    headername = directory / "header" / f"header_{operation}.npy"
    with open(headername, "rb") as file:
        dict_coords = np.load(file, allow_pickle=True).item()

    geometry, coord0, coord1, coord2 = dict_coords.values()
    ret_coords = Coordinates(geometry, coord0, coord1, coord2)

    fileout = directory / repout / f"_{operation}_{field}.{on:04d}.npy"
    with open(fileout, "rb") as file:
        ret_data = np.load(file, allow_pickle=True)

    return GasField._legacy_init(
        field,
        ret_data,
        ret_coords,
        geometry,
        on,
        operation=operation,
        directory=directory,
    )
