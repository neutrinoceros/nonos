import dataclasses
import json
import sys
import warnings
from collections import deque
from collections.abc import ItemsView, KeysView, ValuesView
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import SymmetricalLogLocator

from nonos._geometry import (
    Axis,
    Coordinates,
    Geometry,
    axes_from_geometry,
)
from nonos._readers.binary import NPYReader
from nonos._types import FloatArray, PathT, PlanetData
from nonos.api._angle_parsing import (
    _fequal,
    _parse_planet_file,
    _parse_rotation_angle,
)
from nonos.api.tools import find_around, find_nearest
from nonos.loaders import Recipe, loader_from, recipe_from

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never


if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@dataclass(frozen=True, eq=False, slots=True)
class NamedArray:
    name: str
    data: FloatArray


class Plotable:
    __slots__ = ["abscissa", "ordinate", "field"]

    def __init__(
        self,
        *,
        abscissa: tuple[str, FloatArray],
        ordinate: tuple[str, FloatArray],
        field: tuple[str, FloatArray] | None = None,
    ) -> None:
        self.abscissa = NamedArray(*abscissa)
        self.ordinate = NamedArray(*ordinate)
        self.field = None if field is None else NamedArray(*field)
        if ndim := self.data.ndim > 2:
            raise TypeError(
                f"Plotable doesn't support data with dimensionality>2, got {ndim}"
            )

    @property
    def data(self) -> "NDArray[np.floating]":
        if self.field is not None:
            arr = self.field.data
            assert arr.ndim == 2
        else:
            arr = self.ordinate.data
            assert arr.ndim == 1

        return arr

    def plot(
        self,
        fig: "Figure",
        ax: "Axes",
        *,
        log=False,
        cmap: str | None = "inferno",
        filename=None,
        fmt="png",
        dpi=500,
        title=None,
        unit_conversion=None,
        nbin=None,  # deprecated
        **kwargs,
    ) -> "Artist":
        if nbin is not None:
            warnings.warn(
                "The nbin parameter has no effect and is deprecated",
                stacklevel=2,
            )
        data = self.data
        if unit_conversion is not None:
            data = data * unit_conversion
        if log:
            data = np.log10(data)

        akey = self.abscissa.name
        aval = self.abscissa.data
        okey = self.ordinate.name
        oval = self.ordinate.data

        artist: Artist
        if data.ndim == 2:
            kw = {}
            if (norm := kwargs.get("norm")) is not None:
                if "vmin" in kwargs:
                    norm.vmin = kwargs.pop("vmin")
                if "vmax" in kwargs:
                    norm.vmax = kwargs.pop("vmax")
            else:
                vmin = kwargs.pop("vmin") if "vmin" in kwargs else np.nanmin(data)
                vmax = kwargs.pop("vmax") if "vmax" in kwargs else np.nanmax(data)
                kw.update({"vmin": vmin, "vmax": vmax})

            artist = im = ax.pcolormesh(aval, oval, data, cmap=cmap, **kwargs, **kw)
            ax.set(
                xlim=(aval.min(), aval.max()),
                ylim=(oval.min(), oval.max()),
                xlabel=akey,
                ylabel=okey,
            )
            if title is not None:
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(
                    im, cax=cax, orientation="vertical"
                )  # , format='%.0e')
                cbar.set_label(title)

                cb_axis = cbar.ax.yaxis
                if cb_axis.get_scale() == "symlog":
                    # no minor tick is drawn in symlog norms by default
                    # as of matplotlib 3.7.1, see
                    # https://github.com/matplotlib/matplotlib/issues/25994
                    trf = cb_axis.get_transform()
                    if not isinstance(trf, SymmetricalLogTransform):
                        raise RuntimeError
                    cb_axis.set_major_locator(SymmetricalLogLocator(trf))
                    if float(trf.base).is_integer():
                        locator = SymmetricalLogLocator(
                            trf, subs=list(range(1, int(trf.base)))
                        )
                        cb_axis.set_minor_locator(locator)
        elif data.ndim == 1:
            vmin = kwargs.pop("vmin") if "vmin" in kwargs else np.nanmin(data)
            vmax = kwargs.pop("vmax") if "vmax" in kwargs else np.nanmax(data)
            if "norm" in kwargs:
                kwargs.pop("norm")
            artist = ax.plot(aval, data, **kwargs)[0]
            ax.set(ylim=(vmin, vmax), xlabel=akey)
            if title is not None:
                ax.set_ylabel(title)
        else:
            raise TypeError(
                f"Plotable doesn't support data with dimensionality>2, got {data.ndim}"
            )
        if filename is not None:
            fig.savefig(f"{filename}.{fmt}", bbox_inches="tight", dpi=dpi)

        return artist


class GasField:
    """Idefix field class

    Attributes
    ==========
    data : array corresponding to the 3D data cube.
    coords : Coordinates of the data.
        Edge of the cells.
    coordsmed : Coordinates of the data.
        Center of the cells.
    """

    def __init__(
        self,
        field: str,
        data: np.ndarray,
        coords: Coordinates,
        ngeom: str,
        on: int,
        operation: str,
        *,
        inifile: PathT | None = None,
        code: str | Recipe | None = None,
        directory: PathT | None = None,
        rotate_by: float | None = None,
        rotate_with: str | None = None,
        rotate_grid: int = -1,  # deprecated
    ) -> None:
        self.field = field
        self.operation = operation
        self.native_geometry = Geometry(ngeom)
        self.data = data
        self.coords = coords
        self.on = on

        if directory is None:
            directory = Path.cwd()
        self.directory = Path(directory)

        self._loader = loader_from(
            code=code,
            parameter_file=inifile,
            directory=directory,
        )
        self.inifile = self._loader.parameter_file

        self._rotate_by = _parse_rotation_angle(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_number_argument=(
                "rotate_grid",
                (rotate_grid if rotate_grid >= 0 else None),
            ),
            stacklevel=2,
            planet_azimuth_finder=self,
        )

        # TODO: remove this after deprecation
        self._recipe = recipe_from(parameter_file=inifile, directory=directory)
        self._code = str(code or self._recipe)

    @property
    def code(self) -> str:
        warnings.warn(
            "GasField.code is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._code

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns
        =======
        shape : tuple
        """
        i, j, k = (max(1, n - 1) for n in self.coords.shape)
        return i, j, k

    def map(
        self,
        *wanted,
        rotate_by: float | None = None,
        rotate_with: str | None = None,
        planet_corotation: int | None = None,  # deprecated
    ) -> Plotable:
        rotate_by = _parse_rotation_angle(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_number_argument=("planet_corotation", planet_corotation),
            stacklevel=2,
            planet_azimuth_finder=self,
        )

        data_key = self.field
        # we count the number of 1 in the shape of the data, which gives the real dimension of the data,
        # i.e. the number of reductions already performed (0 -> 3D, 1 -> 2D, 2 -> 1D)
        if self.shape.count(1) not in (1, 2):
            raise ValueError("data has to be 1D or 2D in order to call map.")
        dimension = len(wanted)

        if dimension == 1:
            axis_1 = Axis.from_label(wanted[0])
            meshgrid_conversion = self.coords._meshgrid_conversion_1d(axis_1)

            abscissa_value = list(meshgrid_conversion.values())[0]
            abscissa_key = list(meshgrid_conversion.keys())[0]
            if "phi" in wanted and not _fequal(self._rotate_by, rotate_by):
                phicoord = self.coords.get_axis_array(Axis.AZIMUTH) - rotate_by
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry is Geometry.CARTESIAN:
                    raise NotImplementedError(
                        "rotation isn't implemented for cartesian geometry"
                    )
                elif self.native_geometry is Geometry.POLAR:
                    data_view = np.roll(self.data, -ipi + 1, axis=1)
                elif self.native_geometry is Geometry.SPHERICAL:
                    data_view = np.roll(self.data, -ipi + 1, axis=2)
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if corotation"
                    )
            else:
                data_view = self.data.view()

            return Plotable(
                abscissa=(abscissa_key.label, abscissa_value),
                ordinate=(data_key, data_view.squeeze()),
            )

        elif dimension == 2:
            axis_1_str, axis_2_str = wanted
            axis_1 = Axis.from_label(axis_1_str)
            axis_2 = Axis.from_label(axis_2_str)

            meshgrid_conversion = self.coords._meshgrid_conversion_2d(axis_1, axis_2)
            # meshgrid in polar coordinates P, R (if "R", "phi") or R, P (if "phi", "R")
            # idem for all combinations of R,phi,z
            abscissa_value, ordinate_value = (
                meshgrid_conversion[axis_1],
                meshgrid_conversion[axis_2],
            )
            abscissa_key, ordinate_key = (axis_1, axis_2)
            native_plane_axes = self.coords.native_from_wanted(axis_1, axis_2)
            if Axis.AZIMUTH in native_plane_axes and not _fequal(
                self._rotate_by, rotate_by
            ):
                phicoord = self.coords.get_axis_array(Axis.AZIMUTH) - rotate_by
                # ipi = find_nearest(phicoord, np.pi)
                # if (abs(np.pi-phicoord[ipi])>abs(np.ediff1d(find_around(phicoord, np.pi))[0])):
                #     ipi = find_nearest(phicoord, -np.pi)
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry is Geometry.POLAR:
                    data_view = np.roll(self.data, -ipi + 1, axis=1)
                elif self.native_geometry is Geometry.SPHERICAL:
                    data_view = np.roll(self.data, -ipi + 1, axis=2)
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if corotation"
                    )
            else:
                data_view = self.data.view()

            def rotate_axes(arr, shift: int):
                axes_in = tuple(range(arr.ndim))
                axes_out = deque(axes_in)
                axes_out.rotate(shift)
                return np.moveaxis(arr, axes_in, axes_out)

            # make reduction axis the first axis then drop (squeeze) it,
            # while preserving the original (cyclic) order in the other two axes
            data_view = rotate_axes(data_view, shift=self.shape.index(1)).squeeze()

            naxes = axes_from_geometry(self.native_geometry)
            sorted_pairs = [
                (naxes[0], naxes[1]),
                (naxes[1], naxes[2]),
                (naxes[2], naxes[0]),
            ]
            if native_plane_axes in sorted_pairs:
                data_view = data_view.T

            return Plotable(
                abscissa=(abscissa_key.label, abscissa_value),
                ordinate=(ordinate_key.label, ordinate_value),
                field=(data_key, data_view),
            )
        else:
            raise RuntimeError

    def save(
        self,
        directory: PathT | None = None,
        header_only: bool = False,
    ) -> Path:
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)
        operation = self.operation
        headerdir = directory / "header"
        subdir = directory / self.field.lower()
        file = subdir / f"{operation}_{self.field}.{self.on:04d}.npy"
        if not header_only:
            if not file.is_file():
                subdir.mkdir(exist_ok=True, parents=True)
                with open(file, "wb") as fh:
                    np.save(fh, self.data)

        group_of_files = list(subdir.glob(f"{operation}*"))
        op_suffix = f"_{operation}" if operation != "" else ""
        filename = f"header{op_suffix}.json"
        header_file = headerdir / filename
        if (len(group_of_files) > 0 and not header_file.is_file()) or header_only:
            headerdir.mkdir(exist_ok=True, parents=True)
            if not header_file.is_file():
                dictsaved = self.coords.to_dict()

                def is_array(item: tuple[str, Any]) -> bool:
                    _key, value = item
                    return isinstance(value, np.ndarray)

                for key, value in filter(is_array, dictsaved.items()):
                    dictsaved[key] = value.tolist()
                with open(header_file, "w") as hfile:
                    json.dump(dictsaved, hfile, indent=2)

        src = self.inifile.resolve()
        dest = directory / self.inifile.name
        if dest != src:
            copyfile(src, dest)

        return file

    def find_ir(self, distance: float = 1.0):
        if self.native_geometry is Geometry.POLAR:
            return find_nearest(
                self.coords.get_axis_array_med(Axis.CYLINDRICAL_RADIUS), distance
            )
        elif self.native_geometry is Geometry.SPHERICAL:
            return find_nearest(
                self.coords.get_axis_array_med(Axis.SPHERICAL_RADIUS), distance
            )
        elif self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError
        else:
            assert_never(self.native_geometry)

    def find_imid(self, altitude: float = 0.0):
        if (
            self.native_geometry is Geometry.CARTESIAN
            or self.native_geometry is Geometry.POLAR
        ):
            arr = self.coords.get_axis_array_med(Axis.CARTESIAN_Z)
            return find_nearest(arr, altitude)
        elif self.native_geometry is Geometry.SPHERICAL:
            arr = self.coords.get_axis_array_med(Axis.COLATITUDE)
            return find_nearest(arr, np.pi / 2 - altitude)
        else:
            assert_never(self.native_geometry)

    def find_iphi(self, phi: float = 0):
        if (
            self.native_geometry is Geometry.POLAR
            or self.native_geometry is Geometry.SPHERICAL
        ):
            phiarr = self.coords.get_axis_array(Axis.AZIMUTH)
            mod = len(phiarr) - 1
            return find_nearest(phiarr, phi) % mod
        elif self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError
        else:
            assert_never(self.native_geometry)

    def _load_planet(
        self,
        *,
        planet_number: int | None = None,
        planet_file: str | None = None,
    ) -> PlanetData:
        planet_file = _parse_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        file = self.directory / planet_file
        return self._loader.load_planet_data(file)

    def _get_ind_output_number(self, time) -> int:
        ini = self._loader.load_ini_file()
        target_time = ini.output_time_interval * self.on
        return find_nearest(time, target_time)

    def find_rp(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
    ) -> float:
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        ind_on = self._get_ind_output_number(pd.t)
        return pd.d[ind_on]  # type: ignore [attr-defined]

    def find_rhill(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
    ) -> float:
        ini = self._loader.load_ini_file()
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        oe = pd.get_orbital_elements(ini.frame)
        ind_on = self._get_ind_output_number(pd.t)
        return pow(pd.q[ind_on] / 3.0, 1.0 / 3.0) * oe.a[ind_on]

    def find_phip(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
    ) -> float:
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        ind_on = self._get_ind_output_number(pd.t)
        return np.arctan2(pd.y, pd.x)[ind_on] % (2 * np.pi)

    @staticmethod
    def _parse_operation_name(
        *,
        prefix: str,
        default_suffix: str,
        operation_name: str | None,
    ) -> str:
        if operation_name == "":
            raise ValueError("operation_name cannot be empty")
        suffix = operation_name or default_suffix
        if prefix:
            return f"{prefix}_{suffix}"
        else:
            return suffix

    def latitudinal_projection(self, theta=None, *, operation_name=None) -> "GasField":
        default_suffix = "latitudinal_projection"
        if theta is not None:
            default_suffix += str(np.pi / 2 - theta)
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix=default_suffix,
            operation_name=operation_name,
        )

        imid = self.find_imid()
        if self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError(
                "latitudinal_projection isn't implemented for cartesian geometry"
            )
        elif self.native_geometry is Geometry.POLAR:
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.get_axis_array("phi"),
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            R = self.coords.Rmed
            z = self.coords.zmed
            integral = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
            # integral = np.zeros((self.shape[0],self.shape[2]), dtype='>f4')
            for i in range(self.shape[0]):
                km = find_nearest(z, z.min())
                kp = find_nearest(z, z.max())
                if theta is not None:
                    km = find_nearest(z, -R[i] * theta)
                    kp = find_nearest(z, R[i] * theta)
                integral[i, :] = np.sum(
                    (self.data[i, :, :] * np.ediff1d(self.coords.z)[None, :])[
                        :, km : kp + 1
                    ],
                    axis=1,
                    dtype="float64",
                )
                # integral[i,km] = -1
                # integral[i,kp] = 1
            ret_data = integral.reshape(self.shape[0], self.shape[1], 1)
            # ret_data = integral.reshape(self.shape[0],1,self.shape[2])
        elif self.native_geometry is Geometry.SPHERICAL:
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.get_axis_array("r"),
                find_around(
                    self.coords.get_axis_array("theta"),
                    self.coords.get_axis_array_med("theta")[imid],
                ),
                self.coords.get_axis_array("phi"),
            )
            km = find_nearest(
                self.coords.get_axis_array("theta"),
                self.coords.get_axis_array("theta").min(),
            )
            kp = find_nearest(
                self.coords.get_axis_array("theta"),
                self.coords.get_axis_array("theta").max(),
            )
            if theta is not None:
                kp = find_nearest(
                    self.coords.get_axis_array("theta"), np.pi / 2 + theta
                )
                km = find_nearest(
                    self.coords.get_axis_array("theta"), np.pi / 2 - theta
                )
            ret_data = (
                np.sum(
                    (
                        self.data
                        * self.coords.get_axis_array_med("r")[:, None, None]
                        * np.sin(self.coords.get_axis_array_med("theta")[None, :, None])
                        * np.ediff1d(self.coords.get_axis_array("theta"))[None, :, None]
                    )[:, km : kp + 1, :],
                    axis=1,
                    dtype="float64",
                )
            ).reshape(self.shape[0], 1, self.shape[2])
        else:
            assert_never(self.native_geometry)
        return GasField(
            self.field,
            ret_data.astype("float32", copy=False),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def vertical_projection(self, z=None, *, operation_name=None) -> "GasField":
        default_suffix = "vertical_projection"
        if z is not None:
            default_suffix += str(z)
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix=default_suffix,
            operation_name=operation_name,
        )

        imid = self.find_imid()
        if self.native_geometry is Geometry.CARTESIAN:
            zarr = self.coords.get_axis_array(Axis.CARTESIAN_Z)
            zmed = self.coords.get_axis_array_med(Axis.CARTESIAN_Z)
            ret_coords = self.coords.project_along(Axis.CARTESIAN_Z, zmed[imid].item())
            km = find_nearest(zmed, zarr.min())
            kp = find_nearest(zmed, zarr.max())
            if z is not None:
                km = find_nearest(zmed, -z)
                kp = find_nearest(zmed, z)
            ret_data = (
                np.nansum(
                    (self.data * np.ediff1d(zarr))[:, :, km : kp + 1],
                    axis=2,
                    dtype="float64",
                )
            ).reshape(self.shape[0], self.shape[1], 1)
        elif self.native_geometry is Geometry.POLAR:
            zarr = self.coords.get_axis_array(Axis.CARTESIAN_Z)
            zmed = self.coords.get_axis_array_med(Axis.CARTESIAN_Z)
            ret_coords = self.coords.project_along(Axis.CARTESIAN_Z, zmed[imid].item())
            km = find_nearest(zmed, zarr.min())
            kp = find_nearest(zmed, zarr.max())
            if z is not None:
                km = find_nearest(zmed, -z)
                kp = find_nearest(zmed, z)
            ret_data = (
                np.nansum(
                    (self.data * np.ediff1d(zarr))[:, :, km : kp + 1],
                    axis=2,
                    dtype="float64",
                )
            ).reshape(self.shape[0], self.shape[1], 1)
        elif self.native_geometry is Geometry.SPHERICAL:
            raise NotImplementedError(
                """
                vertical_projection(z) function not implemented in spherical coordinates.\n
                Maybe you could use the function latitudinal_projection(theta)?
                """
            )
        else:
            assert_never(self.native_geometry)
        return GasField(
            self.field,
            ret_data.astype("float32", copy=False),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def vertical_at_midplane(self, *, operation_name=None) -> "GasField":
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix="vertical_at_midplane",
            operation_name=operation_name,
        )
        imid = self.find_imid()
        if self.native_geometry is Geometry.CARTESIAN:
            # find_around looks around the 2 coords values that surround coordmed at imid
            zmed = self.coords.get_axis_array_med(Axis.CARTESIAN_Z)
            ret_coords = self.coords.project_along(Axis.CARTESIAN_Z, zmed[imid].item())
            ret_data = self.data[:, :, imid].reshape(self.shape[0], self.shape[1], 1)
            # do geometry conversion!!! -> chainer la conversion (une fois que reduction de dimension -> conversion puis plot egalement chainable)
        elif self.native_geometry is Geometry.POLAR:
            zmed = self.coords.get_axis_array_med(Axis.CARTESIAN_Z)
            ret_coords = self.coords.project_along(Axis.CARTESIAN_Z, zmed[imid].item())
            ret_data = self.data[:, :, imid].reshape(self.shape[0], self.shape[1], 1)
        elif self.native_geometry is Geometry.SPHERICAL:
            thetamed = self.coords.get_axis_array_med(Axis.COLATITUDE)
            ret_coords = self.coords.project_along(
                Axis.COLATITUDE, thetamed[imid].item()
            )
            ret_data = self.data[:, imid, :].reshape(self.shape[0], 1, self.shape[2])
        else:
            assert_never(self.native_geometry)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def latitudinal_at_theta(self, theta=0.0, *, operation_name=None) -> "GasField":
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix=f"latitudinal_at_theta{np.pi / 2 - theta}",
            operation_name=operation_name,
        )

        imid = self.find_imid(altitude=theta)
        if self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError(
                "latitudinal_at_theta is not implemented for cartesian geometry"
            )
        if self.native_geometry is Geometry.POLAR:
            data_at_theta = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
            zmed = self.coords.get_axis_array_med(Axis.CARTESIAN_Z)
            R = self.coords.get_axis_array(Axis.CYLINDRICAL_RADIUS)
            for i in range(self.shape[0]):
                iz0 = find_nearest(zmed, R[i] / np.tan(np.pi / 2 - theta))
                if np.sign(theta) >= 0:
                    if iz0 < self.shape[2]:
                        data_at_theta[i, :] = self.data[i, :, iz0]
                    else:
                        data_at_theta[i, :] = np.nan
                else:
                    if iz0 > 0:
                        data_at_theta[i, :] = self.data[i, :, iz0]
                    else:
                        data_at_theta[i, :] = np.nan
            ret_data = data_at_theta.reshape(self.shape[0], self.shape[1], 1)
            ret_coords = self.coords.project_along(Axis.CARTESIAN_Z, zmed[imid].item())
        elif self.native_geometry is Geometry.SPHERICAL:
            thetamed = self.coords.get_axis_array_med(Axis.COLATITUDE)
            ret_coords = self.coords.project_along(
                Axis.COLATITUDE, thetamed[imid].item()
            )
            ret_data = self.data[
                :, find_nearest(thetamed, np.pi / 2 - theta), :
            ].reshape(self.shape[0], 1, self.shape[2])
        else:
            assert_never(self.native_geometry)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def vertical_at_z(self, z=0.0, *, operation_name=None) -> "GasField":
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix=f"vertical_at_z{z}",
            operation_name=operation_name,
        )
        imid = self.find_imid(altitude=z)
        if self.native_geometry is Geometry.CARTESIAN:
            zmed = self.coords.get_axis_array(Axis.CARTESIAN_Z)
            ret_coords = self.coords.project_along(Axis.CARTESIAN_Z, zmed[imid].item())
            ret_data = self.data[:, :, find_nearest(zmed, z)].reshape(
                self.shape[0], self.shape[1], 1
            )
        elif self.native_geometry is Geometry.POLAR:
            zmed = self.coords.get_axis_array(Axis.CARTESIAN_Z)
            ret_coords = self.coords.project_along(Axis.CARTESIAN_Z, zmed[imid].item())
            ret_data = self.data[:, :, find_nearest(zmed, z)].reshape(
                self.shape[0], self.shape[1], 1
            )
        elif self.native_geometry is Geometry.SPHERICAL:
            raise NotImplementedError(
                "vertical at z in spherical coordinates not implemented yet."
            )
        else:
            assert_never(self.native_geometry)

        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def azimuthal_at_phi(self, phi=0.0, *, operation_name=None) -> "GasField":
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix=f"azimuthal_at_phi{phi}",
            operation_name=operation_name,
        )
        iphi = self.find_iphi(phi=phi)
        if self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_at_phi"
            )
        elif self.native_geometry is Geometry.POLAR:
            phimed = self.coords.get_axis_array(Axis.AZIMUTH)
            ret_coords = self.coords.project_along(Axis.AZIMUTH, phimed[iphi].item())
            ret_data = self.data[:, iphi, :].reshape(self.shape[0], 1, self.shape[2])
        elif self.native_geometry is Geometry.SPHERICAL:
            phimed = self.coords.get_axis_array(Axis.AZIMUTH)
            ret_coords = self.coords.project_along(Axis.AZIMUTH, phimed[iphi].item())
            ret_data = self.data[:, :, iphi].reshape(self.shape[0], self.shape[1], 1)
        else:
            assert_never(self.native_geometry)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def azimuthal_at_planet(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
        operation_name=None,
    ) -> "GasField":
        planet_file = _parse_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        del planet_number

        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix="azimuthal_at_planet",
            operation_name=operation_name,
        )

        phip = self.find_phip(planet_file=planet_file)
        aziphip = self.azimuthal_at_phi(phi=phip)
        return GasField(
            self.field,
            aziphip.data,
            aziphip.coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def azimuthal_average(self, *, operation_name=None) -> "GasField":
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix="azimuthal_average",
            operation_name=operation_name,
        )

        iphi = self.find_iphi(phi=0)
        if self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_average"
            )
        elif self.native_geometry is Geometry.POLAR:
            phimed = self.coords.get_axis_array_med(Axis.AZIMUTH)
            ret_coords = self.coords.project_along(Axis.AZIMUTH, phimed[iphi].item())
            ret_data = np.nanmean(self.data, axis=1, dtype="float64").reshape(
                self.shape[0], 1, self.shape[2]
            )
        elif self.native_geometry is Geometry.SPHERICAL:
            phimed = self.coords.get_axis_array_med(Axis.AZIMUTH)
            ret_coords = self.coords.project_along(Axis.AZIMUTH, phimed[iphi].item())
            ret_data = np.nanmean(self.data, axis=2, dtype="float64").reshape(
                self.shape[0], self.shape[1], 1
            )
        else:
            assert_never(self.native_geometry)
        return GasField(
            self.field,
            ret_data.astype("float32", copy=False),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def remove_planet_hill_band(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
        operation_name=None,
    ) -> "GasField":
        planet_file = _parse_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        del planet_number

        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix="remove_planet_hill_band",
            operation_name=operation_name,
        )

        phip = self.find_phip(planet_file=planet_file)
        rp = self.find_rp(planet_file=planet_file)
        rhill = self.find_rhill(planet_file=planet_file)
        iphip_m = self.find_iphi(phi=phip - 2 * rhill / rp)
        iphip_p = self.find_iphi(phi=phip + 2 * rhill / rp)
        if self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_average_except_planet_hill"
            )
        elif self.native_geometry is Geometry.POLAR:
            ret_coords = self.coords
            ret_data = self.data.copy()
            if iphip_p >= iphip_m and iphip_p != self.coords.shape[1]:
                ret_data[:, iphip_m : iphip_p + 1, :] = np.nan
            else:
                if iphip_p == self.coords.shape[1]:
                    ret_data[:, iphip_m:iphip_p, :] = np.nan
                else:
                    ret_data[:, 0 : iphip_p + 1, :] = np.nan
                    ret_data[:, iphip_m : self.coords.shape[1], :] = np.nan
        elif self.native_geometry is Geometry.SPHERICAL:
            ret_coords = self.coords
            ret_data = self.data.copy()
            if iphip_p >= iphip_m and iphip_p != self.coords.shape[2]:
                ret_data[:, :, iphip_m : iphip_p + 1] = np.nan
            else:
                if iphip_p == self.coords.shape[2]:
                    ret_data[:, :, iphip_m:iphip_p] = np.nan
                else:
                    ret_data[:, :, 0 : iphip_p + 1] = np.nan
                    ret_data[:, :, iphip_m : self.coords.shape[2]] = np.nan
        else:
            assert_never(self.native_geometry)

        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def radial_at_r(self, distance=1.0, *, operation_name=None) -> "GasField":
        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix=f"radial_at_r{distance}",
            operation_name=operation_name,
        )

        ir1 = self.find_ir(distance=distance)
        if self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for radial_at_r"
            )
        elif self.native_geometry is Geometry.POLAR:
            rmed = self.coords.get_axis_array_med(Axis.CYLINDRICAL_RADIUS)
            ret_coords = self.coords.project_along(
                Axis.CYLINDRICAL_RADIUS, rmed[ir1].item()
            )
        elif self.native_geometry is Geometry.SPHERICAL:
            rmed = self.coords.get_axis_array_med(Axis.SPHERICAL_RADIUS)
            ret_coords = self.coords.project_along(
                Axis.SPHERICAL_RADIUS, rmed[ir1].item()
            )
        else:
            assert_never(self.native_geometry)
        ret_data = self.data[ir1, :, :].reshape(1, self.shape[1], self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def radial_average_interval(
        self, vmin=None, vmax=None, *, operation_name=None
    ) -> "GasField":
        if (vmin is None) or (vmax is None):
            raise ValueError(
                f"The radial interval {vmin=} and {vmax=} should be defined"
            )

        operation = self._parse_operation_name(
            prefix=self.operation,
            default_suffix=f"radial_average_interval_{vmin}_{vmax}",
            operation_name=operation_name,
        )

        irmin = self.find_ir(distance=vmin)
        irmax = self.find_ir(distance=vmax)
        ir = self.find_ir(distance=(vmax - vmin) / 2)
        if self.native_geometry is Geometry.CARTESIAN:
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for radial_at_r"
            )
        elif self.native_geometry is Geometry.POLAR:
            R = self.coords.get_axis_array(Axis.CYLINDRICAL_RADIUS)
            if vmin is None:
                vmin = R.min()
            if vmax is None:
                vmax = R.max()
            Rmed = self.coords.get_axis_array_med(Axis.CYLINDRICAL_RADIUS)
            ret_coords = self.coords.project_along(
                Axis.CYLINDRICAL_RADIUS, Rmed[ir].item()
            )
        elif self.native_geometry is Geometry.SPHERICAL:
            r = self.coords.get_axis_array(Axis.SPHERICAL_RADIUS)
            if vmin is None:
                vmin = r.min()
            if vmax is None:
                vmax = r.max()
            rmed = self.coords.get_axis_array_med(Axis.SPHERICAL_RADIUS)
            ret_coords = self.coords.project_along(
                Axis.SPHERICAL_RADIUS, rmed[ir].item()
            )
        else:
            assert_never(self.native_geometry)

        ret_data = np.nanmean(
            self.data[irmin : irmax + 1, :, :], axis=0, dtype="float64"
        ).reshape(1, self.shape[1], self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def diff(self, on_2) -> "GasField":
        ds_2 = GasDataSet(
            on_2,
            geometry=self.native_geometry,
            inifile=self.inifile,
            directory=self.directory,
        )
        if self.operation != "":
            raise KeyError(
                "For now, diff should only be applied on the initial Field cube."
            )
        # self.operation += "diff"
        ret_data = (self.data - ds_2[self.field].data) / ds_2[self.field].data
        ret_coords = self.coords
        # self.field = r"$\frac{%s - %s_0}{%s_0}$" % (self.field, self.field, self.field)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            self.operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def rotate(
        self,
        planet_corotation: int | None = None,
        *,
        rotate_with: str | None = None,
        rotate_by: float | None = None,
    ) -> "GasField":
        rotate_by = _parse_rotation_angle(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_number_argument=("planet_corotation", planet_corotation),
            planet_azimuth_finder=self,
            stacklevel=2,
        )

        operation = self.operation
        if self.shape.count(1) > 1:
            raise ValueError("data has to be 2D or 3D in order to rotate the data.")
        if not _fequal(self._rotate_by, rotate_by):
            phicoord = self.coords.get_axis_array(Axis.AZIMUTH) - rotate_by
            ipi = find_nearest(phicoord, 0)
            if abs(0 - phicoord[ipi]) > abs(np.ediff1d(find_around(phicoord, 0))[0]):
                ipi = find_nearest(phicoord, 2 * np.pi)

            if self.native_geometry is Geometry.POLAR:
                ret_data = np.roll(self.data, -ipi + 1, axis=1)
            elif self.native_geometry is Geometry.SPHERICAL:
                ret_data = np.roll(self.data, -ipi + 1, axis=2)
            else:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet if corotation"
                )
            self._rotate_by = rotate_by
        else:
            ret_data = self.data

        return GasField(
            self.field,
            ret_data,
            deepcopy(self.coords),
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )


class GasDataSet:
    """Idefix dataset class that contains everything in the .vtk file

    Args:
        input_dataset (int or str): output number or file name
        directory (str): directory of the .vtk
        geometry (str): for retrocompatibility if old vtk format
        inifile (str): name of the simulation's parameter file if no default files (combined with code)
        code (str): name of the code ("idefix", "pluto", "fargo3d", "fargo-adsg")
    Returns:
        dataset
    """

    def __init__(
        self,
        input_dataset: int | PathT,
        /,
        *,
        inifile: PathT | None = None,
        code: str | Recipe | None = None,
        geometry: str | None = None,
        directory: PathT | None = None,
        fluid: str | None = None,
        operation: str | None = None,
    ) -> None:
        if isinstance(input_dataset, str | Path):
            input_dataset = Path(input_dataset)
            directory_from_input = input_dataset.parent
            if directory is None:
                directory = directory_from_input
            elif directory_from_input.resolve() != Path(directory).resolve():
                raise ValueError(
                    f"directory value {directory!r} does not match "
                    f"directory name from input_dataset ({directory_from_input!r})"
                )
            del directory_from_input

        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)

        recipe = recipe_from(
            code=code,
            parameter_file=inifile,
            directory=directory,
        )

        if fluid is not None and recipe is not Recipe.FARGO3D:
            warnings.warn(
                "Unused keyword argument: 'fluid'",
                category=UserWarning,
                stacklevel=2,
            )

        loader = loader_from(
            code=code,
            parameter_file=inifile,
            directory=directory,
        )
        if operation is not None:
            ignored_kwargs = []
            msg = ""
            if fluid is not None:
                ignored_kwargs.append("fluid")
            if geometry is not None:
                ignored_kwargs.append("geometry")
            if ignored_kwargs:
                ignored = ", ".join(repr(_) for _ in ignored_kwargs)
                msg = (
                    "The following keyword arguments are ignored "
                    f"when combined with 'operation': {ignored}"
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
            self._loader = dataclasses.replace(loader, binary_reader=NPYReader)
        else:
            self._loader = loader

        self.on, datafile = self._loader.binary_reader.parse_output_number_and_filename(
            input_dataset,
            directory=directory,
            prefix=operation or "",
        )

        self._read = self._loader.load_bin_data(
            datafile,
            geometry=geometry,
            fluid=fluid,
        )

        self.native_geometry = self._read.geometry
        self.dict = self._read.data
        self.coords = Coordinates(
            self.native_geometry,
            self._read.x1,
            self._read.x2,
            self._read.x3,
        )
        for key in self.dict:
            self.dict[key] = GasField(
                key,
                self.dict[key],
                self.coords,
                self.native_geometry,
                self.on,
                operation="",
                inifile=self._loader.parameter_file,
                code=recipe,
                directory=directory,
            )

        # backward compatibility for self.params
        self._parameters_input = {
            "inifile": inifile,
            "code": code.removesuffix("_vtk") if code is not None else None,
            "directory": directory,
        }

    @classmethod
    def from_npy(
        cls,
        on: int,
        *,
        inifile: PathT | None = None,
        code: str | Recipe | None = None,
        directory: PathT | None = None,
        operation: str,
    ) -> "GasDataSet":
        warnings.warn(
            "GasDataSet.from_npy is deprecated "
            "and will be removed in a future version. "
            "Instead, call GasDataSet(...) directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return GasDataSet(
            on,
            inifile=inifile,
            code=code,
            directory=directory,
            operation=operation,
        )

    def __getitem__(self, key) -> "GasField":
        if key in self.dict:
            return self.dict[key]
        else:
            raise KeyError

    def keys(self) -> KeysView[str]:
        """
        Returns
        =======
        keys of the dict
        """
        return self.dict.keys()

    def values(self) -> ValuesView["GasField"]:
        """
        Returns
        =======
        values of the dict
        """
        return self.dict.values()

    def items(self) -> ItemsView[str, "GasField"]:
        """
        Returns
        =======
        items of the dict
        """
        return self.dict.items()

    @property
    def nfields(self) -> int:
        """
        Returns
        =======
        The number of fields in the GasDataSet
        """
        return len(self.dict)
