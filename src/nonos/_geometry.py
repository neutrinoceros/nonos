from __future__ import annotations

__all__ = [
    "Axis",
    "Coordinates",
    "Geometry",
    "axes_from_geometry",
]
import sys
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import final, overload

import numpy as np

from nonos._types import FloatArray, StrDict

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import assert_never
else:
    from typing_extensions import assert_never

    from nonos._backports import StrEnum


class Geometry(StrEnum):
    CARTESIAN = auto()
    POLAR = auto()
    SPHERICAL = auto()


class Axis(Enum):
    CARTESIAN_X = auto()
    CARTESIAN_Y = auto()
    CARTESIAN_Z = auto()
    AZIMUTH = auto()
    CYLINDRICAL_RADIUS = auto()
    SPHERICAL_RADIUS = auto()
    COLATITUDE = auto()

    @property
    def label(self) -> str:
        return _AXIS_TO_STR[self]

    @classmethod
    def from_label(cls, label: str) -> Axis:
        return _STR_TO_AXIS[label]


_AXIS_TO_STR = {
    Axis.CARTESIAN_X: "x",
    Axis.CARTESIAN_Y: "y",
    Axis.CARTESIAN_Z: "z",
    Axis.AZIMUTH: "phi",
    Axis.CYLINDRICAL_RADIUS: "R",
    Axis.SPHERICAL_RADIUS: "r",
    Axis.COLATITUDE: "theta",
}
_STR_TO_AXIS = {v: k for k, v in _AXIS_TO_STR.items()}
assert all(axis in _AXIS_TO_STR for axis in Axis)


def axes_from_geometry(geometry: Geometry, /) -> tuple[Axis, Axis, Axis]:
    if geometry is Geometry.CARTESIAN:
        return (Axis.CARTESIAN_X, Axis.CARTESIAN_Y, Axis.CARTESIAN_Z)
    elif geometry is Geometry.POLAR:
        return (Axis.CYLINDRICAL_RADIUS, Axis.AZIMUTH, Axis.CARTESIAN_Z)
    elif geometry is Geometry.SPHERICAL:
        return (Axis.SPHERICAL_RADIUS, Axis.COLATITUDE, Axis.AZIMUTH)
    else:
        assert_never(geometry)


def _get_target_geometry(*axes: Axis) -> Geometry:
    axes_set = set(axes)
    if len(axes_set) == 0 or len(axes_set) > 3:
        raise RuntimeError

    candidates: set[Geometry] = set()
    for geom in Geometry:
        if set(axes_from_geometry(geom)).issuperset(axes_set):
            candidates.add(geom)

    if len(candidates) == 0:
        raise RuntimeError
    elif len(candidates) > 1:
        # supposedly unreachable
        raise RuntimeError

    return candidates.pop()


def _native_axis_from_target_axis(
    native_geometry: Geometry,
    axis: Axis,
):
    if axis in axes_from_geometry(native_geometry):
        return axis

    # shorthands are locally helpful for readability
    X = Axis.CARTESIAN_X
    Y = Axis.CARTESIAN_Y
    Z = Axis.CARTESIAN_Z
    PHI = Axis.AZIMUTH
    CR = Axis.CYLINDRICAL_RADIUS
    SR = Axis.SPHERICAL_RADIUS
    THETA = Axis.COLATITUDE

    if native_geometry is Geometry.CARTESIAN:
        pass

    elif native_geometry is Geometry.SPHERICAL:
        # to CARTESIAN
        if axis is X:
            return SR
        if axis is Y:
            return PHI  # grotesque
        if axis is Z:
            return THETA

        # to POLAR
        if axis is CR:
            return SR

    elif native_geometry is Geometry.POLAR:
        # to CARTESIAN
        if axis is X:
            return CR
        if axis is Y:
            return PHI  # grotesque

        # to SPHERICAL
        if axis is SR:
            return CR
        if axis is THETA:
            return Z

    else:
        assert_never(native_geometry)

    raise NotImplementedError(
        f"Transformation from {native_geometry} to {axis} is not implemented"
    )


def _native_plane_from_target_plane(
    native_geometry: Geometry,
    axis_1: Axis,
    axis_2: Axis,
    _recurse: bool = True,
) -> tuple[Axis, Axis]:
    # this replaces Coordinates.native_from_wanted to a certain extent:
    # - I only care about 2D cases
    # - I assume input axis are already validated
    #   (i.e. are members from a uniquely defined geometry)
    #
    # I'll start with a bug-for-bug approach, but I believe that most cases
    #   are actually broken:
    #
    # There are 3 geometries to convert to and from, all with 3 axes.
    # Choosing a target plane requires 2 axes, which gives 3 possible planes
    # per target geometry => 3^3 = 27 (order insensitive) cases
    # I use the following categories:
    #   * native: trivially correct (transform is identity) (9/27)
    #   * not implemented: cartesian to anything different isn't supported (6/27)
    #   * genuine: unambiguously correct (3/27)
    #   * no comment: not 100% correct but arguably useful (7/27)
    #   * grotesque: pretty sure this is never wanted (2/27)
    #
    # "native" and "genuine" cases are the only categories where no information
    # is lost in conversion. In the general case we need all 3 axes from native
    # geometry to project to a well-defined plane in another geometry.

    input_tuple = (axis_1, axis_2)
    if set(input_tuple).issubset(axes_from_geometry(native_geometry)):
        return input_tuple

    # shorthands are locally helpful for readability
    X = Axis.CARTESIAN_X
    Y = Axis.CARTESIAN_Y
    Z = Axis.CARTESIAN_Z
    PHI = Axis.AZIMUTH
    CR = Axis.CYLINDRICAL_RADIUS
    SR = Axis.SPHERICAL_RADIUS
    THETA = Axis.COLATITUDE

    if native_geometry is Geometry.CARTESIAN:
        pass

    elif native_geometry is Geometry.SPHERICAL:
        # to CARTESIAN
        if input_tuple == (X, Y):
            return (SR, PHI)
        if input_tuple == (X, Z):
            return (SR, THETA)
        if input_tuple == (Y, Z):
            return (PHI, THETA)  # grotesque

        # to POLAR
        if input_tuple == (CR, PHI):
            return (SR, PHI)
        if input_tuple == (CR, Z):
            return (SR, THETA)  # genuine
        if input_tuple == (PHI, Z):
            return (PHI, THETA)

    elif native_geometry is Geometry.POLAR:
        # to CARTESIAN
        if input_tuple == (X, Y):
            return (CR, PHI)  # genuine
        if input_tuple == (X, Z):
            return (CR, Z)
        if input_tuple == (Y, Z):
            return (PHI, Z)  # grotesque

        # to SPHERICAL
        if input_tuple == (SR, PHI):
            return (CR, PHI)
        if input_tuple == (SR, THETA):
            return (CR, Z)  # genuine
        if input_tuple == (PHI, THETA):
            return (PHI, Z)

    else:
        assert_never(native_geometry)

    if _recurse:
        # order matters. Rercursion allows use to implement only half of all cases
        out_2, out_1 = _native_plane_from_target_plane(
            native_geometry,
            axis_2,
            axis_1,
            _recurse=False,
        )
        return out_1, out_2

    # note that we can only land here after recursion/input inversion,
    # so we revert the order of arguments back to was was passed from
    # the outside
    raise NotImplementedError(
        f"Transformation from {native_geometry} to ({axis_2}, {axis_1}) "
        "is not implemented"
    )


def _deprecated_axis_array_property(attr: str, label: str, replacement: str):
    def _wrapped(self) -> FloatArray:
        warnings.warn(
            f"Coordinates.{attr} is deprecated. "
            f"Instead, use Coordinates.{replacement}({label!r})",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, replacement)(label)
        except KeyError:
            raise AttributeError(
                f"Coordinates object has no attribute {attr}"
            ) from None

    return property(_wrapped)


@final
@dataclass(frozen=True, eq=False, slots=True)
class Coordinates:
    geometry: Geometry
    x1: FloatArray
    x2: FloatArray
    x3: FloatArray

    def __post_init__(self) -> None:
        if isinstance(self.geometry, str):
            # for convenience
            object.__setattr__(self, "geometry", Geometry(self.geometry))

        x1 = self.x1
        x2 = self.x2
        x3 = self.x3
        if x1.ndim != 1 or x2.ndim != 1 or x3.ndim != 1:
            raise ValueError(
                "Expected input arrays to all be 1D. "
                f"Got {x1.ndim=}, {x2.ndim=}, {x3.ndim=}"
            )
        for attr in ("x1", "x2", "x3"):
            arr = getattr(self, attr)
            if len(arr) == 1:
                object.__setattr__(self, attr, np.repeat(arr, 2))

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.x1), len(self.x2), len(self.x3)

    @property
    def axes(self) -> tuple[Axis, Axis, Axis]:
        return axes_from_geometry(self.geometry)

    def get_axis_index(self, axis: Axis) -> int:
        axes = list(self.axes)
        if axis not in axes:
            raise ValueError(f"axis {axis} isn't native to {self.geometry} geometry")
        return axes.index(axis)

    def get_axis_array(self, axis: Axis | str) -> FloatArray:
        if isinstance(axis, str):
            axis = Axis.from_label(axis)
        idx = self.get_axis_index(axis)
        if idx == 0:
            arr = self.x1
        elif idx == 1:
            arr = self.x2
        elif idx == 2:
            arr = self.x3
        else:
            raise RuntimeError
        return arr.copy()

    def get_axis_array_med(self, axis: Axis | str) -> FloatArray:
        # convenience compatibility shim
        # should probably be removed or refactored
        if isinstance(axis, str):
            axis = Axis.from_label(axis)
        arr = self.get_axis_array(axis)
        return 0.5 * (arr[1:] + arr[:-1])

    def project_along(self, axis: Axis, position: float) -> Coordinates:
        from nonos.api.tools import find_around

        idx = self.get_axis_index(axis)
        arrs = [self.get_axis_array(ax) for ax in self.axes]
        arrs[idx] = find_around(arrs[idx], position)
        return Coordinates(self.geometry, *arrs)

    def to_dict(self) -> StrDict:
        return {"geometry": self.geometry} | {
            axis.label: self.get_axis_array(axis) for axis in self.axes
        }

    @overload
    def native_from_wanted(self, axis_1: Axis, axis_2: None = None, /) -> Axis: ...
    @overload
    def native_from_wanted(
        self, axis_1: Axis, axis_2: Axis, /
    ) -> tuple[Axis, Axis]: ...

    def native_from_wanted(self, axis_1, axis_2=None, /):
        # compatibility shim for old Coordinates class
        # TODO: rename this
        if axis_2 is not None:
            return _native_plane_from_target_plane(self.geometry, axis_1, axis_2)
        else:
            return _native_axis_from_target_axis(self.geometry, axis_1)

    def _meshgrid_conversion_1d(self, axis_1: Axis) -> dict[Axis, FloatArray]:
        native_plane_axis = self.native_from_wanted(axis_1)
        native_meshcoords = self._meshgrid_reduction(native_plane_axis, None)
        return native_meshcoords

    def _meshgrid_conversion_2d(
        self, axis_1: Axis, axis_2: Axis
    ) -> dict[Axis, FloatArray]:
        # TODO: fix interface
        native_plane_axes = self.native_from_wanted(axis_1, axis_2)
        native_meshcoords = self._meshgrid_reduction(*native_plane_axes)
        target_geometry = _get_target_geometry(axis_1, axis_2)
        return self.target_from_native(target_geometry, native_meshcoords)

    def _meshgrid_reduction(
        self, axis_1: Axis, axis_2: Axis | None, /
    ) -> dict[Axis, FloatArray]:
        # TODO: this could easily be split into 2 functions (one for each dimensions)
        axes = axis_1, axis_2
        gaxes = axes_from_geometry(self.geometry)
        if axis_1 not in gaxes:
            raise ValueError(f"expected one of {gaxes}, got {axis_1}")
        if axis_2 is not None and axis_2 not in gaxes:
            raise ValueError(f"expected one of {gaxes}, got {axis_2}")

        dictmesh: dict[Axis, FloatArray] = {}
        if axis_2 is None:
            # 1D case
            dictmesh[axis_1] = self.get_axis_array_med(axis_1)
        else:
            # 2D case
            dictcoords: dict[Axis, FloatArray] = {
                axis_1: self.get_axis_array(axis_1),
                axis_2: self.get_axis_array(axis_2),
            }
            dictmesh[axis_1], dictmesh[axis_2] = np.meshgrid(
                dictcoords[axis_1], dictcoords[axis_2]
            )
            normal = set(gaxes).difference(axes).pop()
            dictmesh[normal] = self.get_axis_array_med(normal)

        return dictmesh

    def target_from_native(
        self,
        target_geometry: Geometry,
        coords: dict[Axis, FloatArray],
    ) -> dict[Axis, FloatArray]:
        if self.geometry is Geometry.CARTESIAN:
            x = coords[Axis.CARTESIAN_X]
            y = coords[Axis.CARTESIAN_Y]
            z = coords[Axis.CARTESIAN_Z]
            if target_geometry is Geometry.CARTESIAN:
                return {
                    Axis.CARTESIAN_X: x,
                    Axis.CARTESIAN_Y: y,
                    Axis.CARTESIAN_Z: z,
                }
            elif target_geometry is Geometry.POLAR:
                return {
                    Axis.CYLINDRICAL_RADIUS: np.sqrt(x**2 + y**2),
                    Axis.AZIMUTH: np.arctan2(y, x),
                    Axis.CARTESIAN_Z: z,
                }
            elif target_geometry is Geometry.SPHERICAL:
                return {
                    Axis.SPHERICAL_RADIUS: np.sqrt(x**2 + y**2 + z**2),
                    Axis.COLATITUDE: np.arctan2(np.sqrt(x**2 + y**2), z),
                    Axis.AZIMUTH: np.arctan2(y, x),
                }
            else:
                assert_never(target_geometry)
                raise ValueError(f"Unknown target geometry {target_geometry}.")

        elif self.geometry is Geometry.SPHERICAL:
            r = coords[Axis.SPHERICAL_RADIUS]
            theta = coords[Axis.COLATITUDE]
            phi = coords[Axis.AZIMUTH]
            if target_geometry is Geometry.CARTESIAN:
                # note: I'm intentionally not reproducing a
                # special case that I don't think is needed (theta.ndim<=1)
                if theta.ndim <= 1:
                    # bug-for-bug compat
                    # this is extremely suspicious and should be inspected
                    # more thoroughly, I suspect it's just working around
                    # a completely different bug
                    return {
                        Axis.CARTESIAN_X: r * np.sin(theta) * np.cos(phi),
                        Axis.CARTESIAN_Y: r * np.sin(theta) * np.sin(phi),
                        Axis.CARTESIAN_Z: np.cos(theta),
                    }
                return {
                    Axis.CARTESIAN_X: r * np.sin(theta) * np.cos(phi),
                    Axis.CARTESIAN_Y: r * np.sin(theta) * np.sin(phi),
                    Axis.CARTESIAN_Z: r * np.cos(theta),
                }
            elif target_geometry is Geometry.POLAR:
                return {
                    Axis.CYLINDRICAL_RADIUS: r * np.sin(theta),
                    Axis.AZIMUTH: phi,
                    Axis.CARTESIAN_Z: r * np.cos(theta),
                }
            elif target_geometry is Geometry.SPHERICAL:
                return {
                    Axis.SPHERICAL_RADIUS: r,
                    Axis.COLATITUDE: theta,
                    Axis.AZIMUTH: phi,
                }
            else:
                assert_never(target_geometry)

        elif self.geometry is Geometry.POLAR:
            R = coords[Axis.CYLINDRICAL_RADIUS]
            phi = coords[Axis.AZIMUTH]
            z = coords[Axis.CARTESIAN_Z]
            if target_geometry is Geometry.CARTESIAN:
                return {
                    Axis.CARTESIAN_X: R * np.cos(phi),
                    Axis.CARTESIAN_Y: R * np.sin(phi),
                    Axis.CARTESIAN_Z: z,
                }
            elif target_geometry is Geometry.POLAR:
                return {
                    Axis.CYLINDRICAL_RADIUS: R,
                    Axis.AZIMUTH: phi,
                    Axis.CARTESIAN_Z: z,
                }
            elif target_geometry is Geometry.SPHERICAL:
                return {
                    Axis.SPHERICAL_RADIUS: np.sqrt(R**2 + z**2),
                    Axis.COLATITUDE: np.arctan2(R, z),
                    Axis.AZIMUTH: phi,
                }
            else:
                assert_never(target_geometry)

        else:
            assert_never(self.geometry)

    x = _deprecated_axis_array_property("x", "x", "get_axis_array")
    y = _deprecated_axis_array_property("y", "y", "get_axis_array")
    z = _deprecated_axis_array_property("z", "z", "get_axis_array")
    r = _deprecated_axis_array_property("r", "r", "get_axis_array")
    theta = _deprecated_axis_array_property("theta", "theta", "get_axis_array")
    phi = _deprecated_axis_array_property("phi", "phi", "get_axis_array")
    R = _deprecated_axis_array_property("R", "R", "get_axis_array")

    xmed = _deprecated_axis_array_property("xmed", "x", "get_axis_array_med")
    ymed = _deprecated_axis_array_property("ymed", "y", "get_axis_array_med")
    zmed = _deprecated_axis_array_property("zmed", "z", "get_axis_array_med")
    rmed = _deprecated_axis_array_property("rmed", "r", "get_axis_array_med")
    thetamed = _deprecated_axis_array_property(
        "thetamed", "theta", "get_axis_array_med"
    )
    phimed = _deprecated_axis_array_property("phimed", "phi", "get_axis_array_med")
    Rmed = _deprecated_axis_array_property("Rmed", "R", "get_axis_array_med")
