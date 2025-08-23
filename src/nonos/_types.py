__all__ = [
    "BinData",
    "BinReader",
    "FloatArray",
    "FrameType",
    "IniData",
    "IniReader",
    "OrbitalElements",
    "PathT",
    "PlanetData",
    "PlanetReader",
    "StrDict",
]
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, final

import numpy as np

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

if TYPE_CHECKING:
    from nonos._geometry import Geometry

PathT: TypeAlias = str | Path
StrDict: TypeAlias = dict[str, Any]

FloatArray: TypeAlias = "np.ndarray[Any, np.dtype[np.float32 | np.float64]]"


class FrameType(Enum):
    FIXED_FRAME = auto()
    CONSTANT_ROTATION = auto()
    PLANET_COROTATION = auto()


@final
@dataclass(frozen=True, eq=False, slots=True)
class BinData:
    data: StrDict
    geometry: "Geometry"
    x1: FloatArray
    x2: FloatArray
    x3: FloatArray

    @classmethod
    def default_init(cls):
        return {
            field.name: field.default for field in cls.__dataclass_fields__.values()
        }


@final
@dataclass(frozen=True, eq=False, slots=True)
class OrbitalElements:
    i: FloatArray
    e: FloatArray
    a: FloatArray


@final
@dataclass(frozen=True, eq=False)
class PlanetData:
    # fields that are required at __init__
    _init_attrs = ["x", "y", "z", "vx", "vy", "vz", "q", "t", "dt"]
    # additional derived field that can be computed on the fly
    _post_init_attrs = ["d"]
    __slots__ = _init_attrs + _post_init_attrs

    # cartesian position
    x: FloatArray
    y: FloatArray
    z: FloatArray

    # cartesian velocity
    vx: FloatArray
    vy: FloatArray
    vz: FloatArray

    # mass ratio (or mass in units of the central star's)
    q: FloatArray

    # time and timestep
    t: FloatArray
    dt: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "d", np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def get_orbital_elements(self, frame: FrameType) -> OrbitalElements:
        if frame is FrameType.FIXED_FRAME:
            hx = self.y * self.vz - self.z * self.vy
            hy = self.z * self.vx - self.x * self.vz
            hz = self.x * self.vy - self.y * self.vx
            hhor = np.hypot(hx, hy)

            h2 = hx * hx + hy * hy + hz * hz
            h = np.sqrt(h2)
            i = np.arcsin(hhor / h)

            d = object.__getattribute__(self, "d")
            Ax = self.vy * hz - self.vz * hy - (1.0 + self.q) * self.x / d
            Ay = self.vz * hx - self.vx * hz - (1.0 + self.q) * self.y / d
            Az = self.vx * hy - self.vy * hx - (1.0 + self.q) * self.z / d

            e = np.sqrt(Ax * Ax + Ay * Ay + Az * Az) / (1.0 + self.q)
            a = h * h / ((1.0 + self.q) * (1.0 - e * e))
            return OrbitalElements(i, e, a)
        elif frame is FrameType.CONSTANT_ROTATION:
            raise NotImplementedError(
                f"PlanetData.set_orbital_elements isn't implemented for {frame=}"
            )
        elif frame is FrameType.PLANET_COROTATION:
            # bug-for-bug compat
            return self.get_orbital_elements(FrameType.FIXED_FRAME)
        else:
            assert_never(frame)

    def get_rotational_rate(self) -> FloatArray:
        d = self.d  # type: ignore [attr-defined]
        return np.sqrt((1.0 + self.q) / pow(d, 3.0))


for key in PlanetData._post_init_attrs:
    PlanetData.__annotations__[key] = FloatArray


@final
@dataclass(frozen=True, slots=True)
class IniData:
    file: Path
    frame: FrameType
    rotational_rate: float
    output_time_interval: float
    meta: StrDict


class BinReader(Protocol):
    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: PathT | int,
        *,
        directory: PathT,
        prefix: str,
    ) -> tuple[int, Path]: ...

    @staticmethod
    def get_bin_files(directory: PathT, /) -> list[Path]: ...

    @staticmethod
    def read(file: PathT, /, **meta) -> BinData: ...


class PlanetReader(Protocol):
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]: ...

    @staticmethod
    def read(file: PathT, /) -> PlanetData: ...


class IniReader(Protocol):
    @staticmethod
    def read(file: PathT, /) -> IniData: ...
