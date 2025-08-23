__all__ = [
    "Fargo3DReader",
    "FargoADSGReader",
    "IdefixReader",
    "NullReader",
]
import re
from pathlib import Path
from typing import final

import numpy as np

from nonos._types import PathT, PlanetData


@final
class NullReader:
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        raise NotImplementedError(
            f"{directory} couldn't be read. The default reader class (NullReader) "
            "was previously selected, possibly by mistake ?"
        )

    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        raise NotImplementedError(
            f"{file} couldn't be read. The default reader class (NullReader) "
            "was previously selected, possibly by mistake ?"
        )


@final
class IdefixReader:
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        return sorted(directory.glob("planet*.dat"))

    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        dt, x, y, z, vx, vy, vz, q, t = np.loadtxt(file).T
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)


@final
class FargoReaderHelper:
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        return [
            fn
            for fn in sorted(directory.glob("planet*.dat"))
            if re.search(r"planet\d+.dat$", str(fn)) is not None
        ]


@final
class Fargo3DReader:
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        return FargoReaderHelper.get_planet_files(directory)

    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        dt, x, y, z, vx, vy, vz, q, t, *_ = np.loadtxt(file).T
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)


@final
class FargoADSGReader:
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        return FargoReaderHelper.get_planet_files(directory)

    @staticmethod
    def read(file: PathT, /) -> PlanetData:
        dt, x, y, vx, vy, q, _, _, t, *_ = np.loadtxt(file).T
        z = np.zeros_like(x)
        vz = np.zeros_like(vx)
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)
