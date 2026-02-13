__all__ = [
    "Fargo3DReader",
    "FargoADSGReader",
    "IdefixReader",
    "NullReader",
]
import os
import re
from pathlib import Path
from typing import Generic, final

import numpy as np

from nonos._types import F, PlanetData


@final
class NullReader(Generic[F]):
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        raise NotImplementedError(
            f"{directory} couldn't be read. The default reader class (NullReader) "
            "was previously selected, possibly by mistake ?"
        )

    @staticmethod
    def read(file: os.PathLike[str], /) -> PlanetData[F]:
        raise NotImplementedError(
            f"{file} couldn't be read. The default reader class (NullReader) "
            "was previously selected, possibly by mistake ?"
        )


@final
class IdefixReader(Generic[F]):
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        return sorted(directory.glob("planet*.dat"))

    @staticmethod
    def read(file: os.PathLike[str], /) -> PlanetData[F]:
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
class Fargo3DReader(Generic[F]):
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        return FargoReaderHelper.get_planet_files(directory)

    @staticmethod
    def read(file: os.PathLike[str], /) -> PlanetData[F]:
        dt, x, y, z, vx, vy, vz, q, t, *_ = np.loadtxt(file).T
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)


@final
class FargoADSGReader(Generic[F]):
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]:
        return FargoReaderHelper.get_planet_files(directory)

    @staticmethod
    def read(file: os.PathLike[str], /) -> PlanetData[F]:
        dt, x, y, vx, vy, q, _, _, t, *_ = np.loadtxt(file).T
        z = np.zeros_like(x)
        vz = np.zeros_like(vx)
        return PlanetData(x, y, z, vx, vy, vz, q, t, dt)
