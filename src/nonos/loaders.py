__all__ = [
    "Loader",
    "Recipe",
    "loader_from",
    "recipe_from",
]
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, final

import numpy as np

import nonos._readers as readers
from nonos._types import BinReader, F, IniReader, PlanetReader

if TYPE_CHECKING:
    from nonos._types import BinData, IniData, PlanetData


@dataclass(slots=True, frozen=True, kw_only=True)
class Recipe(Generic[F]):
    binary_reader: type[BinReader[F]]
    planet_reader: type[PlanetReader[F]]
    ini_reader: type[IniReader]
    dtype: np.dtype[F]


BUILTIN_RECIPES = {
    "idefix-vtk": Recipe(
        binary_reader=readers.binary.VTKReader,
        planet_reader=readers.planet.IdefixReader,
        ini_reader=readers.ini.IdefixVTKReader,
        dtype=np.dtype(">f4"),
    ),
    "pluto-vtk": Recipe(
        binary_reader=readers.binary.VTKReader,
        planet_reader=readers.planet.NullReader,
        ini_reader=readers.ini.PlutoVTKReader,
        dtype=np.dtype(">f4"),
    ),
    "fargo3d": Recipe(
        binary_reader=readers.binary.Fargo3DReader,
        planet_reader=readers.planet.Fargo3DReader,
        ini_reader=readers.ini.Fargo3DReader,
        dtype=np.dtype("=f8"),
    ),
    "fargo-adsg": Recipe(
        binary_reader=readers.binary.FargoADSGReader,
        planet_reader=readers.planet.FargoADSGReader,
        ini_reader=readers.ini.FargoADSGReader,
        dtype=np.dtype("=f8"),
    ),
}


@final
@dataclass(frozen=True, eq=True, slots=True)
class Loader(Generic[F]):
    r"""
    A composable data loader interface.

    Loader instances are immutable and extremely lightweight as they do
    not hold any data other than a Path to a parameter file.
    All actual loading capabilities are deleguated to specialized readers.

    Parameters
    ----------
      parameter_file: Path
        path to an existing parameter file.
      binary_reader: type[BinReader]
        a class that implements the BinReader interface, as defined in nonos._types
      planet_reader: type[PlanetReader]
        a class that implements the PlanetReader interface, as defined in nonos._types
      ini_reader: type[IniReader]
        a class that implements the IniReader interface, as defined in nonos._types

    Raises
    ------
      FileNotFoundError: if `parameter_file` doesn't exist or is a directory.
    """

    parameter_file: Path
    binary_reader: type[BinReader[F]]
    planet_reader: type[PlanetReader[F]]
    ini_reader: type[IniReader]
    dtype: np.dtype[F]

    def __post_init__(self) -> None:
        pf = Path(self.parameter_file).resolve()
        if not pf.is_file():
            raise FileNotFoundError(pf)
        object.__setattr__(self, "parameter_file", pf)

    def load_bin_data(self, file: os.PathLike[str], /, **meta: Any) -> "BinData[F]":
        ini = self.load_ini_file()
        meta = ini.meta | meta
        return self.binary_reader.read(file, **meta)

    def load_planet_data(self, file: os.PathLike[str]) -> "PlanetData[F]":
        return self.planet_reader.read(file)

    def load_ini_file(self) -> "IniData":
        return self.ini_reader.read(self.parameter_file)


def loader_from(
    *,
    code: str | None = None,
    parameter_file: os.PathLike[str] | None = None,
    directory: os.PathLike[str] | None = None,
) -> Loader:
    r"""
    Compose a Loader object following a known Recipe.

    The exact Recipe needs to be uniquely identifiable from the parameters.

    Parameters
    ----------
      code: str (optional)
        This string should match a Recipe enum member.
        Lower case is expected.
        Valid values include, but are not necessarily limited to:
        - 'idefix_vtk'
        - 'pluto_vtk'
        - 'fargo-adsg'
        - 'fargo3d'

      parameter_file: Path or str (optional)
        A path to a parameter file (e.g. idefix.ini). This path can be
        absolute or relative to the `directory` argument.

      directory: Path or str (optional)
        A path to the simulation directory.

    Raises
    ------
      TypeError: if no argument is provided.
    """
    return _compose_loader(
        recipe_from(
            code=code,
            parameter_file=parameter_file,
            directory=directory,
        ),
        _parameter_file_from(
            parameter_file=parameter_file,
            directory=directory,
        ),
    )


def _compose_loader(recipe: Recipe[F], /, parameter_file: Path) -> Loader[F]:
    return Loader(parameter_file, **asdict(recipe))


def _parameter_file_from(
    *,
    parameter_file: os.PathLike[str] | None = None,
    directory: os.PathLike[str] | None = None,
) -> Path:
    if parameter_file is None and directory is None:
        raise TypeError(
            "Missing required keyword arguments: 'parameter_file', 'directory' "
            "(need at least one)"
        )

    if directory is not None:
        directory = Path(directory).resolve()
        if parameter_file is None:
            return _parameter_file_from_dir(directory)

    if parameter_file is not None:
        parameter_file = Path(parameter_file)
        if parameter_file.is_absolute():
            return parameter_file
        elif directory is not None and parameter_file == Path(parameter_file.name):
            return directory / parameter_file

    raise ValueError(
        f"Received apparently inconsistent inputs {parameter_file=} and {directory=}"
    )


def _parameter_file_from_dir(directory: os.PathLike[str], /) -> Path:
    directory = Path(directory).resolve()
    candidates = list(directory.glob("*.ini"))
    candidates.extend(list(directory.glob("*.par")))
    if len(candidates) == 1:
        return candidates[0]
    elif not candidates:
        raise FileNotFoundError(f"Could not find a parameter file in {directory}")
    else:
        raise RuntimeError(
            f"Found multiple parameter files in {directory}\n - "
            + "\n - ".join(str(c) for c in candidates)
        )


def recipe_from(
    *,
    code: str | None = None,
    parameter_file: os.PathLike[str] | None = None,
    directory: os.PathLike[str] | None = None,
) -> Recipe[Any]:
    r"""
    Determine an appropriate loader recipe from user input.

    Parameters
    ----------
      code: str (optional)
        This string should match a Recipe enum member.
        Lower case is expected.
        Valid values include, but are not necessarily limited to:
        - 'idefix_vtk'
        - 'pluto_vtk'
        - 'fargo-adsg'
        - 'fargo3d'

      parameter_file: Path or str (optional)
        A path to a parameter file (e.g. idefix.ini). This path can be
        absolute or relative to the `directory` argument.

      directory: Path or str (optional)
        A path to the simulation directory.

    Returns
    -------
       a Recipe enum member

    Raises
    ------
      TypeError: if no argument is provided.

      ValueError: if `code` is omitted and a working inifile reader cannot
        be uniquely identified.
    """
    if code is not None:
        return _code_to_recipe(code)

    parameter_file = _parameter_file_from(
        parameter_file=parameter_file,
        directory=directory,
    )

    recipe_candidates: list[Recipe[Any]] = []
    for recipe in BUILTIN_RECIPES.values():
        loader = _compose_loader(recipe, parameter_file)
        try:
            loader.load_ini_file()
        except Exception:
            continue
        else:
            recipe_candidates.append(recipe)
    if len(recipe_candidates) == 1:
        return recipe_candidates[0]
    elif len(recipe_candidates) == 0:
        msg = (
            f"Could not determine data format from {parameter_file=!r} "
            "(failed to read with any loader)"
        )
    else:  # pragma: no cover
        msg = (
            f"Could not determine unambiguous data format from {parameter_file=!r} "
            f"(found {len(recipe_candidates)} candidates {recipe_candidates})"
        )

    raise ValueError(msg)


def _code_to_recipe(code: str, /) -> Recipe[Any]:
    if code in ("pluto", "idefix"):
        # backward compatibility layer
        # this could be deprecated at some point
        new_code = f"{code}-vtk"
    else:
        new_code = code
    new_code = new_code.replace("_", "-")
    if new_code not in BUILTIN_RECIPES:
        raise ValueError(f"{code=!r} is not supported")
    return BUILTIN_RECIPES[new_code]
