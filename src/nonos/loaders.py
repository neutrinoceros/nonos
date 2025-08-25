__all__ = [
    "Loader",
    "Recipe",
    "loader_from",
    "recipe_from",
]
import sys
from dataclasses import dataclass
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TypedDict, final

import nonos._readers as readers
from nonos._types import BinReader, IniReader, PlanetReader

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import assert_never
else:
    from typing_extensions import assert_never

    from nonos._backports import StrEnum


if TYPE_CHECKING:
    from nonos._types import BinData, IniData, PathT, PlanetData


class Recipe(StrEnum):
    IDEFIX_VTK = auto()
    PLUTO_VTK = auto()
    FARGO3D = auto()
    FARGO_ADSG = auto()


@final
@dataclass(frozen=True, eq=True, slots=True)
class Loader:
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
    binary_reader: type["BinReader"]
    planet_reader: type["PlanetReader"]
    ini_reader: type["IniReader"]

    def __post_init__(self) -> None:
        pf = Path(self.parameter_file).resolve()
        if not pf.is_file():
            raise FileNotFoundError(pf)
        object.__setattr__(self, "parameter_file", pf)

    def load_bin_data(self, file: "PathT", /, **meta) -> "BinData":
        ini = self.load_ini_file()
        meta = ini.meta | meta
        return self.binary_reader.read(file, **meta)

    def load_planet_data(self, file: "PathT") -> "PlanetData":
        return self.planet_reader.read(file)

    def load_ini_file(self) -> "IniData":
        return self.ini_reader.read(self.parameter_file)


def loader_from(
    *,
    code: str | None = None,
    parameter_file: Optional["PathT"] = None,
    directory: Optional["PathT"] = None,
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
        - 'fargo_adsg'
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


def _compose_loader(recipe: Recipe, /, parameter_file: Path) -> Loader:
    return Loader(parameter_file, **_ingredients_from(recipe))


def _parameter_file_from(
    *,
    parameter_file: Optional["PathT"] = None,
    directory: Optional["PathT"] = None,
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


def _parameter_file_from_dir(directory: "PathT", /) -> Path:
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


class Ingredients(TypedDict):
    binary_reader: type[BinReader]
    planet_reader: type[PlanetReader]
    ini_reader: type[IniReader]


def _ingredients_from(recipe: Recipe, /) -> Ingredients:
    if recipe is Recipe.IDEFIX_VTK:
        return {
            "binary_reader": readers.binary.VTKReader,
            "planet_reader": readers.planet.IdefixReader,
            "ini_reader": readers.ini.IdefixVTKReader,
        }
    elif recipe is Recipe.PLUTO_VTK:
        return {
            "binary_reader": readers.binary.VTKReader,
            "planet_reader": readers.planet.NullReader,
            "ini_reader": readers.ini.PlutoVTKReader,
        }
    elif recipe is Recipe.FARGO3D:
        return {
            "binary_reader": readers.binary.Fargo3DReader,
            "planet_reader": readers.planet.Fargo3DReader,
            "ini_reader": readers.ini.Fargo3DReader,
        }
    elif recipe is Recipe.FARGO_ADSG:
        return {
            "binary_reader": readers.binary.FargoADSGReader,
            "planet_reader": readers.planet.FargoADSGReader,
            "ini_reader": readers.ini.FargoADSGReader,
        }
    else:
        assert_never(recipe)


def recipe_from(
    *,
    code: str | None = None,
    parameter_file: Optional["PathT"] = None,
    directory: Optional["PathT"] = None,
) -> Recipe:
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
        - 'fargo_adsg'
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

    recipe_candidates: list[Recipe] = []
    for recipe in Recipe.__members__.values():
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


def _code_to_recipe(code: str, /) -> Recipe:
    if code in ("pluto", "idefix"):
        # backward compatibility layer
        # this could be deprecated at some point
        code += "_vtk"
    return Recipe(code)
