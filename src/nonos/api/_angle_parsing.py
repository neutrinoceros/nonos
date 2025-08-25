import warnings
from math import isclose
from typing import Protocol


class PlanetAzimuthFinder(Protocol):
    def find_phip(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
    ) -> float: ...


def _parse_planet_file(
    *,
    planet_file: str | None = None,
    planet_number: int | None = None,
) -> str:
    if planet_number is not None and planet_file is not None:
        raise TypeError(
            "received both planet_number and planet_file arguments. "
            "Please pass at most one."
        )
    if planet_file is not None:
        return planet_file
    else:
        return f"planet{planet_number or 0}.dat"


def _parse_rotation_angle(
    *,
    rotate_by: float | None,
    rotate_with: str | None,
    planet_number_argument: tuple[str, int | None],
    planet_azimuth_finder: PlanetAzimuthFinder,
    stacklevel: int,
) -> float:
    planet_number_argname, planet_number = planet_number_argument
    defined_args = {rotate_with, rotate_by, planet_number} - {None}
    if not defined_args:
        # no rotation specified
        return 0.0

    if len(defined_args) > 1:
        raise TypeError(
            "Can only process one argument out of "
            f"(rotate_by, rotate_with, {planet_number_argname})"
        )

    # beyond this point, we know that exactly one parameter was specified,
    # let's funnel it down to a rotate_by form
    if planet_number is not None:
        warnings.warn(
            f"The {planet_number_argname} argument is deprecated and will be removed "
            "in a future version. Instead, please use either rotate_by (float) "
            "or rotate_with (str path to planet log file).",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
        rotate_with = _parse_planet_file(planet_number=planet_number)

    if rotate_with is not None:
        rotate_by = planet_azimuth_finder.find_phip(planet_file=rotate_with)

    if rotate_by is None:
        # this is never supposed to happen, but it's needed to convince mypy that
        # we will not return a None
        raise RuntimeError

    return rotate_by


def _fequal(a: float, b: float, /) -> bool:
    # a fuzzy single-precision floating point comparison
    return isclose(a, b, abs_tol=1e-7, rel_tol=1e-6)
