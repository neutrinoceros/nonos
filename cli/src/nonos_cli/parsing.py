from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def is_set(x: Any) -> bool:
    return x not in (None, "unset")


T1 = TypeVar("T1")
T2 = TypeVar("T2")


def userval_or_default(userval: T1, /, *, default: T2) -> T1 | T2:
    # it'd be nice to avoid a Union as a return type, however it's not clear
    # how to express what this function does in typing language.
    # In practice this is used in places where it's very hard to constrain T1, so
    # the inferred return type always degrades down to Any anyway.
    if is_set(userval):
        return userval
    else:
        return default


def parse_output_number_range(
    on: list[int] | int | Literal["unset"] | None,
    maxval: int | None = None,
) -> list[int]:
    if not is_set(on):
        if maxval is None:
            raise ValueError("Can't parse a range from unset values without a max.")
        return [maxval]

    if isinstance(on, int):
        return [on]

    assert isinstance(on, list) and all(isinstance(o, int) for o in on)

    if len(on) > 3:
        raise ValueError(
            f"Can't parse a range from sequence {on} with more than 3 values."
        )
    if len(on) == 1:
        return on

    if on[1] < on[0]:
        raise ValueError("Can't parse a range with max < min.")

    # make the upper boundary inclusive
    on[1] += 1
    ret = list(range(*on))
    if maxval is not None and (max_requested := ret[-1]) > maxval:
        raise ValueError(
            f"No output beyond {maxval} is available, but {max_requested} was requested."
        )
    return ret


@overload
def parse_range(
    extent: Literal["unset"],
    dim: Literal[1],
) -> tuple[None, None]: ...
@overload
def parse_range(
    extent: Literal["unset"],
    dim: Literal[2],
) -> tuple[None, None, None, None]: ...
@overload
def parse_range(
    extent: tuple[str, str],
    dim: Literal[1],
) -> tuple[float | None, float | None]: ...
@overload
def parse_range(
    extent: tuple[str, str, str, str],
    dim: Literal[2],
) -> tuple[float | None, float | None, float | None, float | None]: ...


def parse_range(extent, dim):
    if not is_set(extent):
        return (None,) * 2 * dim

    if len(extent) != 2 * dim:
        raise ValueError(
            f"Received sequence `extent` with incorrect size {len(extent)}. Expected exactly {2*dim=} values."
        )
    return tuple(float(i) if i != "x" else None for i in extent)


@overload
def range_converter(
    extent: tuple[float | None, float | None],
    abscissa: "NDArray[np.floating]",
    ordinate: "NDArray[np.floating]",
) -> tuple[float, float]: ...


@overload
def range_converter(
    extent: tuple[float | None, float | None, float | None, float | None],
    abscissa: "NDArray[np.floating]",
    ordinate: "NDArray[np.floating]",
) -> tuple[float, float, float, float]: ...


def range_converter(extent, abscissa, ordinate):
    if len(extent) == 4:
        return (
            _ if (_ := extent[0]) is not None else abscissa.min(),
            _ if (_ := extent[1]) is not None else abscissa.max(),
            _ if (_ := extent[2]) is not None else ordinate.min(),
            _ if (_ := extent[3]) is not None else ordinate.max(),
        )
    elif len(extent) == 2:
        return (
            _ if (_ := extent[0]) is not None else abscissa.min(),
            _ if (_ := extent[1]) is not None else abscissa.max(),
        )
    else:
        raise TypeError(f"Expected extent to be of length 2 or 4, got {len(extent)=}")


def parse_image_format(s: str | None) -> str:
    from matplotlib.backend_bases import FigureCanvasBase

    if not is_set(s):
        return FigureCanvasBase.get_default_filetype()

    assert isinstance(s, str)
    _, _, ext = s.rpartition(".")
    if ext not in (
        available := list(FigureCanvasBase.get_supported_filetypes().keys())
    ):
        raise ValueError(
            f"Received unknown file format '{s}'. Available formatted are {available}."
        )
    return ext
