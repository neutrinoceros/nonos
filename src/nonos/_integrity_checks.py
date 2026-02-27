__all__ = [
    "check_field_operands",
    "collect_dtype_exceptions",
    "collect_shape_exceptions",
    "compile_exceptions",
]
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

if TYPE_CHECKING:
    from nonos.api.analysis import Field


def collect_shape_exceptions(
    arr_shape: tuple[int, int, int],
    coordinates_shape: tuple[int, int, int],
    /,
) -> list[Exception]:
    exceptions: list[Exception] = []

    msg_template = (
        "shape mismatch along axis {axis}: array has {array_size} elements, "
        "while cell edges have {coord_size}. "
        "Expected cell edges to be exactly 1 element longer than array in each dimension."
    )
    for i, (s1, s2) in enumerate(zip(arr_shape, coordinates_shape, strict=True)):
        if s1 + 1 == s2:
            continue
        exceptions.append(
            TypeError(msg_template.format(axis=i, array_size=s1, coord_size=s2))
        )
    return exceptions


def collect_dtype_exceptions(*dtypes: np.dtype[Any]) -> list[Exception]:
    exceptions: list[Exception] = []
    if len(unique_kinds := {dt.kind for dt in dtypes}) > 1:
        exceptions.append(TypeError(f"mixed dtype kinds: {sorted(unique_kinds)}"))
    if len(unique_sizes := {dt.itemsize for dt in dtypes}) > 1:
        exceptions.append(TypeError(f"mixed dtype itemsizes: {sorted(unique_sizes)}"))
    return exceptions


def check_field_operands(f1: "Field[Any]", f2: "Field[Any]") -> Exception | None:
    exceptions: list[Exception] = []
    if f1.shape != f2.shape:
        exceptions.append(TypeError("operands have incompatible shapes"))
    elif f1.coordinates != f2.coordinates:
        exceptions.append(TypeError("operands have incompatible coordinates"))
    if f1.dtype.kind != f2.dtype.kind or f1.dtype.itemsize != f2.dtype.itemsize:
        exceptions.append(TypeError("operands have incompatible dtypes"))

    if len(exceptions) == 1:
        return exceptions[0]
    elif exceptions:
        return ExceptionGroup("multiple issues with operands", exceptions)

    return None


def compile_exceptions(
    top_msg: str,
    /,
    *candidate_groups: tuple[str, list[Exception]],
) -> Exception | None:
    exceptions: list[Exception] = []
    for group_msg, excs in candidate_groups:
        if len(excs) == 1:
            exceptions.append(excs[0])
        elif excs:
            exceptions.append(ExceptionGroup(group_msg, excs))

    if len(exceptions) == 1:
        return exceptions[0]
    elif exceptions:
        return ExceptionGroup(top_msg, exceptions)

    return None
