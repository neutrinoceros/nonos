__all__ = [
    "collect_dtype_exceptions",
    "collect_shape_exceptions",
    "compile_exceptions",
]
import sys
from typing import Any

import numpy as np

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


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


def collect_dtype_exceptions(
    dt1: np.dtype[Any],
    dt2: np.dtype[Any],
    /,
) -> list[Exception]:
    exceptions: list[Exception] = []
    if dt1.kind != dt2.kind:
        exceptions.append(TypeError(f"dtype kind mismatch ({dt1.kind} != {dt2.kind})"))
    if dt1.itemsize != dt2.itemsize:
        exceptions.append(
            TypeError(f"dtype itemsize mismatch ({dt1.itemsize} != {dt2.itemsize})")
        )
    return exceptions


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
