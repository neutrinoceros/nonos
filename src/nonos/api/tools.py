__all__ = [
    "bracketing_values",
    "closest_index",
    "closest_value",
    "find_around",  # deprecated
    "find_nearest",  # deprecated
]
import sys
from dataclasses import dataclass

import numpy as np

from nonos._types import F, FArray1D

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


def closest_index(arr: FArray1D[F], v: float, /) -> int:
    """Find the index of the value in arr closest to v

    .. versionadded: 0.20.0
    """
    return int((np.abs(arr - v)).argmin())


def closest_value(arr: FArray1D[F], v: float, /) -> float:
    """
    Find the value in arr closest to v

    .. versionadded: 0.20.0
    """
    return float(arr[closest_index(arr, v)])


@dataclass(slots=True, frozen=True, kw_only=True)
class Interval:
    lo: float
    hi: float

    @property
    def span(self) -> float:
        return self.hi - self.lo

    def as_array(self, *, dtype: np.dtype[F]) -> FArray1D[F]:
        return np.array((self.lo, self.hi), dtype=dtype)


def bracketing_values(arr: FArray1D[F], v: float, /) -> Interval:
    """
    Find the two values in arr, closest to v, sorted from low to high.

    Return
    ------
    Interval(lo, hi):
        attributes: lo, hi
        property: span
        method: as_array

    .. versionadded: 0.20.0
    """
    v1 = closest_value(arr, v)
    ma = np.ma.masked_equal(arr, v1, copy=False)
    if np.all(ma.mask):
        return Interval(lo=v1, hi=v1)
    else:
        v2 = closest_value(ma, v)
        return Interval(
            lo=min(v1, v2),
            hi=max(v1, v2),
        )


@deprecated(
    "nonos.api.tools.find_nearest is deprecated since v0.20.0 "
    "and may be removed in a future version. "
    "Use nonos.api.tools.closest_index instead."
)
def find_nearest(array: FArray1D[F], value: float) -> int:  # pragma: no cover
    return closest_index(array, value)


@deprecated(
    "nonos.api.tools.find_around is deprecated since v0.20.0 "
    "and may be removed in a future version. "
    "Use nonos.api.tools.bracketing_values instead."
)
def find_around(array: FArray1D[F], value: float) -> FArray1D[F]:  # pragma: no cover
    return np.array(bracketing_values(array, value))
