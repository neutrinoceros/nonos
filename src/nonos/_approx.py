__all__ = [
    "bracketing_values",
    "closest_index",
    "closest_value",
]
from dataclasses import dataclass

import numpy as np

from nonos._types import F, FArray1D


def closest_index(arr: FArray1D[F], v: float, /) -> int:
    """Find the index of the value in arr closest to v"""
    return int((np.abs(arr - v)).argmin())


def closest_value(arr: FArray1D[F], v: float, /) -> float:
    """Find the value in arr closest to v"""
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
    """Find the first two values in arr closest to v, sorted from low to high"""
    v1 = closest_value(arr, v)
    ma = np.ma.masked_equal(arr, v1, copy=False)
    if np.all(ma.mask):
        return Interval(lo=v1, hi=v1)
    else:
        v2 = closest_value(ma, v)
        return Interval(
            lo=float(min(v1, v2)),
            hi=float(max(v1, v2)),
        )
