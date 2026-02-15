__all__ = [
    "bracket_values",
    "closest_index",
    "closest_value",
]
import numpy as np

from nonos._types import F, FArray1D


def closest_index(arr: FArray1D[F], v: float, /) -> int:
    """Find the index of the value in arr closest to v"""
    return int((np.abs(arr - v)).argmin())


def closest_value(arr: FArray1D[F], v: float, /) -> float:
    """Find the value in arr closest to v"""
    return arr[closest_index(arr, v)]


def bracket_values(arr: FArray1D[F], v: float, /) -> tuple[float, float]:
    """Find the first two values in arr closest to v"""
    lo = closest_value(arr, v)
    hi = closest_value(np.ma.masked_less_equal(arr, lo, copy=False), v)
    return (float(lo), float(hi))
