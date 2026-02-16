import sys

import numpy as np

from nonos._types import F, FArray1D

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


@deprecated(
    "nonos.api.tools.find_nearest is deprecated since v0.20.0 "
    "and may be removed in a future version. There are no "
    "plans for a replacement in the public API. "
)
def find_nearest(array: FArray1D[F], value: float) -> int:
    from nonos._approx import closest_index

    return closest_index(array, value)


@deprecated(
    "nonos.api.tools.find_around is deprecated since v0.20.0 "
    "and may be removed in a future version. There are no "
    "plans for a replacement in the public API. "
)
def find_around(array: FArray1D[F], value: float) -> FArray1D[F]:
    from nonos._approx import bracketing_values

    return np.array(bracketing_values(array, value))
