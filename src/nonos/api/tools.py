import numpy as np

from nonos._types import D, F, FArray


def find_nearest(array: FArray[D, F], value: float | F) -> int:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


def find_around(array: FArray[D, F], value: float | F) -> FArray[D, F]:
    array = np.asarray(array)
    idx_1 = (np.abs(array - value)).argmin()
    larray = list(array)
    larray.remove(larray[idx_1])
    arraym = np.asarray(larray)
    idx_2 = (np.abs(arraym - value)).argmin()
    return np.asarray([array[idx_1], arraym[idx_2]])
