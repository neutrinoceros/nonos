import sys
from itertools import chain, permutations

import numpy as np
import pytest
from pytest import RaisesExc, RaisesGroup

from nonos._integrity_checks import (
    collect_dtype_exceptions,
    collect_shape_exceptions,
    compile_exceptions,
)

if sys.version_info < (3, 11):
    pass


@pytest.mark.parametrize(
    "off_by",
    list(
        chain(
            permutations((0, 0, 1)),
            permutations((0, 1, 1)),
            permutations((1, 1, 1)),
        ),
    ),
)
def test_collect_shape_exceptions(off_by):
    arr_shape = (3, 5, 7)
    correct_coordinates_shape = tuple(s + 1 for s in arr_shape)

    # sanity check before we move to the actual test
    assert not collect_shape_exceptions(arr_shape, correct_coordinates_shape)

    coordinates_shape = tuple(
        corr + off
        for corr, off in zip(
            correct_coordinates_shape,
            off_by,
            strict=True,
        )
    )

    res = collect_shape_exceptions(arr_shape, coordinates_shape)

    assert len(res) == sum(off_by)
    for exc in res:
        with pytest.raises(TypeError, match=r"^shape mismatch along axis"):
            raise exc


@pytest.mark.parametrize(
    "a, b, neq_attributes",
    [
        pytest.param("f4", "f4", [], id="exact-match"),
        pytest.param(">f4", "<f4", [], id="endianess-neq-ok"),
        pytest.param("i4", "f4", ["kind"], id="kind-neq"),
        pytest.param("f4", "f8", ["itemsize"], id="itemsize-neq"),
        pytest.param("i4", "f8", ["kind", "itemsize"], id="kind-and-itemsize-neq"),
    ],
)
def test_collect_dtype_exceptions(a, b, neq_attributes):
    for op in [list, reversed]:
        dt1, dt2 = [np.dtype(x) for x in op((a, b))]
        res = collect_dtype_exceptions(dt1, dt2)

        for exc, attr in zip(res, neq_attributes, strict=True):
            at1 = getattr(dt1, attr)
            at2 = getattr(dt2, attr)
            with pytest.raises(
                TypeError, match=rf"^dtype {attr} mismatch \({at1} != {at2}\)$"
            ):
                raise exc


@pytest.mark.parametrize(
    "groups, ctx",
    [
        pytest.param((), None, id="no-exc"),
        pytest.param((("g1", []),), None, id="single-empty-groups"),
        pytest.param((("g1", []), ("g2", [])), None, id="two-empty-groups"),
        pytest.param(
            (("g1", [TypeError("t1")]),),
            RaisesExc(TypeError, match="t1"),
            id="single-exc",
        ),
        pytest.param(
            (("g1", [TypeError("t1"), RuntimeError("r1")]), ("g2", [ValueError("v2")])),
            RaisesGroup(
                RaisesGroup(
                    RaisesExc(TypeError, match="t1"),
                    RaisesExc(RuntimeError, match="r1"),
                    match="g1",
                ),
                RaisesExc(ValueError, match="v2"),
                match="top",
            ),
            id="nested-groups",
        ),
    ],
)
def test_compile_exceptions(groups, ctx):
    res = compile_exceptions("top", *groups)

    if ctx is None:
        assert res is None
        return

    with ctx:
        raise res
