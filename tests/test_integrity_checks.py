import re
import sys
from itertools import chain, permutations
from uuid import uuid4

import numpy as np
import pytest
from pytest import RaisesExc, RaisesGroup

from nonos._geometry import Coordinates, Geometry
from nonos._integrity_checks import (
    check_field_operands,
    collect_dtype_exceptions,
    collect_shape_exceptions,
    compile_exceptions,
)
from nonos.api import Field

if sys.version_info >= (3, 13):
    from copy import replace
else:
    from dataclasses import replace


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
                TypeError,
                match="^"
                + re.escape(f"mixed dtype {attr}s: {sorted((at1, at2))}")
                + "$",
            ):
                raise exc


@pytest.fixture
def stub_coordinates():
    return Coordinates(
        geometry=Geometry.CARTESIAN,
        x1=np.arange(3, dtype="f8"),
        x2=np.arange(4, dtype="f8"),
        x3=np.arange(5, dtype="f8"),
    )


def test_check_field_operands(stub_coordinates, subtests):
    data1 = np.arange(24, dtype="f8").reshape(2, 3, 4)
    data2 = np.ones_like(data1)
    c = stub_coordinates
    f1 = Field(name=uuid4(), data=data1, coordinates=c)
    f2 = Field(name=uuid4(), data=data2, coordinates=c)

    # first check that basic inputs are compatible
    assert check_field_operands(f1, f2) is None

    for cvar, subid in [
        (replace(c, geometry=Geometry.SPHERICAL), "neq-geometry"),
        (replace(c, x1=2 * c.x1), "neq-x1"),
    ]:
        exc = check_field_operands(f1, f2.replace(coordinates=cvar))
        assert exc is not None
        with (
            subtests.test(subid),
            pytest.raises(
                TypeError,
                match=r"^operands have incompatible coordinates$",
            ),
        ):
            raise exc

    exc = check_field_operands(f1, f2.astype("f4"))
    assert exc is not None
    with (
        subtests.test("neq-dtype"),
        pytest.raises(
            TypeError,
            match=r"^operands have incompatible dtypes$",
        ),
    ):
        raise exc

    f3 = Field(
        name=uuid4(),
        data=data2.reshape(4, 3, 2),
        coordinates=replace(c, x1=c.x3, x3=c.x1),
    )
    exc = check_field_operands(f1, f3)
    assert exc is not None
    with (
        subtests.test("neq-shape"),
        pytest.raises(
            TypeError,
            match=r"^operands have incompatible shapes$",
        ),
    ):
        raise exc

    exc = check_field_operands(f1, f3.astype("f4"))
    assert exc is not None
    with (
        subtests.test("neq-shape-neq-dtype"),
        RaisesGroup(
            RaisesExc(TypeError, match=r"^operands have incompatible dtypes$"),
            RaisesExc(TypeError, match=r"^operands have incompatible shapes$"),
            match="multiple issues with operands",
        ),
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
