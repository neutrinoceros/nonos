import re

import numpy as np
import pytest

from nonos_cli.parsing import (
    parse_image_format,
    parse_range,
    parse_snapshot_number_range,
    range_converter,
    userval_or_default,
)


@pytest.mark.parametrize(
    "received, expected",
    [
        (("val", "default"), "val"),
        (("unset", "default"), "default"),
        ((None, "default"), "default"),
        (("unset", None), None),
        ((None, None), None),
    ],
)
def test_userval_or_default(received, expected):
    userval, default = received
    assert userval_or_default(userval, default=default) == expected


@pytest.mark.parametrize(
    "received, expected",
    [
        (1, [1]),
        (0, [0]),
        ([1], [1]),
        ([0, 1], [0, 1]),
        ([0, 2], [0, 1, 2]),
        ([0, 5], [0, 1, 2, 3, 4, 5]),
        ([10, 20, 5], [10, 15, 20]),
        ([0, 299, 50], [0, 50, 100, 150, 200, 250]),
    ],
)
def test_range_outputs(received, expected):
    assert parse_snapshot_number_range(received) == expected


def test_from_maxval():
    maxval = 1.0
    assert parse_snapshot_number_range(None, maxval=maxval) == [maxval]


def test_unparsable_data():
    with pytest.raises(
        ValueError, match="Can't parse a range from unset values without a max."
    ):
        parse_snapshot_number_range("unset")


def test_invalid_request():
    with pytest.raises(
        ValueError, match="No output beyond 5 is available, but 10 was requested."
    ):
        parse_snapshot_number_range([1, 10], maxval=5)


def test_invalid_nargs():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Can't parse a range from sequence [1, 2, 3, 4] with more than 3 values."
        ),
    ):
        parse_snapshot_number_range([1, 2, 3, 4])


@pytest.mark.parametrize(
    "dim, extent",
    [
        (2, ("0.4", "8", "0", "-0.2", "0.2")),
        (1, (0.4, 8, 0.2)),
    ],
)
def test_invalid_nargs_parse_range(dim, extent):
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Received sequence `extent` with incorrect size {len(extent)}. Expected exactly {2*dim=} values."
        ),
    ):
        parse_range(extent, dim)


@pytest.mark.parametrize(
    "received",
    [
        [1, 0],
        [1, 0, 1],
        [1, 0, 2],
    ],
)
def test_invalid_range(received):
    with pytest.raises(ValueError, match="Can't parse a range with max < min."):
        parse_snapshot_number_range(received)


@pytest.mark.parametrize(
    "abscissa, ordinate, dim, expected",
    [
        (
            np.linspace(0.2, 10, 100),
            np.linspace(-np.pi, np.pi, 100),
            2,
            (0.2, 10.0, -np.pi, np.pi),
        ),
        (np.linspace(0.2, 10, 100), np.zeros(2), 1, (0.2, 10.0)),
    ],
)
def test_parse_range(abscissa, ordinate, dim, expected):
    extent1 = parse_range("unset", dim=dim)
    assert range_converter(extent1, abscissa=abscissa, ordinate=ordinate) == expected
    extent2 = parse_range(("0.5", "5", "-0.2", "0.2"), dim=2)
    assert range_converter(
        extent2,
        abscissa=np.linspace(0.2, 10, 100),
        ordinate=np.linspace(-0.4, 0.4, 100),
    ) == (0.5, 5, -0.2, 0.2)
    extent3 = parse_range(("0.4", "9.5"), dim=1)
    assert range_converter(
        extent3, abscissa=np.linspace(0.2, 10, 100), ordinate=np.zeros(2)
    ) == (0.4, 9.5)
    extent4 = parse_range(("0.5", "x", "-0.2", "x"), dim=2)
    assert extent4 == (0.5, None, -0.2, None)
    assert range_converter(
        extent4,
        abscissa=np.linspace(0.2, 10, 100),
        ordinate=np.linspace(-0.4, 0.4, 100),
    ) == (0.5, 10.0, -0.2, 0.4)


def test_range_converter_error():
    with pytest.raises(
        TypeError,
        match="Expected extent to be of length 2 or 4,",
    ):
        range_converter(extent=[], abscissa=None, ordinate=None)


@pytest.mark.parametrize(
    "received, expected",
    [
        (".png", "png"),
        (".pdf", "pdf"),
        ("png", "png"),
        ("pdf", "pdf"),
    ],
)
def test_image_format(received, expected):
    assert parse_image_format(received) == expected


@pytest.mark.parametrize("received", ["unset", None])
def test_image_format_default(received):
    result = parse_image_format(received)
    assert isinstance(result, str)
    assert re.fullmatch(r"\w+", result)


def test_invalid_image_format():
    fake_ext = ".pnd"
    with pytest.raises(
        ValueError,
        match=f"^(Received unknown file format '{fake_ext}'. Available formatted are)",
    ):
        parse_image_format(fake_ext)
