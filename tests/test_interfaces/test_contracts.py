"""
This module contains low-level tests for the internal details of
nonos._reader and nonos.loaders.
They are meant to ensure that the constraints ingrained
in the original design continue to hold in the future, and provide
immediate feedback if a refactor breaks any of the initial promises,
in areas that cannot be easily checked by a type checker.

These tests are *not* sacred, and may need to be adjusted in case of
a new change in design.
"""

import inspect
from enum import Enum
from types import ModuleType
from typing import Protocol

import pytest

import nonos._readers as readers
from nonos import _types


def get_classes_from(module: ModuleType) -> list[type]:
    retv: list[type] = []
    for objname in module.__all__:
        obj = module.__dict__[objname]
        if inspect.isclass(obj):
            if issubclass(obj, Protocol | Enum):  # type: ignore [arg-type]
                continue
            retv.append(obj)
    return retv


_reader_classes: list[type] = []
for module in [
    readers.ini,
    readers.planet,
    readers.binary,
]:
    _reader_classes.extend(get_classes_from(module))

_all_classes: list[type] = _reader_classes.copy()
_all_classes.extend(get_classes_from(_types))


@pytest.fixture(params=_all_classes, ids=lambda cls: f"{cls.__module__}.{cls.__name__}")
def interface_class(request):
    return request.param


def test_abstract_final_pattern(interface_class):
    # check that all interface classes are exactly one of
    # - @final
    # - Protocol

    cls = interface_class
    isfinal = getattr(cls, "__final__", False)
    isprotocol = issubclass(cls, Protocol)
    assert isfinal ^ isprotocol
