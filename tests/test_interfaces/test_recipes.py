"""
Test basic functionalities of loaders composed according
to known recipes.
These tests are at an intermediate between contract tests
and integration tests.
"""

from contextlib import nullcontext
from types import GenericAlias
from typing import Any

import pytest

from nonos._types import BinData, IniData, PlanetData
from nonos.loaders import Loader, loader_from


def validate_dataclass_instance(instance, cls):
    if hasattr(cls, "_init_attrs"):
        # special case, initially designed for PlanetData
        attrs = cls._init_attrs
    else:
        # general case: __slots__ are exactly the fields
        # that are expected at initialization
        attrs = cls.__slots__
    for key in attrs:
        if isinstance(expected_type := cls.__annotations__[key], str | GenericAlias):
            # skip unresolved annotations and generics
            continue
        obj = getattr(instance, key)
        assert isinstance(obj, expected_type)


class CheckLoader:
    parameter_file: tuple[str, str]  # parent dir and filename
    code: str
    loader: Loader
    expected_n_bin_files: int
    expected_n_planet_files: int | None
    expected_data_keys: list[str]
    meta: dict[str, Any]

    @pytest.fixture
    def initloader(self, test_data_dir):
        loader = loader_from(
            code=self.code,
            parameter_file=test_data_dir.joinpath(*self.parameter_file),
        )
        directory = loader.parameter_file.parent
        return loader, directory

    def test_binary_files(self, initloader):
        loader, directory = initloader
        reader = loader.binary_reader
        files = reader.get_bin_files(directory)
        assert len(files) == self.expected_n_bin_files

    def test_load_binary_data(self, initloader):
        loader, directory = initloader
        if self.expected_n_bin_files == 0:  # pragma: no cover
            pytest.skip("no actual data in store")
        reader = loader.binary_reader

        files = reader.get_bin_files(directory)
        gd = loader.load_bin_data(files[0], **self.meta)
        validate_dataclass_instance(gd, BinData)
        assert sorted(gd.data.keys()) == sorted(self.expected_data_keys)

    def test_planet_files(self, initloader):
        loader, directory = initloader
        reader = loader.planet_reader

        if self.expected_n_planet_files is None:
            ctx = pytest.raises(NotImplementedError)
        else:
            ctx = nullcontext()

        with ctx:
            files = reader.get_planet_files(directory)
            assert len(files) == self.expected_n_planet_files

    def test_load_planet_data(self, tmp_path, initloader):
        loader, directory = initloader
        if self.expected_n_planet_files == 0:  # pragma: no cover
            pytest.skip("no actual data in store")
        reader = loader.planet_reader

        if self.expected_n_planet_files is None:
            file = tmp_path / "not_a_file"
            ctx = pytest.raises(NotImplementedError)
        else:
            files = reader.get_planet_files(directory)
            file = files[0]
            ctx = nullcontext()

        with ctx:
            pd = loader.load_planet_data(file)
            validate_dataclass_instance(pd, PlanetData)

    def test_load_ini_file(self, initloader):
        loader, _directory = initloader
        ini = loader.load_ini_file()
        validate_dataclass_instance(ini, IniData)


class TestIdefixLoader(CheckLoader):
    code = "idefix_vtk"
    parameter_file = ("idefix_planet3d", "idefix.ini")
    meta = {"geometry": "polar"}
    expected_n_bin_files = 2
    expected_n_planet_files = 1
    expected_data_keys = ["RHO", "VX1", "VX2", "VX3"]


class TestPlutoLoader(CheckLoader):
    code = "pluto_vtk"
    parameter_file = ("pluto_spherical", "pluto.ini")
    meta = {}
    expected_n_bin_files = 0
    expected_n_planet_files = None  # not implemented
    expected_data_keys = []


class TestFargo3DLoader(CheckLoader):
    code = "fargo3d"
    parameter_file = ("fargo3d_planet2d", "variables.par")
    meta = {}
    expected_n_bin_files = 2
    expected_n_planet_files = 1
    expected_data_keys = ["RHO"]


class TestFargoADSGLoader(CheckLoader):
    code = "fargo-adsg"
    parameter_file = ("fargo_adsg_planet", "planetpendragon_200k.par")
    meta = {}
    expected_n_bin_files = 1
    expected_n_planet_files = 1
    expected_data_keys = ["RHO"]
