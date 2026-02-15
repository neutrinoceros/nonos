import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nonos._readers.binary import NPYReader, VTKReader
from nonos._types import BinData


@pytest.mark.parametrize("dtype", [np.dtype("float32"), np.dtype("float64")])
def test_bindata(dtype):
    bd = BinData.default_init(dtype=dtype)
    with pytest.raises(TypeError):
        bd.finalize()


class TestVTKReader:
    # fmt: off
    @pytest.mark.parametrize(
        "file,expected_fields",
        [
            (
                ("micro_cubes", "micro_orszagtang.vtk"),
                [
                    "BX1","BX2", "BX3",
                    "PRS",
                    "RHO",
                    "TR0", "TR1",
                    "VX1", "VX2", "VX3"
                ],
            ),
            (
                ("micro_cubes", "micro_disk.vtk"),
                ["RHO", "VX1", "VX2", "VX3"],
            ),
        ],
    )
    # fmt: on
    def test_fields(self, test_data_dir, file, expected_fields):
        bd = VTKReader.read(test_data_dir.joinpath(*file))
        fields = sorted(bd.data.keys())
        assert fields == expected_fields

    def test_inconsistent_geometry(self, test_data_dir):
        # this simulation is in cartesian geometry
        file = test_data_dir / "micro_cubes" / "micro_orszagtang.vtk"
        with pytest.raises(ValueError):
            VTKReader.read(file, geometry="polar")

    def test_native_coords(self, test_data_dir):
        file = test_data_dir / "micro_cubes" / "micro_khi.vtk"
        bdL = VTKReader.read(file, cell="edges")
        bdC = VTKReader.read(file, cell="centers")
        assert len(bdC.x1) == len(bdL.x1) - 1
        assert len(bdC.x2) == len(bdL.x2) - 1
        assert len(bdC.x3) == len(bdL.x3) - 1

        assert_allclose(bdL.x1[:-1] + 0.5*np.diff(bdL.x1), bdC.x1)
        assert_allclose(bdL.x2[:-1] + 0.5*np.diff(bdL.x2), bdC.x2)
        assert_allclose(bdL.x3[:-1] + 0.5*np.diff(bdL.x3), bdC.x3)


class TestNPYReader:
    @pytest.fixture
    def initdir(self, tmp_path):
        # setup a file tree that mocks what you'd get after
        # saving some reductions
        header_dir = tmp_path / "header"
        header_dir.mkdir()
        fake_grid = {
            "geometry": "cartesian",
            "x": [0, 0.5, 1],
            "y": [0, 0.5, 1],
            "z": [0, 0.5, 1],
        }

        with open(header_dir / "header_azimuthal_average.json", "w") as fh:
            json.dump(fake_grid, fh)
        with open(header_dir / "header_vertical_at_midplane.json", "w") as fh:
            json.dump(fake_grid, fh)

        rng = np.random.default_rng(seed=0)

        rho_dir = tmp_path / "rho"
        rho_dir.mkdir()
        rho_file = rho_dir.joinpath("azimuthal_average_RHO.0001.npy")
        fake_rho = rng.normal(0, 1, size=8).resize(2, 2, 2)
        with open(rho_file, "wb") as fb:
            np.save(fb, fake_rho)

        rho_file_rr = rho_dir.joinpath("radial_at_r1.3_RHO.0001.npy")
        fake_rho_rr = rng.normal(0, 1, size=8).resize(2, 2, 2)
        with open(rho_file_rr, "wb") as fb:
            np.save(fb, fake_rho_rr)

        foo_dir = tmp_path / "foo"
        foo_dir.mkdir()
        foo_file = foo_dir.joinpath("__FOO.0456.npy")
        fake_foo = rng.normal(0, 1, size=8).resize(2, 2, 2)
        with open(foo_file, "wb") as fb:
            np.save(fb, fake_foo)

        vx1_dir = tmp_path / "vx1"
        vx1_dir.mkdir()
        vx1_file = vx1_dir.joinpath("_vertical_at_midplane_VX1.1200.npy")
        fake_vx1 = rng.normal(0, 1, size=8).resize(2, 2, 2)
        with open(vx1_file, "wb") as fb:
            np.save(fb, fake_vx1)

        # sprinkle some "traps" too (files and dirs that *shouldn't* match)
        tmp_path.joinpath("trap1.0001").mkdir()
        foo_dir.joinpath("trap2.json").touch()
        rho_dir.joinpath("trap3_RHO.0001.npy").mkdir()
        rho_file_2 = rho_dir.joinpath("azimuthal_average_RHO.0002.npy")
        rho_file_2.touch()

        rho_dir.joinpath("trap4.npy").touch()

        real_files = sorted([foo_file, rho_file, rho_file_rr, rho_file_2, vx1_file])
        return tmp_path, real_files

    def test_get_bin_files(self, initdir):
        tmp_path, expected_files = initdir

        subdirs = []
        for f in tmp_path.glob("*"):
            if f.is_dir():
                subdirs.append(f)

        for subdir in subdirs:
            results = NPYReader.get_bin_files(subdir)
            assert results == expected_files

    @pytest.mark.parametrize(
        "file_or_number, prefix, expected_output_number, expected_filename",
        [
            (
                "azimuthal_average_RHO.0001.npy",
                "azimuthal_average",
                1,
                ("rho", "azimuthal_average_RHO.0001.npy"),
            ),
            (1, "azimuthal_average", 1, ("rho", "azimuthal_average_RHO.0001.npy")),
            ("__FOO.0456.npy", "", 456, ("foo", "__FOO.0456.npy")),
            (456, "", 456, ("foo", "__FOO.0456.npy")),
        ],
    )
    def test_parse_output_number_and_filename(
        self,
        initdir,
        file_or_number,
        prefix,
        expected_output_number,
        expected_filename,
    ):
        tmp_path, _files = initdir
        directory = tmp_path.resolve()

        output_number, filename = NPYReader.parse_output_number_and_filename(
            file_or_number,
            directory=directory,
            prefix=prefix,
        )
        assert output_number == expected_output_number
        assert filename == directory.joinpath(*expected_filename)

    def test_parse_output_number_and_filename_invalid_file(self, initdir):
        tmp_path, _files = initdir
        directory = tmp_path.resolve()

        with pytest.raises(ValueError, match="Filename 'invalid' is not recognized"):
            NPYReader.parse_output_number_and_filename(
                "invalid",
                directory=directory,
                prefix="azimuthal_average",
            )

    def test_read(self, initdir):
        _tmp_path, files = initdir
        NPYReader.read(files[1])

    def test_read_invalid_file(self, initdir):
        tmp_path, _files = initdir
        with pytest.raises(ValueError, match="Filename 'invalid' is not recognized"):
            NPYReader.read(tmp_path / "invalid")
