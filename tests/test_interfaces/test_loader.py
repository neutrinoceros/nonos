import os
from pathlib import Path

import pytest

from nonos.loaders import BUILTIN_RECIPES, Loader, Readers


class TestLoader:
    @pytest.fixture(params=Loader.__slots__, ids=lambda s: s.removesuffix("_"))
    def loader_slot(self, request):
        return request.param

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Loader(
                components=None, parameter_file=tmp_path / "this_file_does_not_exist"
            )

    def test_read_only_interface(self, tmp_path, loader_slot):
        # mock a minimal loader, even if not type-check compliant
        parameter_file = tmp_path / "fake.ini"
        parameter_file.touch()
        loader = Loader(components=None, parameter_file=parameter_file)

        public_name = loader_slot.removeprefix("_")
        assert hasattr(loader, public_name)
        with pytest.raises(AttributeError):
            setattr(loader, public_name, None)


class TestGetRecipe:
    def test_no_args(self):
        with pytest.raises(TypeError):
            Readers.resolve()

    def test_only_position_arg(self):
        with pytest.raises(TypeError):
            Readers.resolve("idefix_vtk")

    @pytest.mark.parametrize(
        "aliases, expected",
        [
            pytest.param(
                ["idefix", "idefix_vtk", "idefix-vtk"],
                BUILTIN_RECIPES["idefix-vtk"],
                id="idefix",
            ),
            pytest.param(
                ["pluto", "pluto_vtk", "pluto-vtk"],
                BUILTIN_RECIPES["pluto-vtk"],
                id="pluto",
            ),
            pytest.param(["fargo3d"], BUILTIN_RECIPES["fargo3d"], id="fargo3D"),
            pytest.param(
                ["fargo_adsg", "fargo-adsg"],
                BUILTIN_RECIPES["fargo-adsg"],
                id="fargo-adsg",
            ),
        ],
    )
    def test_recipe_from_code(self, subtests, aliases, expected):
        for alias in aliases:
            with subtests.test(alias):
                assert Readers.resolve(code=alias) == expected

    @pytest.mark.parametrize("in_", ["", "bleurg"])
    def test_invalid_code(self, in_):
        with pytest.raises(ValueError):
            Readers.resolve(code=in_)

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param(
                Path("idefix_planet3d", "idefix.ini"),
                BUILTIN_RECIPES["idefix-vtk"],
                id="idefix_vtk",
            ),
            pytest.param(
                Path("pluto_spherical", "pluto.ini"),
                BUILTIN_RECIPES["pluto-vtk"],
                id="pluto_vtk",
            ),
            pytest.param(
                Path("fargo3d_planet2d", "variables.par"),
                BUILTIN_RECIPES["fargo3d"],
                id="fargo3d",
            ),
            pytest.param(
                Path("fargo_adsg_planet", "planetpendragon_200k.par"),
                BUILTIN_RECIPES["fargo-adsg"],
                id="fargo-adsg",
            ),
        ],
    )
    def test_valid_parameter_file(self, test_data_dir, in_, expected):
        assert Readers.resolve(parameter_file=test_data_dir / in_) == expected

    def test_ambiguous_parameter_file(self, tmp_path):
        fake_ini = tmp_path / "fake.ini"
        fake_ini.write_text("")
        with pytest.raises(
            ValueError,
            match=r"^Could not determine data format from",
        ):
            Readers.resolve(
                parameter_file=fake_ini,
            )

    def test_ambiguous_parameter_file_with_directory(self, tmp_path):
        fake_ini = tmp_path / "fake.ini"
        fake_ini.write_text("")
        with pytest.raises(
            ValueError,
            match=r"^Could not determine data format from parameter_file",
        ):
            Readers.resolve(parameter_file=fake_ini, directory=tmp_path)

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param(
                "idefix_planet3d",
                BUILTIN_RECIPES["idefix-vtk"],
                id="idefix_vtk",
            ),
            pytest.param(
                "pluto_spherical",
                BUILTIN_RECIPES["pluto-vtk"],
                id="pluto_vtk",
            ),
            pytest.param(
                "fargo3d_planet2d",
                BUILTIN_RECIPES["fargo3d"],
                id="fargo3d",
            ),
        ],
    )
    def test_directory(self, test_data_dir, in_, expected):
        assert Readers.resolve(directory=test_data_dir / in_) is expected

    def test_ambiguous_directory(self, tmp_path):
        tmp_path.joinpath("idefix.ini").touch()
        tmp_path.joinpath("pluto.ini").touch()
        with pytest.raises(
            RuntimeError,
            match=r"^Found multiple parameter files",
        ):
            Readers.resolve(directory=tmp_path)


@pytest.mark.parametrize(
    "parameter_file, code",
    [
        pytest.param(
            ("idefix_planet3d", "idefix.ini"),
            "idefix_vtk",
            id="idefix_vtk",
        ),
        pytest.param(
            ("pluto_spherical", "pluto.ini"),
            "pluto_vtk",
            id="pluto_vtk",
        ),
        pytest.param(
            ("fargo3d_planet2d", "variables.par"),
            "fargo3d",
            id="fargo3d",
        ),
        pytest.param(
            ("fargo_adsg_planet", "planetpendragon_200k.par"),
            "fargo-adsg",
            id="fargo-adsg",
        ),
    ],
)
class TestLoaderFrom:
    def test_loaders_from_user_inputs(self, test_data_dir, parameter_file, code):
        parameter_file = test_data_dir.joinpath(*parameter_file)
        directory = parameter_file.parent
        loader0 = Loader.resolve(
            code=code,
            parameter_file=parameter_file,
            directory=directory,
        )
        loader1 = Loader.resolve(
            parameter_file=parameter_file,
            directory=directory,
        )
        assert loader1 == loader0

        loader2 = Loader.resolve(
            parameter_file=parameter_file.name,
            directory=directory,
        )
        assert loader2 == loader0

        loader3 = Loader.resolve(directory=directory)
        assert loader3 == loader0

        loader4 = Loader.resolve(parameter_file=parameter_file)
        assert loader4 == loader0

        loader5 = Loader.resolve(
            code=code,
            parameter_file=parameter_file,
        )
        assert loader5 == loader0

        loader6 = Loader.resolve(
            code=code,
            directory=directory,
        )
        assert loader6 == loader0

    def test_Loader_code_alone_error(
        self,
        parameter_file,  # noqa: ARG002,
        code,
    ):
        with pytest.raises(TypeError):
            Loader.resolve(code=code)

    def test_Loader_code_alone_with_chdir_error(
        self, test_data_dir, parameter_file, code
    ):
        os.chdir(test_data_dir.joinpath(*parameter_file).parent)
        with pytest.raises(TypeError):
            Loader.resolve(code=code)

    def test_Loader_inconsistent_inputs_error(
        self,
        test_data_dir,
        parameter_file,
        code,  # noqa: ARG002,
    ):
        parameter_file = test_data_dir.joinpath(*parameter_file).name
        with pytest.raises(
            ValueError, match=r"Received apparently inconsistent inputs"
        ):
            Loader.resolve(parameter_file=parameter_file)
