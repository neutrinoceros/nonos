import os
from pathlib import Path

import pytest

from nonos.loaders import Loader, Recipe, loader_from, recipe_from


class TestLoader:
    @pytest.fixture(params=Loader.__slots__, ids=lambda s: s.removesuffix("_"))
    def loader_slot(self, request):
        return request.param

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Loader(
                parameter_file=tmp_path / "this_file_does_not_exist",
                binary_reader=None,
                planet_reader=None,
                ini_reader=None,
                dtype=None,
            )

    def test_read_only_interface(self, tmp_path, loader_slot):
        # mock a minimal loader, even if not type-check compliant
        parameter_file = tmp_path / "fake.ini"
        parameter_file.touch()
        loader = Loader(
            parameter_file=parameter_file,
            binary_reader=None,
            planet_reader=None,
            ini_reader=None,
            dtype=None,
        )

        public_name = loader_slot.removeprefix("_")
        assert hasattr(loader, public_name)
        with pytest.raises(AttributeError):
            setattr(loader, public_name, None)


class TestGetRecipe:
    def test_no_args(self):
        with pytest.raises(TypeError):
            recipe_from()

    def test_only_position_arg(self):
        with pytest.raises(TypeError):
            recipe_from("idefix_vtk")

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param("idefix_vtk", Recipe.IDEFIX_VTK, id="idefix_vtk"),
            pytest.param("pluto_vtk", Recipe.PLUTO_VTK, id="pluto_vtk"),
            pytest.param("fargo3d", Recipe.FARGO3D, id="fargo3D"),
            pytest.param("fargo_adsg", Recipe.FARGO_ADSG, id="fargo_adsg"),
            pytest.param("idefix", Recipe.IDEFIX_VTK, id="(backward-compat) idefix"),
            pytest.param("pluto", Recipe.PLUTO_VTK, id="(backward-compat) pluto"),
        ],
    )
    def test_recipe_from_code(self, in_, expected):
        assert recipe_from(code=in_) is expected

    @pytest.mark.parametrize(
        "in_",
        [
            # using '-' instead of '_'
            "idefix-vtk",
            "pluto-vtk",
            "fargo-adsg",
            # empty string
            "",
            # and something random
            "bleurg",
        ],
    )
    def test_invalid_code(self, in_):
        with pytest.raises(ValueError):
            recipe_from(code=in_)

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param(
                Path("idefix_planet3d", "idefix.ini"),
                Recipe.IDEFIX_VTK,
                id="idefix_vtk",
            ),
            pytest.param(
                Path("pluto_spherical", "pluto.ini"),
                Recipe.PLUTO_VTK,
                id="pluto_vtk",
            ),
            pytest.param(
                Path("fargo3d_planet2d", "variables.par"),
                Recipe.FARGO3D,
                id="fargo3d",
            ),
            pytest.param(
                Path("fargo_adsg_planet", "planetpendragon_200k.par"),
                Recipe.FARGO_ADSG,
                id="fargo_adsg",
            ),
        ],
    )
    def test_valid_parameter_file(self, test_data_dir, in_, expected):
        assert recipe_from(parameter_file=test_data_dir / in_) is expected

    def test_ambiguous_parameter_file(self, tmp_path):
        fake_ini = tmp_path / "fake.ini"
        fake_ini.write_text("")
        with pytest.raises(
            ValueError,
            match=r"^Could not determine data format from",
        ):
            recipe_from(
                parameter_file=fake_ini,
            )

    def test_ambiguous_parameter_file_with_directory(self, tmp_path):
        fake_ini = tmp_path / "fake.ini"
        fake_ini.write_text("")
        with pytest.raises(
            ValueError,
            match=r"^Could not determine data format from parameter_file",
        ):
            recipe_from(parameter_file=fake_ini, directory=tmp_path)

    @pytest.mark.parametrize(
        "in_, expected",
        [
            pytest.param(
                "idefix_planet3d",
                Recipe.IDEFIX_VTK,
                id="idefix_vtk",
            ),
            pytest.param(
                "pluto_spherical",
                Recipe.PLUTO_VTK,
                id="pluto_vtk",
            ),
            pytest.param(
                "fargo3d_planet2d",
                Recipe.FARGO3D,
                id="fargo3d",
            ),
        ],
    )
    def test_directory(self, test_data_dir, in_, expected):
        assert recipe_from(directory=test_data_dir / in_) is expected

    def test_ambiguous_directory(self, tmp_path):
        tmp_path.joinpath("idefix.ini").touch()
        tmp_path.joinpath("pluto.ini").touch()
        with pytest.raises(
            RuntimeError,
            match=r"^Found multiple parameter files",
        ):
            recipe_from(directory=tmp_path)


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
            "fargo_adsg",
            id="fargo_adsg",
        ),
    ],
)
class TestLoaderFrom:
    def test_loaders_from_user_inputs(self, test_data_dir, parameter_file, code):
        parameter_file = test_data_dir.joinpath(*parameter_file)
        directory = parameter_file.parent
        loader0 = loader_from(
            code=code,
            parameter_file=parameter_file,
            directory=directory,
        )
        loader1 = loader_from(
            parameter_file=parameter_file,
            directory=directory,
        )
        assert loader1 == loader0

        loader2 = loader_from(
            parameter_file=parameter_file.name,
            directory=directory,
        )
        assert loader2 == loader0

        loader3 = loader_from(directory=directory)
        assert loader3 == loader0

        loader4 = loader_from(parameter_file=parameter_file)
        assert loader4 == loader0

        loader5 = loader_from(
            code=code,
            parameter_file=parameter_file,
        )
        assert loader5 == loader0

        loader6 = loader_from(
            code=code,
            directory=directory,
        )
        assert loader6 == loader0

    def test_loader_from_code_alone_error(
        self,
        parameter_file,  # noqa: ARG002,
        code,
    ):
        with pytest.raises(TypeError):
            loader_from(code=code)

    def test_loader_from_code_alone_with_chdir_error(
        self, test_data_dir, parameter_file, code
    ):
        os.chdir(test_data_dir.joinpath(*parameter_file).parent)
        with pytest.raises(TypeError):
            loader_from(code=code)

    def test_loader_from_inconsistent_inputs_error(
        self,
        test_data_dir,
        parameter_file,
        code,  # noqa: ARG002,
    ):
        parameter_file = test_data_dir.joinpath(*parameter_file).name
        with pytest.raises(
            ValueError, match=r"Received apparently inconsistent inputs"
        ):
            loader_from(parameter_file=parameter_file)
