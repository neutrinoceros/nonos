from itertools import combinations

import pytest

from nonos.api._angle_parsing import _parse_planet_file, _parse_rotation_angle


class TestParsePlanetFile:
    def test_from_filename(self):
        input_ = "test"
        assert _parse_planet_file(planet_file=input_) == input_

    def test_from_number(self):
        input_ = 456
        assert _parse_planet_file(planet_number=input_) == f"planet{input_}.dat"

    def test_both_args(self):
        with pytest.raises(TypeError):
            _parse_planet_file(
                planet_file="test",
                planet_number=1,
            )

    def test_no_args(self):
        assert _parse_planet_file() == "planet0.dat"


class MockPlanetAzimuthFinder:
    def find_phip(
        self,
        planet_number: int | None = None,  # noqa: ARG002
        *,
        planet_file: str | None = None,  # noqa: ARG002
    ) -> float:
        return 0.0


class TestParseRotationAngle:
    example_inputs = {
        "rotate_by": 1.0,
        "rotate_with": "planet0.dat",
        "planet_number_argument": ("test", 0),
    }
    default_kwargs = {
        "rotate_by": None,
        "rotate_with": None,
        "planet_number_argument": ("test", None),
        "planet_azimuth_finder": MockPlanetAzimuthFinder(),
        "stacklevel": 2,
    }

    @pytest.mark.parametrize("kwargs", combinations(example_inputs.items(), 2))
    def test_two_inputs(self, kwargs):
        conf = {**self.default_kwargs, **dict(kwargs)}
        with pytest.raises(TypeError, match="Can only process one argument"):
            _parse_rotation_angle(**conf)

    def test_all_inputs(self):
        conf = {**self.default_kwargs, **self.example_inputs}
        with pytest.raises(TypeError, match="Can only process one argument"):
            _parse_rotation_angle(**conf)

    def test_from_rotate_with(self):
        conf = {
            **self.default_kwargs,
            "rotate_with": self.example_inputs["rotate_with"],
        }
        result = _parse_rotation_angle(**conf)
        assert result == 0.0

    def test_from_planet_number(self):
        conf = {
            **self.default_kwargs,
            "planet_number_argument": self.example_inputs["planet_number_argument"],
        }
        with pytest.deprecated_call():
            result = _parse_rotation_angle(**conf)
        assert result == 0.0
