from math import isnan

import inifix
import pytest

from nonos._readers.ini import Fargo3DReader, FargoADSGReader
from nonos._types import FrameType, IniReader

GenericFargo_to_FargoADSG_rosetta_stone = {
    "NINTERM": "Ninterm",
    "OMEGAFRAME": "OmegaFrame",
    "FRAME": "Frame",
}


def compile_to_FargoADSG(text: str) -> str:
    retv = text
    for k, v in GenericFargo_to_FargoADSG_rosetta_stone.items():
        retv = retv.replace(k, v)
    return retv


def compile_to_Fargo3D(text: str) -> str:
    return text


@pytest.mark.parametrize("reader", [Fargo3DReader, FargoADSGReader])
class TestFargoReaders:
    def compile(self, text: str, reader: IniReader) -> str:
        if reader is FargoADSGReader:
            return compile_to_FargoADSG(text)
        elif reader is Fargo3DReader:
            return compile_to_Fargo3D(text)
        else:
            raise RuntimeError

    @pytest.mark.parametrize(
        "template_lines, expected_frame, expected_rotational_rate",
        [
            pytest.param(
                [
                    "NINTERM 10",
                    "DT 0.2",
                    "FRAME F",
                    "OMEGAFRAME 3.0",
                ],
                FrameType.CONSTANT_ROTATION,
                3.0,
                id="explicit constant rotation",
            ),
            pytest.param(
                [
                    "NINTERM 10",
                    "DT 0.2",
                    "FRAME C",
                    "OMEGAFRAME 3.0",
                ],
                FrameType.PLANET_COROTATION,
                float("nan"),
                id="explicit planet corotation",
            ),
            pytest.param(
                [
                    "NINTERM 10",
                    "DT 0.2",
                    "FRAME F",
                ],
                FrameType.FIXED_FRAME,
                0.0,
                id="implicit rotation rate",
            ),
            pytest.param(
                [
                    "NINTERM 10",
                    "DT 0.2",
                ],
                FrameType.FIXED_FRAME,
                0.0,
                id="implicit frame and rotation rate",
            ),
        ],
    )
    def test_read_frame(
        self,
        tmp_path,
        reader,
        template_lines,
        expected_frame,
        expected_rotational_rate,
    ):
        body = self.compile("\n".join(template_lines), reader)
        inifile = tmp_path / "variables.par"
        inifile.write_text(body)

        expected_meta = inifix.loads(body)

        ini = reader.read(inifile)
        assert ini.frame is expected_frame
        if isnan(expected_rotational_rate):
            assert isnan(ini.rotational_rate)
        else:
            assert ini.rotational_rate == expected_rotational_rate
        assert ini.output_time_interval == 2.0
        assert ini.meta == expected_meta

    def test_unknown_frame(self, tmp_path, reader):
        body = self.compile(
            "\n".join(
                [
                    "NINTERM 10",
                    "DT 0.2",
                    "FRAME TEST",
                    "OMEGAFRAME 3.0",
                ]
            ),
            reader=reader,
        )

        inifile = tmp_path / "variables.par"
        inifile.write_text(body)

        with pytest.raises(NotImplementedError):
            reader.read(inifile)
