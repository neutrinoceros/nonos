__all__ = [
    "Fargo3DReader",
    "FargoADSGReader",
    "IdefixVTKReader",
    "PlutoVTKReader",
]
from pathlib import Path
from typing import final

import inifix

from nonos._types import FrameType, IniData, PathT


@final
class IdefixVTKReader:
    @staticmethod
    def read(file: PathT, /) -> IniData:
        class IdefixIniOutput:
            def __init__(self, *, vtk: list, **_kwargs) -> None:
                self.vtk = float(vtk[0])

        class IdefixIniHydro:
            def __init__(self, rotation: list = [0.0]) -> None:  # noqa: B006
                self.rotation = float(rotation[0])
                if self.rotation != 0.0:
                    self.frame = FrameType.CONSTANT_ROTATION
                else:
                    self.frame = FrameType.FIXED_FRAME

        class IdefixIni:
            def __init__(self, *, Hydro, Output, **_kwargs) -> None:
                self.hydro = IdefixIniHydro(rotation=Hydro.get("rotation", [0.0]))
                self.output = IdefixIniOutput(vtk=Output["vtk"])

        meta = inifix.load(file, sections="require", parse_scalars_as_lists=True)
        ini = IdefixIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=ini.hydro.frame,
            rotational_rate=ini.hydro.rotation,
            output_time_interval=ini.output.vtk,
            meta=inifix.load(file, sections="require", skip_validation=True),
        )


@final
class PlutoVTKReader:
    @staticmethod
    def read(file: PathT, /) -> IniData:
        class PlutoIniOutput:
            def __init__(self, *, vtk: list, **_kwargs) -> None:
                self.vtk = float(vtk[0])

        class PlutoIni:
            def __init__(self, **kwargs) -> None:
                self.output = PlutoIniOutput(**kwargs["Static Grid Output"])

        meta = inifix.load(file, sections="require", parse_scalars_as_lists=True)
        ini = PlutoIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=FrameType.FIXED_FRAME,
            rotational_rate=0.0,
            output_time_interval=ini.output.vtk,
            meta=inifix.load(file, sections="require", skip_validation=True),
        )


@final
class Fargo3DReader:
    @staticmethod
    def read(file: PathT, /) -> IniData:
        class Fargo3DIni:
            def __init__(
                self,
                *,
                NINTERM: list,
                DT: list,
                FRAME: list = ["F"],  # noqa: B006
                OMEGAFRAME: list = [0.0],  # noqa: B006
                **_kwargs,
            ) -> None:
                self.NINTERM = int(NINTERM[0])
                self.DT = float(DT[0])
                self.FRAME: FrameType
                self.OMEGAFRAME: float

                FRAME_unwrapped = str(FRAME[0])
                if FRAME_unwrapped == "F":
                    self.OMEGAFRAME = float(OMEGAFRAME[0])
                    if self.OMEGAFRAME == 0.0:
                        self.FRAME = FrameType.FIXED_FRAME
                    else:
                        self.FRAME = FrameType.CONSTANT_ROTATION
                elif FRAME_unwrapped == "C":
                    self.FRAME = FrameType.PLANET_COROTATION
                    self.OMEGAFRAME = float("nan")
                else:
                    raise NotImplementedError

        meta = inifix.load(file, sections="forbid", parse_scalars_as_lists=True)
        ini = Fargo3DIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=ini.FRAME,
            rotational_rate=ini.OMEGAFRAME,
            output_time_interval=ini.NINTERM * ini.DT,
            meta=inifix.load(file, sections="forbid", skip_validation=True),
        )


@final
class FargoADSGReader:
    @staticmethod
    def read(file: PathT, /) -> IniData:
        class FargoADSGIni:
            def __init__(
                self,
                *,
                Ninterm: list,
                DT: list,
                Frame: list = ["F"],  # noqa: B006
                OmegaFrame: list = [0.0],  # noqa: B006
                **_kwargs,
            ) -> None:
                self.NINTERM = int(Ninterm[0])
                self.DT = float(DT[0])
                self.FRAME: FrameType
                self.OMEGAFRAME: float

                Frame_unwrapped = str(Frame[0])

                if Frame_unwrapped == "F":
                    self.OMEGAFRAME = float(OmegaFrame[0])
                    if self.OMEGAFRAME == 0.0:
                        self.FRAME = FrameType.FIXED_FRAME
                    else:
                        self.FRAME = FrameType.CONSTANT_ROTATION
                elif Frame_unwrapped == "C":
                    self.FRAME = FrameType.PLANET_COROTATION
                    self.OMEGAFRAME = float("nan")
                else:
                    raise NotImplementedError

        meta = inifix.load(file, sections="forbid", parse_scalars_as_lists=True)
        ini = FargoADSGIni(**meta)

        return IniData(
            file=Path(file).resolve(),
            frame=ini.FRAME,
            rotational_rate=ini.OMEGAFRAME,
            output_time_interval=ini.NINTERM * ini.DT,
            meta=inifix.load(file, sections="forbid", skip_validation=True),
        )
