import os
import re
import textwrap
from importlib.metadata import version
from importlib.util import find_spec

import inifix
import pytest

from nonos_cli import main
from nonos_cli.config import DEFAULTS

TIMESTAMP_PREFIX = r"\[\d\d:\d\d:\d\d\]\s+"


@pytest.fixture()
def minimal_paramfile(tmp_path):
    ifile = tmp_path / "nonos.ini"
    # check that this setup still makes sense
    assert DEFAULTS["field"] == "RHO"
    with open(ifile, "w") as fh:
        fh.write("field  VX1")
    return ifile


def test_no_inifile(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main([])
    assert ret != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "Could not find a parameter file in" in err


def test_default_conf(capsys):
    ret = main(["-config"])
    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""

    # validate output is reusable
    inifix.loads(out)


def test_version(capsys):
    ret = main(["-version"])
    assert ret == 0

    out, err = capsys.readouterr()
    assert err == ""
    assert out == (
        f"nonos-cli  {version('nonos-cli')}\nnonos      {version('nonos')}\n"
    )


def test_logo(capsys):
    ret = main(["-logo"])

    assert ret == 0
    out, err = capsys.readouterr()
    expected = textwrap.dedent(
        r"""
                                                                     `!)}$$$$})!`
                 `,!>|))|!,                                        :}&#&$}{{}$&#&};`
              ~)$&&$$}}$$&#$).                                   '}#&}|~'.```.'!($#$+
           `=$&$(^,..``.'~=$&&=                                `|&#}!'`         `'?$#}.
          !$&}:.`         `.!$#$'                             :$#$^'`       ``     .$#$`
         ^&&+.              `'(&$!                          `)&&),`                 !##!
        `$#^      `.   `     `={$&{`                       ^$&$(!'  ``     `,.  ``  !##!
        ,#$ ``                 .>}&${?!:'`   ```..'',:!+|{$&${:`   `'`             `}&}`
        `$$`                       ,;|{}$$$&&&&&$$$$}}()<!,`   '`                 `}&}`
         +&}`   `   |:|.\    |:|            `.```                                  :$$!
          !$$'`!}|  |:|\.\   |:|      __                      __       ___       .{#$
           '$&})$:  |:| \.\  |:|   /./  \.\   |:|.\  |:|   /./  \.\   |:|  \.\  '$$#}
            `}#&;   |:|  \.\ |:|  |:|    |:|  |:|\.\ |:|  |:|    |:|  |:|___     :$)&$`
            `{&!    |:|   \.\|:|  |:|    |:|  |:| \.\|:|  |:|    |:|       |:|    :!{&}`
           :$$,     |:|    \.|:|   \.\__/./   |:|  \.|:|   \.\__/./   \.\__|:|     `:}#}`
          ^&$.                                                                       .$#^
          +&$.                 ``'^)}$$$$$({}}}$$$$$$$$$$}}(|>!.`~:,.                 }#)
         '&#|                 ,|$##$>'`                `'~!)$##$$$)?^,`           ` :&&:
         ,&#}`  ``       .` `:{$&}:                          ~}&$)^^+=^`  `` ..  .|&#)
          |&#$:```   `` '::!}$&},                              !$&$|++^^:,:~',!!($#&^
           ,}&#${^~,,,:!|}$&&(.                                  ^$#$}{)|?|)(}$&#$?`
             :{&##$$$$$&##$).                                      ~($&#&&##&$}=,
               `:|}$$$$}):`

        Visualization tool for idefix/pluto/fargo3d (M)HD simulations of protoplanetary disks
        """.lstrip("\n")
    )
    expected += "\n".join(
        ["", f"nonos-cli  {version('nonos-cli')}", f"nonos      {version('nonos')}", ""]
    )
    assert out == expected
    assert err == ""


ARGS_TO_CHECK = {
    "vanilla_conf": ["-geometry", "polar"],
    "diff": ["-geometry", "polar", "-diff"],
    "log": ["-geometry", "polar", "-log"],
    "movie_xy": ["-geometry", "polar", "-all", "-plane", "x", "y"],
    "movie_with_diff": ["-geometry", "polar", "-all", "-diff"],
    "movie_with_multiproc": ["-geometry", "polar", "-all", "-ncpu", "2"],
}


@pytest.mark.parametrize("argv", ARGS_TO_CHECK.values(), ids=ARGS_TO_CHECK.keys())
def test_plot_simple(argv, simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(argv + ["-dir", str(simulation_dir), "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""
    assert ret == 0
    assert len(list(tmp_path.glob("*.png"))) > 0


@pytest.mark.parametrize("format", ["pdf", "png", "jpg"])
def test_common_image_formats(format, simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-dir", str(simulation_dir), "-fmt", format, "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""
    assert ret == 0
    assert len(list(tmp_path.glob(f"*.{format}"))) > 0


def test_plot_simple_corotation(planet_simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    # just check that the call returns no err
    ret = main(["-cor", "0", "-dir", str(planet_simulation_dir), "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    assert ret == 0


def test_unknown_geometry(test_data_dir, tmp_path):
    os.chdir(tmp_path)
    with pytest.raises(
        RuntimeError, match=r"Geometry couldn't be determined from data"
    ):
        main(["-dir", str(test_data_dir / "idefix_rwi")])


def test_newvtk_geometry(test_data_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-cor", "0", "-dir", str(test_data_dir / "idefix_newvtk_planet2d")])
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""
    assert ret == 0


def test_verbose_info(simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-v", "-dir", str(simulation_dir), "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert out == ""
    assert "Operation took" in err
    assert "INFO" in err
    assert ret == 0


def test_load_config_file(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-input", "nonos.ini", "-config"])

    assert ret == 0
    out, err = capsys.readouterr()
    assert "Using parameters from" in err
    conf = inifix.loads(out)
    assert conf["field"] == "VX1"


def test_isolated_mode(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-config", "-isolated"])

    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""
    conf = inifix.loads(out)
    assert conf["field"] == DEFAULTS["field"]


def test_pbar(simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-pbar", "-dir", str(simulation_dir), "-geometry", "polar"])

    out, err = capsys.readouterr()
    assert err == ""
    assert "Processing snapshots" in out
    assert ret == 0


@pytest.mark.parametrize(
    "cmap",
    [
        pytest.param(
            cmap_name,
            marks=pytest.mark.skipif(
                not find_spec(pkg_name), reason=f"{pkg_name} is not installed"
            ),
        )
        for pkg_name, cmap_name in [
            ("cblind", "cb.rainbow"),
            ("cmocean", "cmo.thermal"),
            ("cmasher", "cmr.dusk"),
            ("cmyt", "cmyt.arbre"),
        ]
    ],
)
def test_colormap_extensions_integration(cmap, capsys, test_data_dir, tmp_path):
    simdir = str(test_data_dir / "idefix_planet3d")
    os.chdir(tmp_path)
    ret = main(["-dir", simdir, "-geometry", "polar", "-cmap", cmap])
    assert ret == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""


@pytest.mark.parametrize(
    "cmap, pkg",
    [
        pytest.param(
            cmap_name,
            pkg_name,
            marks=pytest.mark.skipif(
                find_spec(pkg_name), reason=f"{pkg_name} is installed"
            ),
        )
        for pkg_name, cmap_name in [
            ("cblind", "cb.rainbow"),
            ("cmocean", "cmo.thermal"),
            ("cmasher", "cmr.dusk"),
            ("cmyt", "cmyt.arbre"),
        ]
    ],
)
def test_colormap_extensions_missing_package(
    cmap, pkg, capsys, test_data_dir, tmp_path
):
    simdir = str(test_data_dir / "idefix_planet3d")
    os.chdir(tmp_path)
    ret = main(["-dir", simdir, "-geometry", "polar", "-cmap", cmap])
    assert ret == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert re.fullmatch(
        rf"{TIMESTAMP_PREFIX}🍗 WARNING\s* requested colormap {cmap!r}, "
        rf"but {pkg} is not installed\. The default colormap will be used instead\.\n",
        err,
    )


def test_unknown_colormap_package_prefix(capsys, test_data_dir, tmp_path):
    simdir = str(test_data_dir / "idefix_planet3d")
    os.chdir(tmp_path)
    ret = main(
        ["-dir", simdir, "-geometry", "polar", "-cmap", "cmunknown.thismapdoesnexist"]
    )
    assert ret == 0

    out, err = capsys.readouterr()
    assert out == ""
    assert re.fullmatch(
        rf"{TIMESTAMP_PREFIX}🍗 WARNING\s* requested colormap 'cmunknown.thismapdoesnexist' "
        r"with the unknown prefix 'cmunknown'\. The default colormap will be used instead\.\n",
        err,
    )
