## nonos-cli
[![PyPI](https://img.shields.io/pypi/v/nonos-cli.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/nonos-cli/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/nonos-cli)](https://pypi.org/project/nonos-cli/)
[![Documentation Status](https://readthedocs.org/projects/nonos/badge/?version=latest)](https://nonos.readthedocs.io/en/latest/cli/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

`nonos-cli` is a companion command line interface (CLI) to create nonos plots
directly from the terminal, adding a `nonos` command to your environment.

### Installation

The recommended method is to install `nonos-cli` globally, but still in isolation,
with `uv`. Get a minimal install with
```shell
uv tool install nonos-cli
```
You may want to include optional dependencies too, which is achieved with
```shell
uv tool install 'nonos-cli[all]'
```

Alternatively, use `pipx` (which also supports the addition of `[all]`)
```shell
pipx install nonos-cli
```

#### Install in development (editable) mode

If you want to hack around the CLI and run some manual tests without
re-installing it between edits, you'll need to first clone [the source
repository](https://github.com/la-niche/nonos), then, from the top level, run

```shell
uv tool install --editable ./cli
```
or
```shell
pipx install --editable ./cli
```

Alternatively, and using uv, you can also run the in-tree version of the CLI without installing it globally with the following workflow

```shell
uv sync --all-packages
uv run nonos <...>
```

### Usage

The nonos CLI gets its parameters from three sources:
- command line parameters
- a configuration file
- default values

Command line parameters take priority over the configuration file, which itself takes priority over default values.

To get help, run
```shell
nonos --help
```

<!-- [[[cog
from io import StringIO
import os
import cog
from nonos_cli import get_parser

columns = os.environ.get("COLUMNS")
os.environ["COLUMNS"] = "90"
help = StringIO()
get_parser().print_help(file=help)
os.environ["COLUMNS"] = columns or ""

cog.out(f"```\n{help.getvalue()}\n```")
]]] -->
```
usage: nonos [-h] [-dir DATADIR] [-field FIELD]
             [-geometry {polar,cylindrical,spherical,cartesian}]
             [-operation {vm,vp,vz,lt,lp,aa,ap,apl,rr} [{vm,vp,vz,lt,lp,aa,ap,apl,rr} ...]]
             [-plane PLANE [PLANE ...]] [-corotate COROTATE] [-range RANGE [RANGE ...]]
             [-vmin VMIN] [-vmax VMAX] [-theta THETA] [-z Z] [-phi PHI]
             [-distance DISTANCE] [-cpu NCPU] [-on ON [ON ...] | -all] [-diff] [-log]
             [-pbar] [-scaling SCALING] [-cmap CMAP] [-title TITLE]
             [-uc UNIT_CONVERSION] [-fmt FORMAT] [-dpi DPI] [-input INPUT | -isolated]
             [-d | -version | -logo | -config] [-v]

Visualization tool for idefix/pluto/fargo3d (M)HD simulations of protoplanetary disks

options:
  -h, --help            show this help message and exit
  -dir DATADIR          location of output files and param files (default: '.').
  -field FIELD          name of field to plot (default: 'RHO').
  -geometry {polar,cylindrical,spherical,cartesian}
                        if the geometry of idefix outputs is not recognized (default:
                        'unset').
  -operation {vm,vp,vz,lt,lp,aa,ap,apl,rr} [{vm,vp,vz,lt,lp,aa,ap,apl,rr} ...]
                        operation to apply to the fild (default: 'unset').
  -plane PLANE [PLANE ...]
                        abscissa and ordinate of the plane of projection (default:
                        'unset'), example: r phi
  -corotate COROTATE    planet number that defines with which planet the grid corotates.
  -range RANGE [RANGE ...]
                        range of matplotlib window (default: unset), example: x x -2 2
  -vmin VMIN            min value (default: unset)
  -vmax VMAX            max value (default: unset)
  -theta THETA          if latitudinal operation (default: unset)
  -z Z                  if vertical operation (default: unset)
  -phi PHI              if azimuthal operation (default: unset)
  -distance DISTANCE    if radial operation (default: unset)
  -cpu, -ncpu NCPU      number of parallel processes (default: 1).
  -on ON [ON ...]       output number(s) (on) to plot. This can be a single value or a
                        range (start, end, [step]) where both ends are inclusive.
                        (default: last output available).
  -all                  save an image for every available snapshot (this will force
                        show=False).
  -scaling SCALING      scale the overall sizes of features in the graph (fonts,
                        linewidth...) (default: 1).
  -cmap CMAP            choice of colormap for the 2D maps (default: 'RdYlBu_r').
  -title TITLE          name of the field in the colorbar for the 2D maps (default:
                        'unset').
  -uc, -unit_conversion UNIT_CONVERSION
                        conversion factor for the considered quantity (default: '1').
  -fmt, -format FORMAT  select output image file format (default: unset)
  -dpi DPI              image file resolution (default: 200)

boolean flags:
  -diff                 plot the relative perturbation of the field f, i.e. (f-f0)/f0.
  -log                  plot the log10 of the field f, i.e. log(f).
  -pbar                 display a progress bar

CLI-only options:
  -input, -i INPUT      specify a configuration file.
  -isolated             ignore any existing 'nonos.ini' file.
  -d, -display          open a graphic window with the plot (only works with a single
                        image)
  -version, --version   show raw version number and exit
  -logo                 show Nonos logo with version number, and exit.
  -config               show configuration and exit.
  -v, -verbose, --verbose
                        increase output verbosity (-v: info, -vv: debug).

```
<!-- [[[end]]] -->

The `-operation` command allows you to choose what operation is applied to the data, and can be paired with `-z`, `-theta`, `-phi`, or `-distance` depending on the operation.
- `vm`: vertical_at_midplane
- `vz`: vertical_at_z, can be paired with `-z` to give the altitude at which the vertical slice is performed.
- `vp`: vertical_projection, can be paired with `-z` to give the interval of the vertical integral.
- `lt`: latitudinal_at_theta, can be paired with `-theta` to give the latitude at which the latitudinal slice is performed.
- `lp`: latitudinal_projection, can be paired with `-theta` to give the interval of the latitudinal integral.
- `ap`: azimuthal_at_phi, can be paired with `-phi` to give the azimuth at which the azimuthal slice is performed.
- `apl`: azimuthal_at_planet, has to be paired with `-corotate` to perform a slice at the planet azimuth.
- `aa`: azimuthal_average
- `rr`: radial_at_r, can be paired with `-distance` to give the distance at which the radial slice is performed.
You can cumulate some operations, like `lp` and `aa` which will given for example for `-field=RHO` the gas surface density.

Note that for old idefix outputs, you will need to add the `-geometry` command to process the data.

### Using a configuration file

The CLI will read parameters from a local file named `nonos.ini` if it exists,
or any other name specified using the `-i/-input` parameter.
To ignore any existing `nonos.ini` file, use the `-isolated` flag.

One way to configure nonos is to use
```shell
nonos -config
```

which prints the current configuration to stdout.
You can then redirect it to get a working configuration file as
```shell
nonos -config > nonos.ini
```
This method can also be used to store a complete configuration file from command line arguments:
```shell
nonos -ncpu 8 -cmap viridis -operation vm -diff -vmin=-10 -vmax=+100 -config
```
As of nonos 0.19.0 + nonos-cli 0.1.0, this will print
```
# Generated with nonos 0.19.0 + nonos-cli 0.1.0
datadir            .
field              RHO
operation          vm
theta              unset
z                  unset
phi                unset
distance           unset
geometry           unset
on                 unset
diff               True
log                False
range              unset
vmin               -1e1
vmax               1e2
plane              unset
progressBar        False
corotate           unset
ncpu               8
scaling            1
cmap               viridis
title              unset
unit_conversion    1
format             unset
dpi                200
```
