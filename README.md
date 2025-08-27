# nonos
[![PyPI](https://img.shields.io/pypi/v/nonos.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/nonos/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/nonos)](https://pypi.org/project/nonos/)
[![Documentation Status](https://readthedocs.org/projects/nonos/badge/?version=latest)](https://nonos.readthedocs.io/en/latest/?badge=latest)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

nonos is a Python 2D visualization library for planet-disk numerical simulations.
It supports vtk-formatted data from Pluto and Idefix, and dat-formatted data for Fargo-adsg and Fargo3D.

For more, read [the documentation !](https://nonos.readthedocs.io/en/latest/?badge=latest)

##### Data Formats
We list here the accepted formats for the data:
Pluto and Idefix: data.\*\*\*\*.vtk
Fargo-adsg: gasdens.dat, gasvy\*.dat, gasvx\*.dat
Fargo3D: same as Fargo-adsg + gasvz\*.dat

## Development status

`nonos` is considered public beta software: we are actively improving the
design and API, but we are not at the point where we want to bless the
current state as stable yet. We *are* trying to keep breaking changes to a
minimum, and run deprecation cycles to minimize the pain, however they might
happen in any minor release, so if you rely on `nonos` for your own work
(thank you !), we strongly encourage you to follow along releases and
upgrade frequently, so we have more opportunities to discuss if something
breaks.

## Installation

Get nonos and its minimal set of dependencies as

```shell
python -m pip install nonos
```

Optionally, you can install with the companion command line interface too
```shell
python -m pip install "nonos[cli]"
```

or, to also get all optional dependencies (CLI included)
```shell
python -m pip install "nonos[all]"
```

## Usage

```python
from nonos.api import GasDataSet
import matplotlib.pyplot as plt

plt.close("all")
# We use GasDataSet which takes as argument the output number of the output file given by idefix/pluto/fargo
# contains in particular a dictionary with the different fields.
ds = GasDataSet(43, geometry="polar", directory="nonos/tests/data/idefix_planet3d")
# We select the GasField "RHO", then
# we perform a vertical slice in the midplane,
# and make the result plotable in the xy plane,
# rotating the grid given the planet number 0
# (which orbit is described in the planet0.dat file).
dsop = ds["RHO"].vertical_at_midplane().map("x", "y", planet_corotation=0)
fig, ax = plt.subplots()
# dsop is now a Plotable object.
# We represent its log10, with a given colormap,
# and we display the colorbar by adding the argument title.
dsop.plot(fig, ax, log=True, cmap="inferno", title=r"$\rho_{\rm mid}$")
ax.set_aspect("equal")

# This time, we perform a latitudinal projection,
# i.e. the integral of "RHO" between -theta and theta,
# and then an azimuthal average,
# before mapping it in the radial ("R") direction.
dsop = ds["RHO"].latitudinal_projection(theta=3*0.05).azimuthal_average().map("R")
fig, ax = plt.subplots()
# We display the y-axis by adding the argument title.
dsop.plot(fig, ax, c="k", title=r"$\Sigma$")
plt.show()
```


### Reusing `nonos`' style
*requires matplotlib >= 3.7*

`nonos` CLI uses a custom style that can be reused programmatically, without
importing the package, using matplotlib API
```python
import matplotlib.pyplot as plt
plt.style.use("nonos.default")
```

See [`matplotlib.style`'s documentation](https://matplotlib.org/stable/api/style_api.html) for more.
