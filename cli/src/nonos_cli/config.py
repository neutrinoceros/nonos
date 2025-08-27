DEFAULTS = {
    # directory where the simulation results are stored
    "datadir": ".",
    # which field?
    "field": "RHO",
    # which operation?
    "operation": "unset",
    # if latitudinal operation
    "theta": "unset",
    # if vertical operation
    "z": "unset",
    # if azimuthal operation
    "phi": "unset",
    # if radial operation
    "distance": "unset",
    # if the geometry of idefix outputs is not recognized
    "geometry": "unset",
    # a single output number ('on') to display or a range (min, max, [step])
    "on": "unset",
    # perturbation of field
    "diff": False,
    # field in log
    "log": False,
    # default extent of the matplotlib window
    "range": "unset",
    # default min field
    "vmin": "unset",
    # default max field
    "vmax": "unset",
    # plane to represent (default is midplane)
    "plane": "unset",
    # do we display the progress (loading+plotting)
    "progressBar": False,
    # do the grid rotate with the planet?
    "corotate": "unset",
    # number of cpus to use
    "ncpu": 1,
    # scaling factor for font size of text in graphs (among other things)
    "scaling": 1,
    # choice of colormap
    "cmap": "RdYlBu_r",
    # name of colorbar
    "title": "unset",
    # conversion factor
    "unit_conversion": 1,
    # select image file format
    "format": "unset",
    # select image resolution
    "dpi": 200,
}
