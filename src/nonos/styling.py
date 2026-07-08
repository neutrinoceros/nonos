import sys

import matplotlib as mpl

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


def scale_mpl(scaling: float) -> None:
    # Scale all the parameters by the same factor depending on the level
    # this heavily borrows from seaborn.set_context (v0.11.1) see
    # https://github.com/mwaskom/seaborn/blob/a41703e7fddf8f66b1fd5f994f983b37e865a3b2/seaborn/rcmod.py#L439
    # https://github.com/mwaskom/seaborn/blob/a41703e7fddf8f66b1fd5f994f983b37e865a3b2/seaborn/rcmod.py#L338

    mpl.rcParams |= {
        "font.size": 12.0 * scaling,
        "axes.labelsize": 12.0 * scaling,
        "axes.titlesize": 12.0 * scaling,
        "xtick.labelsize": 11.0 * scaling,
        "ytick.labelsize": 11.0 * scaling,
        "legend.fontsize": 11.0 * scaling,
        "legend.title_fontsize": 12.0 * scaling,
        "axes.linewidth": 1.25 * scaling,
        "grid.linewidth": 1.0 * scaling,
        "lines.linewidth": 1.5 * scaling,
        "lines.markersize": 6.0 * scaling,
        "patch.linewidth": 1.0 * scaling,
        "xtick.major.width": 1.25 * scaling,
        "ytick.major.width": 1.25 * scaling,
        "xtick.minor.width": 1.0 * scaling,
        "ytick.minor.width": 1.0 * scaling,
        "xtick.major.size": 6.0 * scaling,
        "ytick.major.size": 6.0 * scaling,
        "xtick.minor.size": 4.0 * scaling,
        "ytick.minor.size": 4.0 * scaling,
    }


@deprecated(
    "nonos.styling.set_mpl_style is deprecated since v0.20.0 "
    "and may be removed in a future version. "
    "Use matplolib.style.use('nonos.default') + nonos.styling.scale_mpl(<scaling>) "
    "directly instead"
)
def set_mpl_style(scaling: float) -> None:
    import matplotlib.style

    matplotlib.style.use("nonos.default")
    scale_mpl(scaling)
