import sys
import unicodedata

from loguru import logger
from termcolor import cprint

_BONE_EMOJI = unicodedata.lookup("BONE")


def configure_logger(level: int | str = 30, **kwargs) -> None:
    logger.remove()  # remove pre-existing handler
    logger.add(
        sink=sys.stdout,
        format="[{time:HH:mm:ss}] nonos <level>{level:<8}</level> {message}",
        level=level,
        **kwargs,
    )


def print_warn(message) -> None:
    """
    Pretty-print a warning.
    """
    print(_BONE_EMOJI, end=" ", file=sys.stderr)
    cprint("Warning", color="red", attrs=["bold"], end=" ", file=sys.stderr)
    print(message, file=sys.stderr)


def print_err(message) -> None:
    """
    Pretty-print an error message.
    """
    print(_BONE_EMOJI, end=" ", file=sys.stderr)
    cprint(
        "Error",
        color="white",
        on_color="on_red",
        attrs=["bold"],
        end=" ",
        file=sys.stderr,
    )
    print(message, file=sys.stderr)


def parse_verbose_level(verbose: int) -> str:
    levels = ["WARNING", "INFO", "DEBUG"]
    level = levels[min(len(levels) - 1, verbose)]  # capped to number of levels
    return level


configure_logger(level="WARNING")
