import sys
import unicodedata

from loguru import logger

_BANDAGE = unicodedata.lookup("ADHESIVE BANDAGE")
_SANGLIER = unicodedata.lookup("POULTRY LEG")
_BONE = unicodedata.lookup("BONE")
_SKULL = unicodedata.lookup("SKULL")

if sys.version_info >= (3, 11):
    _XRAY = unicodedata.lookup("X-RAY")
else:
    # hardcoding the character for Python 3.10 to workaround a lookup error
    _XRAY = "🩻"


def configure_logger(level: int | str = 30, **kwargs) -> None:
    logger.remove()  # remove pre-existing handler
    logger.add(
        sink=sys.stderr,
        format="[{time:HH:mm:ss}] {level.icon} <level>{level:^8}</level> {message}",
        level=level,
        **kwargs,
    )
    logger.configure(
        levels=[
            {"name": "DEBUG", "icon": _BANDAGE, "color": "<GREEN>"},
            {"name": "INFO", "icon": _XRAY, "color": "<green>"},
            {"name": "WARNING", "icon": _SANGLIER, "color": "<b><yellow>"},
            {"name": "ERROR", "icon": _BONE, "color": "<b><red>"},
            {"name": "CRITICAL", "icon": _SKULL, "color": "<b><RED>"},
        ],
    )


def parse_verbose_level(verbose: int) -> str:
    levels = ["WARNING", "INFO", "DEBUG"]
    level = levels[min(len(levels) - 1, verbose)]  # capped to number of levels
    return level


configure_logger(level="WARNING")
