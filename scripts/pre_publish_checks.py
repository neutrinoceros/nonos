# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "loguru==0.7.3",
#     "packaging==24.2",
#     "pyyaml==6.0.2",
#     "tomli==2.2.1 ; python_version < '3.11'",
# ]
# ///
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger
from packaging.requirements import Requirement
from packaging.specifiers import Specifier
from packaging.version import Version

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


logger.remove()
logger.add(sys.stderr, colorize=True, format="<level>{level:<5} {message}</level>")

REV_REGEXP = re.compile(r"rev:\s+v.*")
STABLE_VER_REGEXP = re.compile(r"^\d+\.*\d+\.\d+$")
STABLE_TAG_REGEXP = re.compile(r"^v\d+\.*\d+\.\d+$")
ROOT = Path(__file__).parents[1]
LIB_DIR = ROOT
README = LIB_DIR / "README.md"
CLI_PYPROJECT_TOML = ROOT / "cli" / "pyproject.toml"
LIB_PYPROJECT_TOML = LIB_DIR / "pyproject.toml"
CITATION_FILE = ROOT / "CITATION.cff"


@dataclass(frozen=True, slots=True)
class Metadata:
    citation_version: Version
    current_lib_static_version: Version
    current_cli_requirement: Requirement
    lib_requires_python: Specifier
    cli_requires_python: Specifier
    latest_git_tag: str

    @property
    def latest_git_version(self) -> Version:
        if not STABLE_TAG_REGEXP.match(self.latest_git_tag):
            logger.error(f"Failed to parse git tag (got {self.latest_git_tag})")
            raise SystemExit(1)
        return Version(self.latest_git_tag)


def check_lib_version(md: Metadata) -> int:
    if not STABLE_VER_REGEXP.match(str(md.current_lib_static_version)):
        logger.error(
            f"Current static version {md.current_lib_static_version} doesn't "
            "conform to expected pattern for a stable sem-ver version.",
        )
        return 1
    elif md.current_lib_static_version < md.latest_git_version:
        logger.error(
            f"Current static version {md.current_lib_static_version} appears "
            f"to be older than latest git tag {md.latest_git_tag}",
        )
        return 1
    else:
        logger.info("Check static version: ok", file=sys.stderr)
        return 0


def check_citation(md: Metadata) -> int:
    if md.citation_version != md.current_lib_static_version:
        logger.error(
            f"CITATION.cff has outdated version data ({md.citation_version}) "
            f"nonos is currently on version {md.current_lib_static_version}",
        )
        return 1
    else:
        logger.info("Check CITATION.cff: ok")
        return 0


def main() -> int:
    with open(CITATION_FILE, "rb") as fh:
        citation_version = Version(yaml.load(fh, yaml.SafeLoader)["version"])

    with open(LIB_PYPROJECT_TOML, "rb") as fh:
        lib_table = tomllib.load(fh)
        current_lib_static_version = Version(lib_table["project"]["version"])
        current_lib_requires_python = Specifier(lib_table["project"]["requires-python"])
    with open(CLI_PYPROJECT_TOML, "rb") as fh:
        cli_table = tomllib.load(fh)
        current_cli_requires_python = Specifier(cli_table["project"]["requires-python"])
        cli_requirements = [
            Requirement(_) for _ in cli_table["project"]["dependencies"]
        ]
    for req in cli_requirements:
        if req.name == "nonos":
            current_cli_requirement = req
            break
    else:
        raise RuntimeError(f"failed to parse {CLI_PYPROJECT_TOML}")

    cp = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0", "--exclude=cli-v*"],
        check=True,
        capture_output=True,
    )
    cp_stdout = cp.stdout.decode().strip()

    md = Metadata(
        citation_version,
        current_lib_static_version,
        current_cli_requirement,
        current_lib_requires_python,
        current_cli_requires_python,
        cp_stdout,
    )

    return check_lib_version(md) + check_citation(md)


if __name__ == "__main__":
    raise SystemExit(main())
