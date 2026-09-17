"""Stage allowlisted Python helpers without changing primary or reference namespaces."""

import shutil
from pathlib import Path

from scripts.experiments.c2_local_inputs import InputError


def copy_helpers(helpers: tuple[Path, ...], source_root: Path | None, target: Path) -> tuple[Path, ...]:
    target.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for helper in helpers:
        if source_root is None:
            raise InputError("helper paths require their source root")
        destination = target / helper.resolve().relative_to(source_root.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(helper, destination)
        copied.append(destination)
    return tuple(copied)
