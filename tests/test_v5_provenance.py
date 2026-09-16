"""Verify the selected interpreter's source binding, not a copied venv's label."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
EXPECTED_PACKAGE: Final = PROJECT_ROOT / "src" / "kernel_optimizer" / "__init__.py"
CHECK_IMPORT: Final = """
import importlib
import sys
from importlib.metadata import version
from pathlib import Path

import kernel_optimizer

actual = Path(kernel_optimizer.__file__).resolve()
expected = Path(sys.argv[1]).resolve()
print(actual, flush=True)
assert actual == expected, (actual, expected)
for module in ('pydantic', 'httpx', 'yaml', 'optuna', 'pytest'):
    importlib.import_module(module)
for distribution in ('pydantic', 'httpx', 'PyYAML', 'optuna', 'pytest'):
    print(distribution, version(distribution), flush=True)
assert int(version('pytest').split('.')[0]) >= 8
"""


def test_v5_import_and_dependencies() -> None:
    # Given the interpreter chosen to run this test and this checkout's source.
    command = [sys.executable, "-B", "-c", CHECK_IMPORT, str(EXPECTED_PACKAGE)]

    # When the package and declared dependencies are actually imported.
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        check=False,
    )
    print(result.stdout, end="")
    print(f"IMPORT_CHECK_EXIT_CODE={result.returncode}")

    # Then both source provenance and the pytest version contract hold.
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.splitlines()[0]) == EXPECTED_PACKAGE


def test_reject_v4_import_binding() -> None:
    # Given a portable old checkout, independent of any real v4 installation.
    with TemporaryDirectory(prefix="v5-old-binding-") as directory:
        old_source = Path(directory) / "v4" / "src"
        old_package = old_source / "kernel_optimizer" / "__init__.py"
        old_package.parent.mkdir(parents=True)
        _ = old_package.write_text("", encoding="utf-8")
        assert old_package.resolve() != EXPECTED_PACKAGE
        environment = {**os.environ, "PYTHONPATH": str(old_source)}

        # When the same check imports a real package through the stale binding.
        result = subprocess.run(
            [sys.executable, "-B", "-c", CHECK_IMPORT, str(EXPECTED_PACKAGE)],
            cwd=directory, env=environment, capture_output=True, text=True,
            timeout=30, check=False,
        )
        print(result.stdout, end="")
        print(result.stderr, end="")
        print(f"STALE_BINDING_EXIT_CODE={result.returncode}")

        # Then rejection is caused by the actual wrong origin, not a failed import.
        assert Path(result.stdout.strip()) == old_package.resolve()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "AssertionError" in result.stderr
        assert "ModuleNotFoundError" not in result.stderr
