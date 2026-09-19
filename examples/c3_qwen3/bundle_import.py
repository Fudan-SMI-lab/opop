import importlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import TYPE_CHECKING
from uuid import uuid4

from .runner_records import RunnerError

if TYPE_CHECKING:
    from .model_binding import BundleSpec


class BundleImport:
    def __init__(self, bundle: "BundleSpec") -> None:
        self.bundle = bundle
        self.namespace = "_c3_bundle_" + uuid4().hex
        self.temporary = TemporaryDirectory(prefix="c3-bundle-")
        self.root = Path(self.temporary.name).resolve()

    def open(self) -> ModuleType:
        success = False
        try:
            for name, source in self.bundle.sources.items():
                target = self.root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source)
            package = ModuleType(self.namespace)
            package.__path__ = [str(self.root)]
            sys.modules[self.namespace] = package
            entry = Path(self.bundle.document.entry).with_suffix("").parts
            module = importlib.import_module(".".join((self.namespace, *entry)))
            self.check_imports()
            success = True
            return module
        finally:
            if not success:
                self.close()

    def check_imports(self) -> None:
        for name, module in tuple(sys.modules.items()):
            if name.startswith(self.namespace + ".") and module.__file__:
                relative = Path(module.__file__).resolve().relative_to(self.root).as_posix()
                if relative not in self.bundle.sources:
                    raise RunnerError(f"undeclared bundle dependency: {relative}")

    def close(self) -> None:
        for name in tuple(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + "."):
                del sys.modules[name]
        self.temporary.cleanup()
