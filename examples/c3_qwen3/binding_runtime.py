import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType, MappingProxyType
from time import perf_counter
from typing import TYPE_CHECKING

from .bundle_import import BundleImport
from .operator_types import (
    FixtureCodec, Forward, NamedModel, OperatorCall, OperatorFixture, OperatorSite, Replacement, Value,
)
from .runner_records import BindingReceipt, Phase, RunnerError

if TYPE_CHECKING:
    from .model_binding import BundleSpec


class ModelBinding:
    """Own only this model's forward assignments; diagnostic counters never score timing."""

    def __init__(self, model: NamedModel, codec: FixtureCodec) -> None:
        self.modules = dict(model.named_modules())
        self.originals = {name: module.forward for name, module in self.modules.items()}
        self.instance_forward = {name: "forward" in vars(module) for name, module in self.modules.items()}
        self.installed: dict[str, Forward] = {}
        self.codec = codec
        self.sites: dict[str, tuple[str, ...]] = {}
        self.active: dict[str, Replacement] = {}
        self.imported: BundleImport | None = None
        self.bundle: BundleSpec | None = None
        self.phase: Phase = "prefill"
        self.tracing = False
        self.capturing = False
        self.record_trace = False
        self.counts: dict[str, dict[str, int]] = {}
        self.trace: list[dict[str, str | int]] = []
        self.fixtures: dict[tuple[str, Phase], OperatorFixture] = {}
        self.fixture_bytes = 0

    def register_site(self, site_id: str, module_paths: tuple[str, ...]) -> None:
        if self.bundle is not None or not module_paths or site_id in self.sites:
            raise RunnerError("register unique nonempty sites before binding")
        occupied = {p for paths in self.sites.values() for p in paths}
        if any(p not in self.modules or p in occupied for p in module_paths):
            raise RunnerError("site module missing or belongs to another group")
        self.sites[site_id] = module_paths

    def restore(self) -> None:
        for path in self.installed:
            if self.instance_forward[path]:
                self.modules[path].forward = self.originals[path]
            else:
                delattr(self.modules[path], "forward")
        self.installed.clear()
        self.active.clear()
        self.bundle = None
        if self.imported is not None:
            self.imported.close()
            self.imported = None

    def bind(self, bundle: "BundleSpec") -> BindingReceipt:
        self.restore()
        success = False
        try:
            if bundle.document.sites:
                self.imported = BundleImport(bundle)
                module = self.imported.open()
                for site in bundle.document.sites:
                    if site.site_id not in self.sites:
                        raise RunnerError(f"unresolved traced site: {site.site_id}")
                    function = getattr(module, site.replacement_callable, None)
                    if not isinstance(function, Replacement):
                        raise RunnerError(f"replacement is not callable: {site.replacement_callable}")
                    self.active[site.site_id] = function
            self.bundle = bundle
            self._install()
            success = True
            return BindingReceipt(bundle.bundle_sha256, bundle.source_hashes, bundle.params_sha256,
                                  tuple(self.active), True)
        finally:
            if not success:
                self.restore()

    def _install(self) -> None:
        for site_id, paths in self.sites.items():
            for path in paths:
                forward = self._forward(site_id, path)
                self.modules[path].forward = forward
                self.installed[path] = forward

    def verify_installed(self) -> None:
        for site in self.active:
            for path in self.sites[site]:
                if self.modules[path].forward is not self.installed[path]:
                    self.restore()
                    raise RunnerError(f"candidate dispatch changed after binding: {path}")

    def _forward(self, site_id: str, path: str) -> Forward:
        module = self.modules[path]
        original = self.originals[path]
        site = OperatorSite(module, MappingProxyType(dict(module.named_parameters())), path)

        def forward(*args: Value, **kwargs: Value) -> Value:
            call = OperatorCall(args, kwargs)
            fixture_key = (site_id, self.phase)
            capture = self.capturing and fixture_key not in self.fixtures
            if capture and self.fixture_bytes + self.codec.size(call) * 3 > 128 * 1024 * 1024:
                raise RunnerError("selected local fixtures exceed the 128 MiB resident budget")
            before = self.codec.clone(call) if capture else None
            replacement = self.active.get(site_id)
            started = perf_counter() if self.record_trace else 0.0
            event = None
            if self.tracing:
                row = self.counts.setdefault(path, {"replacement_calls": 0, "old_calls": 0})
                if replacement is not None:
                    row["replacement_calls"] += 1
            if self.record_trace:
                event = {"site_id": site_id, "module_path": path, "phase": self.phase,
                         "shape": self.codec.describe(call), "source": inspect.getsourcefile(original) or "unknown"}
                self.trace.append(event)
            success = False
            try:
                if replacement is None:
                    result = original(*args, **kwargs)
                else:
                    if self.bundle is None:
                        raise RunnerError("replacement has no bundle identity")
                    result = replacement(site, call, self.bundle.params)
                if before is not None:
                    copied = self.codec.clone(OperatorCall((result, self.codec.state(call)), {}))
                    size = self.codec.size(before) + self.codec.size(copied)
                    if self.fixture_bytes + size > 128 * 1024 * 1024:
                        raise RunnerError("selected local fixtures exceed the 128 MiB resident budget")
                    self.fixtures[fixture_key] = OperatorFixture(path, before, copied.args[0], copied.args[1])
                    self.fixture_bytes += size
                if event is not None:
                    event["cpu_dispatch_us_inclusive"] = int((perf_counter() - started) * 1e6)
                success = True
                return result
            finally:
                if not success:
                    self.restore()
        return forward

    @contextmanager
    def diagnostic(self, *, capture: bool = False, trace: bool = False) -> Iterator[None]:
        if self.tracing or (capture and self.active):
            raise RunnerError("nested diagnostic or candidate-as-baseline fixture")
        self.counts = {}
        self.trace = []
        self.tracing, self.capturing = True, capture
        self.record_trace = trace or capture
        self._install()
        previous = sys.getprofile()
        codes = {}
        for paths in self.sites.values():
            for path in paths:
                fn = self.originals[path]
                codes.setdefault(getattr(fn, "__code__", None), []).append(path)
                unwrapped = inspect.unwrap(fn)
                if unwrapped is not fn:
                    codes.setdefault(getattr(unwrapped, "__code__", None), []).append(path)

        def profile[T](frame: FrameType, event: str, arg: T) -> None:
            if event == "call":
                for path in codes.get(frame.f_code, ()):
                    if frame.f_locals.get("self") is self.modules[path]:
                        row = self.counts.setdefault(path, {"replacement_calls": 0, "old_calls": 0})
                        row["old_calls"] += 1
            if previous is not None:
                previous(frame, event, arg)
        sys.setprofile(profile)
        success = False
        try:
            yield
            if self.imported is not None:
                self.imported.check_imports()
            success = True
        finally:
            sys.setprofile(previous)
            self.tracing, self.capturing = False, False
            self.record_trace = False
            if not success:
                self.restore()

    def coverage(self) -> dict[str, dict[str, int]]:
        return {site: {key: sum(self.counts.get(p, {}).get(key, 0) for p in paths)
                       for key in ("replacement_calls", "old_calls")}
                for site, paths in self.sites.items()}

    def source_catalog(self) -> dict[str, str]:
        return {path: inspect.getsource(self.originals[path]) for paths in self.sites.values() for path in paths}

    def require_coverage(self) -> None:
        for site in self.active:
            for path in self.sites[site]:
                row = self.counts.get(path, {})
                if not row.get("replacement_calls") or row.get("old_calls"):
                    self.restore()
                    raise RunnerError(f"replacement not executed exclusively: {path}")

    def validate_local(self) -> None:
        success = False
        try:
            for site, replacement in self.active.items():
                fixtures = [fixture for (sid, _), fixture in self.fixtures.items() if sid == site]
                if not fixtures or self.bundle is None:
                    raise RunnerError(f"missing original local fixtures: {site}")
                for fixture in fixtures:
                    call = self.codec.clone(fixture.call)
                    module = self.modules[fixture.module_path]
                    context = OperatorSite(module, MappingProxyType(dict(module.named_parameters())), fixture.module_path)
                    result = replacement(context, call, self.bundle.params)
                    self.codec.check(fixture.expected, result)
                    self.codec.check_state(fixture.touched_state, self.codec.state(call))
            success = True
        finally:
            if not success:
                self.restore()
