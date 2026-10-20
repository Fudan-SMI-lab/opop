"""Candidate callables are installed at observed sites, with actual-call coverage."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.util
from pathlib import Path

from kernel_optimizer.project.discovery import _phase, tensor_description


class BindingSession:
    def __init__(self, adapter, state, specs: list[dict], *, rank=0, world_size=1, bundle_id=None,
                 backend="legacy"):
        if backend not in {"legacy", "scoped"}:
            raise ValueError(f"unknown binding backend: {backend}")
        self.backend = backend
        self.execution_plan = {"requested": backend, "effective": "direct",
                               "reason": "no functional replacements"}
        self.scope_enabled = False
        self._invoking = False
        self._scope_depth = 0
        self._scope_frames = []
        self.adapter, self.state = adapter, state
        self.rank, self.world_size = rank, world_size
        self.all_specs = {spec["id"]: spec for spec in specs}
        self.specs = {spec["id"]: {**spec, **spec.get("rank_specs", {}).get(str(rank), {})}
                      for spec in specs if rank in spec.get("ranks", range(world_size))}
        self.models = adapter.models(state)
        self.references, self.modules = {}, {}
        self.counts = Counter()
        self.fallbacks = Counter()
        self.reference_calls = Counter()
        self.routes = Counter()
        self.tracking = True
        self.diagnostic = False
        self.bundle_id = bundle_id
        self.restores = []
        self.functionals = {}
        self.handles = []
        self.active_modules = ContextVar("candidate_module_context", default=())
        self._mode = None
        for ident, spec in self.specs.items():
            if spec["kind"] == "module":
                root, _, path = spec["target"].partition(".")
                module = self.models[root].get_submodule(path) if path else self.models[root]
                self.modules[ident] = module
                self.references[ident] = module.forward
            elif spec["kind"] == "project":
                self.references[ident] = adapter.run
            else:
                import torch
                function = torch.ops
                for part in spec["target"].split("."):
                    function = getattr(function, part)
                self.references[ident] = function

    def install(self, file: Path, parameters: dict, *, diagnostic_passthrough=False) -> list[str]:
        observed_references = {}
        for ident, function in self.references.items():
            def reference(*args, _function=function, _ident=ident, **kwargs):
                if self.tracking:
                    self.reference_calls[_ident] += 1
                return _function(*args, **kwargs)
            observed_references[ident] = reference
        if diagnostic_passthrough:
            # Explicit ablation only: preserve the binding sites but omit the
            # candidate installer and computation. This cannot publish a version.
            replacements = observed_references
        else:
            loader = importlib.util.spec_from_file_location("opop_candidate_" + str(len(self.restores)), file)
            module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
            replacements = module.build({"state": self.state, "models": self.models, "modules": self.modules,
                                     "references": observed_references, "targets": self.specs, "parameters": parameters,
                                     "rank": self.rank, "world_size": self.world_size, "all_targets": self.all_specs,
                                     "record_route": self.record_route})
        if not isinstance(replacements, dict):
            raise ValueError("build(context) must return a call_id -> callable mapping")
        if set(replacements) - set(self.specs):
            raise ValueError("candidate returned an undeclared replacement site")
        physical = {}
        for ident, function in replacements.items():
            if not callable(function):
                raise TypeError("replacement must be callable")
            spec = self.specs[ident]
            def wrapped(*args, _function=function, _ident=ident, **kwargs):
                if self.tracking:
                    self.counts[_ident] += 1
                if self.diagnostic:
                    import json
                    import torch
                    # Kineto versions may write range names into JSON without
                    # escaping embedded quotes. A hex payload preserves exact IDs.
                    label = "opop_candidate::"+json.dumps({"bundle": self.bundle_id, "call": _ident, "phase": _phase.get()}, separators=(",", ":")).encode().hex()
                    with torch.profiler.record_function(label):
                        return _function(*args, **kwargs)
                return _function(*args, **kwargs)
            if spec["kind"] == "functional":
                self.functionals[ident] = wrapped
            else:
                owner = self.modules[ident] if spec["kind"] == "module" else self.adapter
                attr = "forward" if spec["kind"] == "module" else "run"
                key = (id(owner), attr)
                entry = physical.setdefault(key, (owner, attr, getattr(owner, attr), []))
                if any(other["phase"] == spec["phase"] for other, _ in entry[3]):
                    raise ValueError("ambiguous replacements for one physical site and phase")
                entry[3].append((spec, wrapped))
        for owner, attr, original, variants in physical.values():
            def dispatch(*args, _variants=variants, _original=original, **kwargs):
                for spec, replacement in _variants:
                    if spec["phase"] != _phase.get():
                        continue
                    if spec["kind"] == "module":
                        signature = {"args": tensor_description(args), "kwargs": tensor_description(kwargs)}
                        if signature not in spec["signatures"]:
                            if self.tracking:
                                self.fallbacks[spec["id"]] += 1
                            continue
                    return replacement(*args, **kwargs)
                return _original(*args, **kwargs)
            self.restores.append((owner, attr, original))
            setattr(owner, attr, dispatch)
        if self.functionals:
            self._create_dispatch_mode()
        return list(replacements)

    def record_route(self, call_id, route):
        """Optional candidate-declared path telemetry; independent coverage stays mandatory."""
        if not self.tracking:
            return
        if call_id not in self.specs or not isinstance(route, str) or not route:
            raise ValueError("route telemetry requires a declared call and a nonempty path label")
        self.routes[(call_id, route)] += 1

    def path_telemetry(self):
        return {"reference_invocations": dict(self.reference_calls),
                "candidate_declared_routes": [{"call": c, "route": r, "count": n} for (c, r), n in sorted(self.routes.items())],
                "interpretation": "Reference counters observe calls through context.references only, not all possible captured aliases. A reference call may be fallback or part of a composite; declared route labels are candidate telemetry, not independent proof. Corroborate implementation with traces."}

    def _create_dispatch_mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        binding = self
        # A generated model candidate can address thousands of shape-specific
        # sites. Searching every site on every aten call makes binding itself
        # dominate the project; index by operator, phase and module context.
        indexed = {}
        for ident, replacement in self.functionals.items():
            spec = self.specs[ident]
            key = (spec["target"], spec["phase"], spec.get("site_parent"))
            indexed.setdefault(key, []).append((spec, replacement))
        parents = {self.specs[ident].get("site_parent") for ident in self.functionals}
        phases = {self.specs[ident]["phase"] for ident in self.functionals}
        available = {hashlib.sha256((root + ("."+path if path else "")+"|"+phase).encode()).hexdigest()[:24]
                     for root, model in self.models.items() for path, _ in model.named_modules()
                     for phase in phases}
        self.scope_enabled = self.backend == "scoped" and bool(parents) and parents <= available
        self.execution_plan.update(
            effective="scoped" if self.scope_enabled else "legacy",
            reason="all functional sites have observed module boundaries" if self.scope_enabled else
                   "legacy requested" if self.backend == "legacy" else
                   "functional sites include an unscoped or unresolved boundary",
            functional_sites=len(self.functionals), scope_ids=sorted(p for p in parents if p))
        if any(spec.get("site_parent") for spec in self.specs.values() if spec["id"] in self.functionals):
            seen = set()
            for root, model in self.models.items():
                for path, module in model.named_modules():
                    if id(module) in seen:
                        continue
                    seen.add(id(module))
                    target = root + ("."+path if path else "")
                    def before(mod, args, target=target):
                        ident = hashlib.sha256((target+"|"+_phase.get()).encode()).hexdigest()[:24]
                        binding.active_modules.set((*binding.active_modules.get(), ident))
                        entered = binding.scope_enabled and binding._invoking and ident in parents and binding._scope_depth == 0
                        if entered:
                            binding._mode.__enter__()
                            binding._scope_depth += 1
                        binding._scope_frames.append((id(mod), entered))
                    def after(mod, args, result):
                        # Another pre-hook can fail before ours is called.
                        if not binding._scope_frames or binding._scope_frames[-1][0] != id(mod):
                            return
                        _, entered = binding._scope_frames.pop()
                        if entered:
                            binding._scope_depth -= 1
                            binding._mode.__exit__(None, None, None)
                        binding.active_modules.set(binding.active_modules.get()[:-1])
                    self.handles.append(module.register_forward_pre_hook(before))
                    self.handles.append(module.register_forward_hook(after, always_call=True))
        class ReplacementMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                signature = None
                target, phase = str(func), _phase.get()
                stack = binding.active_modules.get()
                parent = stack[-1] if stack else hashlib.sha256(b"project.invoke|run").hexdigest()[:24]
                variants = indexed.get((target, phase, parent), ())
                global_variants = indexed.get((target, phase, None), ())
                for spec, replacement in (*variants, *global_variants):
                    if signature is None:
                        signature = {"args": tensor_description(args), "kwargs": tensor_description(kwargs or {})}
                    if signature in spec["signatures"]:
                        return replacement(*args, **(kwargs or {}))
                return func(*args, **(kwargs or {}))
        self._mode = ReplacementMode()

    def use_legacy(self, reason):
        """Preserve global mode ordering when several functional bundles coexist."""
        self.scope_enabled = False
        if self._mode is not None:
            self.execution_plan.update(effective="legacy", reason=reason)

    @contextmanager
    def activate(self):
        previous = self._invoking
        self._invoking = True
        try:
            if self._mode is not None and not self.scope_enabled:
                with self._mode:
                    yield
            else:
                yield
        finally:
            self._invoking = previous

    def invoke(self):
        with self.activate():
            return self.adapter.run(self.state)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for owner, attr, original in reversed(self.restores):
            setattr(owner, attr, original)
        self.restores.clear()
