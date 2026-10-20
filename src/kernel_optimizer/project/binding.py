"""Candidate callables are installed at observed sites, with actual-call coverage."""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import hashlib
import importlib.util
from pathlib import Path

from kernel_optimizer.project.discovery import _phase, tensor_description


def signature_guard(signatures):
    """Compile immutable observed metadata, rechecking live tensors on every call.

    No tensor-id cache: in-place metadata changes and reused Python identities
    must still fail the original shape/stride/dtype/phase binding contract.
    """
    import torch

    def compile_value(expected):
        if isinstance(expected, dict):
            if set(expected) == {"shape", "stride", "dtype", "device", "requires_grad"}:
                shape, stride = tuple(expected["shape"]), tuple(expected["stride"])
                def tensor(value):
                    if isinstance(value, torch.Tensor):
                        return (tuple(value.shape) == shape and tuple(value.stride()) == stride
                                and str(value.dtype) == expected["dtype"]
                                and value.device.type == expected["device"]
                                and value.requires_grad == expected["requires_grad"])
                    return tensor_description(value) == expected
                return tensor
            children = {key: compile_value(value) for key, value in expected.items()}
            def mapping(value):
                if not isinstance(value, dict):
                    return tensor_description(value) == expected
                # Discovery stringifies dictionary keys; retain that behavior.
                items = value if all(isinstance(key, str) for key in value) else {str(k): v for k, v in value.items()}
                return items.keys() == children.keys() and all(check(items[key]) for key, check in children.items())
            return mapping
        if isinstance(expected, list):
            children = tuple(compile_value(value) for value in expected)
            return lambda value: (isinstance(value, (tuple, list)) and len(value) == len(children)
                                  and all(check(item) for check, item in zip(children, value)))
        def scalar(value):
            if type(value) in (bool, int, float, str, type(None)):
                return value == expected
            return tensor_description(value) == expected
        return scalar

    variants = [(compile_value(s["args"]), compile_value(s["kwargs"])) for s in signatures]
    return lambda args, kwargs: any(a(args) and k(kwargs) for a, k in variants)


class BindingSession:
    def __init__(self, adapter, state, specs: list[dict], *, rank=0, world_size=1, bundle_id=None):
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
            guarded = [(spec, replacement, signature_guard(spec["signatures"])) for spec, replacement in variants]
            def dispatch(*args, _variants=guarded, _original=original, **kwargs):
                for spec, replacement, matches in _variants:
                    if spec["phase"] != _phase.get():
                        continue
                    if spec["kind"] == "module":
                        if not matches(args, kwargs):
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
            operator = self.references[ident]
            key = (spec["phase"], spec.get("site_parent"))
            indexed.setdefault(operator, {}).setdefault(key, []).append((signature_guard(spec["signatures"]), replacement))
        root_parent = hashlib.sha256(b"project.invoke|run").hexdigest()[:24]
        if any(spec.get("site_parent") for spec in self.specs.values() if spec["id"] in self.functionals):
            seen = set()
            for root, model in self.models.items():
                for path, module in model.named_modules():
                    if id(module) in seen:
                        continue
                    seen.add(id(module))
                    target = root + ("."+path if path else "")
                    phase_ids = {}
                    def before(mod, args, target=target, phase_ids=phase_ids):
                        phase = _phase.get()
                        ident = phase_ids.get(phase)
                        if ident is None:
                            ident = phase_ids[phase] = hashlib.sha256((target+"|"+phase).encode()).hexdigest()[:24]
                        binding.active_modules.set((*binding.active_modules.get(), ident))
                    def after(mod, args, result):
                        binding.active_modules.set(binding.active_modules.get()[:-1])
                    self.handles.append(module.register_forward_pre_hook(before))
                    self.handles.append(module.register_forward_hook(after, always_call=True))
        class ReplacementMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                by_context = indexed.get(func)
                if by_context is None:
                    return func(*args, **(kwargs or {}))
                phase = _phase.get()
                stack = binding.active_modules.get()
                parent = stack[-1] if stack else root_parent
                for variants in (by_context.get((phase, parent), ()), by_context.get((phase, None), ())):
                    for matches, replacement in variants:
                        if matches(args, kwargs or {}):
                            return replacement(*args, **(kwargs or {}))
                return func(*args, **(kwargs or {}))
        self._mode = ReplacementMode()

    def invoke(self):
        if self._mode is None:
            return self.adapter.run(self.state)
        with self._mode:
            return self.adapter.run(self.state)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for owner, attr, original in reversed(self.restores):
            setattr(owner, attr, original)
        self.restores.clear()
