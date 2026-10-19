"""Candidate callables are installed at observed sites, with actual-call coverage."""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import hashlib
import importlib.util
from pathlib import Path

from kernel_optimizer.project.discovery import _phase, tensor_description


class BindingSession:
    def __init__(self, adapter, state, specs: list[dict], *, rank=0, world_size=1):
        self.adapter, self.state = adapter, state
        self.rank, self.world_size = rank, world_size
        self.all_specs = {spec["id"]: spec for spec in specs}
        self.specs = {spec["id"]: {**spec, **spec.get("rank_specs", {}).get(str(rank), {})}
                      for spec in specs if rank in spec.get("ranks", range(world_size))}
        self.models = adapter.models(state)
        self.references, self.modules = {}, {}
        self.counts = Counter()
        self.fallbacks = Counter()
        self.tracking = True
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

    def install(self, file: Path, parameters: dict) -> list[str]:
        loader = importlib.util.spec_from_file_location("opop_candidate_" + str(len(self.restores)), file)
        module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
        replacements = module.build({"state": self.state, "models": self.models, "modules": self.modules,
                                     "references": self.references, "targets": self.specs, "parameters": parameters,
                                     "rank": self.rank, "world_size": self.world_size, "all_targets": self.all_specs})
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
                    def after(mod, args, result):
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
