"""Candidate callables are installed at observed sites, with actual-call coverage."""
from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
import hashlib
import importlib.util
from pathlib import Path

from kernel_optimizer.project.discovery import _phase, tensor_description


class BindingSession:
    def __init__(self, adapter, state, specs: list[dict]):
        self.adapter, self.state = adapter, state
        self.specs = {spec["id"]: spec for spec in specs}
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
                                     "references": self.references, "targets": self.specs, "parameters": parameters})
        if not isinstance(replacements, dict) or not replacements:
            raise ValueError("build(context) must return a nonempty call_id -> callable mapping")
        if set(replacements) - set(self.specs):
            raise ValueError("candidate returned an undeclared replacement site")
        physical = set()
        for ident, function in replacements.items():
            if not callable(function):
                raise TypeError("replacement must be callable")
            spec = self.specs[ident]
            def wrapped(*args, _function=function, _ident=ident, _spec=spec, **kwargs):
                if _spec["kind"] == "module":
                    signature = {"args": tensor_description(args), "kwargs": tensor_description(kwargs)}
                    if signature not in _spec["signatures"]:
                        if self.tracking:
                            self.fallbacks[_ident] += 1
                        return self.references[_ident](*args, **kwargs)
                if self.tracking:
                    self.counts[_ident] += 1
                return _function(*args, **kwargs)
            if spec["kind"] == "functional":
                self.functionals[ident] = wrapped
            else:
                owner = self.modules[ident] if spec["kind"] == "module" else self.adapter
                attr = "forward" if spec["kind"] == "module" else "run"
                key = (id(owner), attr)
                if key in physical:
                    raise ValueError("multiple replacements for the same physical site require a composed guard")
                physical.add(key)
                self.restores.append((owner, attr, getattr(owner, attr)))
                setattr(owner, attr, wrapped)
        if self.functionals:
            self._create_dispatch_mode()
        return list(replacements)

    def _create_dispatch_mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        binding = self
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
                for ident, replacement in binding.functionals.items():
                    spec = binding.specs[ident]
                    if spec["target"] != str(func) or spec["phase"] != _phase.get():
                        continue
                    stack = binding.active_modules.get()
                    parent = stack[-1] if stack else hashlib.sha256(b"project.invoke|run").hexdigest()[:24]
                    if spec.get("site_parent") is not None and spec["site_parent"] != parent:
                        continue
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
