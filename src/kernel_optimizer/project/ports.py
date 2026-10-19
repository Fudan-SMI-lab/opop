"""Small replaceable interfaces; project-specific semantics belong in adapters."""
from __future__ import annotations

from typing import Any, Protocol
from kernel_optimizer.models.project import CapabilityManifest, CandidateBundle, ProjectContract


class ProjectAdapter(Protocol):
    def capabilities(self) -> CapabilityManifest: ...
    def prepare(self, contract: ProjectContract, context: dict) -> Any: ...
    def reset(self, state: Any) -> None: ...
    def run(self, state: Any, workload: dict) -> Any: ...
    def bind(self, state: Any, bundle: CandidateBundle) -> Any: ...
    def restore(self, state: Any, binding: Any) -> None: ...
    def capture(self, state: Any, workload: dict) -> dict: ...


class Collector(Protocol):
    def probe(self, context: dict) -> CapabilityManifest: ...
    def collect(self, operation: Any, context: dict) -> dict: ...
    def normalize(self, raw: dict, context: dict) -> dict: ...
    def validate(self, observation: dict) -> tuple[str, ...]: ...


class Registry:
    def __init__(self):
        self._providers: dict[str, Any] = {}

    def register(self, name: str, provider: Any) -> None:
        if name in self._providers:
            raise ValueError(f"provider already registered: {name}")
        self._providers[name] = provider

    def resolve(self, name: str) -> Any:
        try:
            return self._providers[name]
        except KeyError:
            raise ValueError(f"unsupported provider {name}; available: {sorted(self._providers)}") from None
