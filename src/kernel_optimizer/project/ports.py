"""Small replaceable interfaces; project-specific semantics belong in adapters."""
from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from kernel_optimizer.models.project import ProjectRevision
    from kernel_optimizer.project.search import ProjectSearch


class ProjectAdapter(Protocol):
    def prepare(self, context: dict, request: dict) -> Any: ...
    def reset(self, state: Any) -> None: ...
    def run(self, state: Any) -> Any: ...
    def observe(self, state: Any, result: Any) -> Any: ...
    def metrics(self, state: Any, result: Any, elapsed_ms: float) -> dict[str, float]: ...
    def models(self, state: Any) -> dict[str, Any]: ...


class CandidateInstaller(Protocol):
    def install(self, file: Any, parameters: dict) -> list[str]: ...
    def invoke(self) -> Any: ...
    def close(self) -> None: ...


class Collector(Protocol):
    """A separate diagnostic job; it never supplies clean objective samples.

    Probe/normalization/validation are owned by each backend. The coordinator's
    actual plugin boundary is this method, shared by Nsys, telemetry and calibration.
    """
    def collect(self, search: ProjectSearch, revision: ProjectRevision) -> dict: ...


class MetricCollector(Protocol):
    """Explicitly declared diagnostic objective sampling inside a project worker."""
    def metric(self, adapter: ProjectAdapter, state: Any, operation: Any,
               synchronize: Any, context: dict, source: dict, repeats: int) -> dict: ...


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

    def items(self):
        return self._providers.items()
