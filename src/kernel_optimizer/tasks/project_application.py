"""Parameter-only callable loading and scoped artifact services, not an evaluator runtime."""

import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Final, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import ValidationError

from kernel_optimizer.models.candidate_artifact import CandidateArtifact, ParameterApplicationRequest
from kernel_optimizer.models.contract_common import CallableRef, FiniteJsonValue
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.project_content import (
    ProjectArtifactError, SnapshotDocument, load_candidate, materialize_project, store_snapshot, validate_artifact,
)


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Available only during a declared application: private working tree and artifact store."""

    root: Path
    store: RunStore
    artifact: CandidateArtifact


_CONTEXT: Final[ContextVar[ApplicationContext]] = ContextVar("project_application_context")


def application_context() -> ApplicationContext:
    """Callable API: retrieve the current invocation's root/store/input artifact; never cwd."""
    try:
        return _CONTEXT.get()
    except LookupError as exc:
        raise ProjectArtifactError("application_context", "outside parameter application") from exc


@runtime_checkable
class ParameterCallable(Protocol):
    def __call__(self, request: ParameterApplicationRequest) -> CandidateArtifact | FiniteJsonValue: ...


@contextmanager
def project_namespace(root: Path) -> Iterator[str]:
    """Unique root package keeps project-relative imports local through the entire call."""
    name = "_project_" + uuid4().hex
    package = ModuleType(name)
    package.__path__ = [str(root)]
    package.__package__ = name
    sys.modules[name] = package
    try:
        initializer = root / "__init__.py"
        if initializer.is_file():
            spec = importlib.util.spec_from_file_location(name, initializer, submodule_search_locations=[str(root)])
            if spec is None or spec.loader is None:
                raise ProjectArtifactError(str(initializer), "package loader unavailable")
            package = importlib.util.module_from_spec(spec)
            sys.modules[name] = package
            spec.loader.exec_module(package)
        yield name
    except (ImportError, SyntaxError) as exc:
        raise ProjectArtifactError(str(root), "project import failed") from exc
    finally:
        for module in tuple(sys.modules):
            if module == name or module.startswith(name + "."):
                del sys.modules[module]


def apply_parameters(
    store: RunStore, request: ParameterApplicationRequest, workspace: Path,
) -> CandidateArtifact:
    parsed_request = ParameterApplicationRequest.model_validate_json(request.model_dump_json())
    artifact = load_candidate(store, parsed_request.candidate_ref)
    workspace.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="application-", dir=workspace) as temporary:
        root = materialize_project(store, artifact, Path(temporary) / "project")
        token = _CONTEXT.set(ApplicationContext(root=root, store=store, artifact=artifact))
        try:
            with project_namespace(root) as namespace:
                ref = artifact.parameter_application
                try:
                    function = load_callable(namespace, ref)
                    result = function(ParameterApplicationRequest.model_validate_json(parsed_request.model_dump_json()))
                    if not isinstance(result, CandidateArtifact):
                        raise ProjectArtifactError(ref.callable, "must return CandidateArtifact")
                    result = CandidateArtifact.model_validate_json(result.model_dump_json())
                    validate_artifact(store, result)
                    if not {t.id for t in artifact.accepted_transforms} <= {t.id for t in result.accepted_transforms}:
                        raise ProjectArtifactError("accepted_transforms", "retain history with explicit supersession")
                    verified = materialize_project(store, result, Path(temporary) / "returned")
                    with project_namespace(verified) as returned_namespace:
                        _ = load_callable(returned_namespace, result.entrypoint)
                        _ = load_callable(returned_namespace, result.parameter_application)
                    bound_snapshot = store_snapshot(store, SnapshotDocument(files=result.files, application=parsed_request))
                    result = result.model_copy(update={"snapshot_ref": bound_snapshot})
                except (ImportError, AttributeError, SyntaxError, ValidationError) as exc:
                    raise ProjectArtifactError(ref.module, "application or returned artifact invalid") from exc
                _ = store.put_artifact(result.model_dump_json(), "parameterized-project")
                return result
        finally:
            _CONTEXT.reset(token)


def load_callable(namespace: str, ref: CallableRef) -> ParameterCallable:
    module = importlib.import_module(namespace + "." + ref.module)
    function = getattr(module, ref.callable, None)
    if not isinstance(function, ParameterCallable):
        raise ProjectArtifactError(ref.callable, "declared symbol is not callable")
    return function
