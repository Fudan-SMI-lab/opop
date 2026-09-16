"""Full declared project snapshots; legacy PARAMS materialization is independent."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from kernel_optimizer.models.candidate_artifact import (
    AcceptedTransform, CandidateArtifact, FileChange, ParameterApplicationRequest,
)
from kernel_optimizer.models.contract_common import CallableRef, FileRef
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.project_content import (
    SnapshotDocument, callable_path, project_path, store_snapshot, validate_artifact,
)
from kernel_optimizer.tasks.project_content import ProjectArtifactError as ProjectArtifactError
from kernel_optimizer.tasks.project_content import materialize_project as materialize_project


@dataclass(frozen=True, slots=True)
class ProjectSpec:
    """Execution manifest for all required files; parent enables derived change metadata."""

    paths: tuple[str, ...]
    backend_type: str
    entrypoint: CallableRef
    parameter_application: CallableRef
    parent: CandidateArtifact | None = None
    accepted_transforms: tuple[AcceptedTransform, ...] | None = None


def snapshot_project(store: RunStore, root: Path, spec: ProjectSpec) -> CandidateArtifact:
    """Capture exact bytes, derive changes, and persist a complete independent file index."""
    paths = tuple(sorted(project_path(path) for path in spec.paths))
    if not paths or len(set(paths)) != len(paths):
        raise ProjectArtifactError("paths", "nonempty unique required files expected")
    _ = callable_path(spec.entrypoint, set(paths))
    _ = callable_path(spec.parameter_application, set(paths))
    files: list[FileRef] = []
    for path in paths:
        try:
            content = (root / path).read_bytes()
        except OSError as exc:
            raise ProjectArtifactError(path, "required source file unavailable") from exc
        files.append(FileRef(path=path, content_ref=store.put_artifact(content, path),
                             sha256=hashlib.sha256(content).hexdigest()))
    parent_ref: str | None = None
    previous: dict[str, str] = {}
    transforms = spec.accepted_transforms
    if spec.parent is not None:
        validate_artifact(store, spec.parent)
        parent_ref = store.put_artifact(spec.parent.model_dump_json(), "parent-project")
        previous = {file.path: file.sha256 for file in spec.parent.files}
        if transforms is None:
            transforms = spec.parent.accepted_transforms
        if not {t.id for t in spec.parent.accepted_transforms} <= {t.id for t in transforms}:
            raise ProjectArtifactError("accepted_transforms", "retain history with explicit supersession")
    changes = tuple(
        FileChange(operation="replace" if file.path in previous else "add", path=file.path)
        for file in files if previous.get(file.path) != file.sha256
    ) + tuple(FileChange(operation="delete", path=path) for path in sorted(previous.keys() - set(paths)))
    snapshot_ref = store_snapshot(store, SnapshotDocument(files=tuple(files)))
    artifact = CandidateArtifact(
        artifact_kind="project_snapshot", backend_type=spec.backend_type,
        snapshot_ref=snapshot_ref, files=tuple(files), parent_ref=parent_ref, changes=changes,
        entrypoint=spec.entrypoint, parameter_application=spec.parameter_application,
        accepted_transforms=transforms or (),
    )
    validate_artifact(store, artifact)
    _ = store.put_artifact(artifact.model_dump_json(), "candidate-project")
    return artifact


def apply_project_parameters(
    store: RunStore, request: ParameterApplicationRequest, workspace: Path,
) -> CandidateArtifact:
    """Invoke the declared application callable with a scoped, documented project context."""
    from kernel_optimizer.tasks.project_application import apply_parameters

    return apply_parameters(store, request, workspace)
