"""Verified content and project-relative manifest paths over the existing RunStore."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import ValidationError

from kernel_optimizer.models.candidate_artifact import CandidateArtifact, ParameterApplicationRequest
from kernel_optimizer.models.contract_common import CallableRef, ContractModel, FileRef
from kernel_optimizer.store.run_store import RunStore


class ProjectArtifactError(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        self.field: str = field
        self.reason: str = reason
        super().__init__(f"{field}: {reason}")


class SnapshotDocument(ContractModel):
    files: tuple[FileRef, ...]
    application: ParameterApplicationRequest | None = None


def store_snapshot(store: RunStore, document: SnapshotDocument) -> str:
    return store.put_artifact(json.dumps(document.model_dump(mode="json"), sort_keys=True,
                                         separators=(",", ":"), allow_nan=False), "project-snapshot")


def project_path(path: str) -> str:
    """Require canonical portable relative paths; reject aliases rather than merge them."""
    posix = PurePosixPath(path)
    if (not path or posix.is_absolute() or PureWindowsPath(path).drive
            or "\\" in path or ":" in path or "\x00" in path
            or any(part in ("", ".", "..") for part in path.split("/"))):
        raise ProjectArtifactError(path, "expected normalized project-relative path")
    return path


def read_content(store: RunStore, ref: str) -> bytes:
    """Resolve a store reference and verify its actual content, never original source."""
    if re.fullmatch(r"sha256:[0-9a-f]{64}", ref) is None:
        raise ProjectArtifactError(ref, "expected SHA256 content reference")
    try:
        content = store.get_artifact(ref)
    except OSError as exc:
        raise ProjectArtifactError(ref, "content unavailable") from exc
    if hashlib.sha256(content).hexdigest() != ref.removeprefix("sha256:"):
        raise ProjectArtifactError(ref, "content hash mismatch")
    return content


def callable_path(ref: CallableRef, paths: set[str]) -> str:
    if not all(part.isidentifier() for part in ref.module.split(".")) or not ref.callable.isidentifier():
        raise ProjectArtifactError(ref.module, "expected Python module and exported symbol")
    module = ref.module.replace(".", "/")
    for path in (module + ".py", module + "/__init__.py"):
        if path in paths:
            return path
    raise ProjectArtifactError(ref.module, "declared execution module missing from snapshot")


def validate_artifact(store: RunStore, artifact: CandidateArtifact) -> None:
    """Check model-copy bypasses, manifest availability and every FileRef hash."""
    try:
        parsed = CandidateArtifact.model_validate_json(artifact.model_dump_json())
        document = SnapshotDocument.model_validate_json(read_content(store, parsed.snapshot_ref))
    except ValidationError as exc:
        raise ProjectArtifactError("manifest", "invalid artifact or snapshot document") from exc
    if parsed.artifact_kind != "project_snapshot":
        raise ProjectArtifactError("artifact_kind", "project_snapshot required")
    files = document.files
    if files != parsed.files:
        raise ProjectArtifactError("files", "snapshot document differs from complete manifest")
    paths = {project_path(file.path) for file in files}
    if len({path.casefold() for path in paths}) != len(paths):
        raise ProjectArtifactError("files", "case-aliased paths")
    for path in paths:
        if any(str(parent) in paths for parent in PurePosixPath(path).parents):
            raise ProjectArtifactError(path, "file also declared as a directory")
    for change in parsed.changes:
        _ = project_path(change.path)
    _ = callable_path(parsed.entrypoint, paths)
    _ = callable_path(parsed.parameter_application, paths)
    for file in files:
        content = read_content(store, file.content_ref)
        if hashlib.sha256(content).hexdigest() != file.sha256:
            raise ProjectArtifactError(file.path, "FileRef hash mismatch")


def load_candidate(store: RunStore, ref: str) -> CandidateArtifact:
    try:
        artifact = CandidateArtifact.model_validate_json(read_content(store, ref))
    except ValidationError as exc:
        raise ProjectArtifactError(ref, "invalid CandidateArtifact") from exc
    validate_artifact(store, artifact)
    return artifact


def materialize_project(store: RunStore, artifact: CandidateArtifact, destination: Path) -> Path:
    """Write only to a new directory, rejecting stale files instead of silently reusing them."""
    validate_artifact(store, artifact)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ProjectArtifactError(str(destination), "destination must be a new directory") from exc
    for file in artifact.files:
        target = destination / file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        _ = target.write_bytes(read_content(store, file.content_ref))
    return destination
