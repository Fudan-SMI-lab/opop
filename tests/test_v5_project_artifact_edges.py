"""Artifact completeness, imports, byte fidelity and transformation regressions."""

import importlib
from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

import pytest
from kernel_optimizer.models.candidate_artifact import AcceptedTransform, ParameterApplicationRequest
from kernel_optimizer.models.contract_common import CallableRef
from kernel_optimizer.paramspace.project_materializer import ProjectMaterializer
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks.project_application import application_context, project_namespace
from kernel_optimizer.tasks.project_artifacts import ProjectSpec, materialize_project, snapshot_project
from kernel_optimizer.tasks.project_content import ProjectArtifactError, SnapshotDocument, store_snapshot


def project(tmp_path: Path, source: str) -> tuple[RunStore, Path, ProjectSpec]:
    store = RunStore.create(tmp_path, "run", {})
    root = tmp_path / "source"
    root.mkdir()
    _ = (root / "pipeline.py").write_text(source, encoding="utf-8")
    spec = ProjectSpec(paths=("pipeline.py",), backend_type="novel-cpu-capability",
                       entrypoint=CallableRef(module="pipeline", callable="run"),
                       parameter_application=CallableRef(module="pipeline", callable="apply_parameters"))
    return store, root, spec


GOOD = "def run():\n    return 2\ndef apply_parameters(request):\n    return 42\n"


@runtime_checkable
class NumericCallable(Protocol):
    def __call__(self) -> int: ...


def run_value(module: ModuleType) -> int:
    function = getattr(module, "run", None)
    assert isinstance(function, NumericCallable)
    return function()


@pytest.mark.parametrize("path", ["../escape.py", "/abs.py", "a//b.py", "./pipeline.py", "a/../pipeline.py", "a\\b.py", "C:/file.py", ""])
def test_reject_invalid_declared_path(tmp_path: Path, path: str) -> None:
    # Given
    store, root, spec = project(tmp_path, GOOD)
    bad = ProjectSpec(paths=("pipeline.py", path), backend_type=spec.backend_type,
                      entrypoint=spec.entrypoint, parameter_application=spec.parameter_application)
    # When / Then
    with pytest.raises(ProjectArtifactError):
        _ = snapshot_project(store, root, bad)


@pytest.mark.parametrize("paths", [("pipeline.py", "pipeline.py"), ("missing.py",), ("pipeline.py", "missing.dat")])
def test_reject_missing_or_duplicate_manifest(tmp_path: Path, paths: tuple[str, ...]) -> None:
    # Given
    store, root, spec = project(tmp_path, GOOD)
    # When / Then
    with pytest.raises(ProjectArtifactError):
        _ = snapshot_project(store, root, ProjectSpec(paths=paths, backend_type=spec.backend_type,
                         entrypoint=spec.entrypoint, parameter_application=spec.parameter_application))


def test_binary_resources_and_crlf_survive_root_rename(tmp_path: Path) -> None:
    # Given
    store, root, spec = project(tmp_path, GOOD)
    content = bytes(range(256)) + b"\r\n\x00"
    _ = (root / "asset.bin").write_bytes(content)
    spec = ProjectSpec(paths=("pipeline.py", "asset.bin"), backend_type=spec.backend_type,
                       entrypoint=spec.entrypoint, parameter_application=spec.parameter_application)
    artifact = snapshot_project(store, root, spec)
    renamed = root.rename(tmp_path / "renamed")
    same = snapshot_project(store, renamed, spec)
    _ = (renamed / "asset.bin").write_bytes(b"dirty source")
    # When
    _ = materialize_project(store, artifact, tmp_path / "out")
    # Then
    assert artifact == same
    assert (tmp_path / "out" / "asset.bin").read_bytes() == content


def test_dirty_destination_rejected_without_overwrite(tmp_path: Path) -> None:
    # Given
    store, root, spec = project(tmp_path, GOOD)
    artifact = snapshot_project(store, root, spec)
    # When / Then
    with pytest.raises(ProjectArtifactError):
        _ = materialize_project(store, artifact, root)
    assert (root / "pipeline.py").read_text(encoding="utf-8") == GOOD


@pytest.mark.parametrize("fault", ["file-sha", "snapshot-missing", "snapshot-mismatch", "deleted-import", "incomplete-return"])
def test_missing_file_or_bad_application_rejected(tmp_path: Path, fault: str) -> None:
    # Given
    source = '''
from kernel_optimizer.tasks.project_application import application_context
def run():
    return 2
def apply_parameters(request):
    artifact = application_context().artifact
    return artifact.model_copy(update={"files": ()})
'''
    store, root, spec = project(tmp_path, source)
    if fault == "deleted-import":
        _ = (root / "pipeline.py").write_text("from .deleted import VALUE\n" + source, encoding="utf-8")
    artifact = snapshot_project(store, root, spec)
    if fault == "file-sha":
        file = artifact.files[0].model_copy(update={"sha256": "0" * 64})
        index = store_snapshot(store, SnapshotDocument(files=(file,)))
        artifact = artifact.model_copy(update={"files": (file,), "snapshot_ref": index})
    if fault == "snapshot-missing":
        artifact = artifact.model_copy(update={"snapshot_ref": "sha256:" + "0" * 64})
    if fault == "snapshot-mismatch":
        artifact = artifact.model_copy(update={"snapshot_ref": store.put_artifact("[]")})
    ref = store.put_artifact(artifact.model_dump_json())
    # When / Then
    with pytest.raises(ProjectArtifactError):
        _ = ProjectMaterializer(store, tmp_path / "work").materialize(
            ParameterApplicationRequest(candidate_ref=ref, params={}))


def test_application_context_has_no_stale_invocation() -> None:
    # Given / When / Then
    with pytest.raises(ProjectArtifactError):
        _ = application_context()


def test_multiple_trees_relative_imports_and_composed_transforms(tmp_path: Path) -> None:
    # Given
    store, root, spec = project(tmp_path, "from .helper import VALUE\ndef run():\n    return (VALUE + 3) * 2\n")
    _ = (root / "helper.py").write_text("VALUE = 4\n", encoding="utf-8")
    transforms = (AcceptedTransform(id="add", status="active", validation_ref="validation:add"),
                  AcceptedTransform(id="multiply", status="active", validation_ref="validation:multiply"))
    spec = ProjectSpec(paths=("pipeline.py", "helper.py"), backend_type=spec.backend_type,
                       entrypoint=spec.entrypoint, parameter_application=spec.entrypoint,
                       accepted_transforms=transforms)
    first = snapshot_project(store, root, spec)
    _ = (root / "helper.py").write_text("VALUE = 10\n", encoding="utf-8")
    second = snapshot_project(store, root, spec)
    a = materialize_project(store, first, tmp_path / "a")
    b = materialize_project(store, second, tmp_path / "b")
    # When
    with project_namespace(a) as na, project_namespace(b) as nb:
        ma = importlib.import_module(na + ".pipeline")
        mb = importlib.import_module(nb + ".pipeline")
        outputs = (run_value(ma), run_value(mb), run_value(ma))
    # Then
    assert outputs == (14, 26, 14)
    assert first.accepted_transforms == transforms


def test_legal_supersession_does_not_require_old_implementation(tmp_path: Path) -> None:
    # Given
    store, root, spec = project(tmp_path, GOOD)
    old = AcceptedTransform(id="old", status="active", validation_ref="validation:old")
    parent = snapshot_project(store, root, ProjectSpec(paths=spec.paths, backend_type=spec.backend_type,
        entrypoint=spec.entrypoint, parameter_application=spec.parameter_application, accepted_transforms=(old,)))
    _ = (root / "pipeline.py").write_text("def run():\n    return 99\n", encoding="utf-8")
    transforms = (AcceptedTransform(id="old", status="superseded", superseded_by="new", validation_ref="validation:old"),
                  AcceptedTransform(id="new", status="active", validation_ref="validation:new"))
    # When
    result = snapshot_project(store, root, ProjectSpec(paths=spec.paths, backend_type=spec.backend_type,
        entrypoint=spec.entrypoint, parameter_application=spec.entrypoint, parent=parent, accepted_transforms=transforms))
    output = materialize_project(store, result, tmp_path / "out")
    with project_namespace(output) as name:
        value = run_value(importlib.import_module(name + ".pipeline"))
    # Then
    assert value == 99
    assert result.accepted_transforms == transforms


def test_application_cannot_return_broken_import_tree(tmp_path: Path) -> None:
    # Given
    source = '''
from kernel_optimizer.tasks.project_application import application_context
from kernel_optimizer.tasks.project_artifacts import ProjectSpec, snapshot_project
def run():
    return 2
def apply_parameters(request):
    c = application_context()
    (c.root / "pipeline.py").write_text("from .missing import VALUE\\ndef run(): return VALUE\\n")
    return snapshot_project(c.store, c.root, ProjectSpec(paths=("pipeline.py",), backend_type="cpu",
        entrypoint=c.artifact.entrypoint, parameter_application=c.artifact.entrypoint, parent=c.artifact))
'''
    store, root, spec = project(tmp_path, source)
    artifact = snapshot_project(store, root, spec)
    ref = store.put_artifact(artifact.model_dump_json())
    # When / Then
    with pytest.raises(ProjectArtifactError):
        _ = ProjectMaterializer(store, tmp_path / "work").materialize(ParameterApplicationRequest(candidate_ref=ref, params={}))
