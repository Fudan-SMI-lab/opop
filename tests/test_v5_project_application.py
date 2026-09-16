"""Parameter invocation lifecycle and explicit transform history."""

import importlib
import sys
from pathlib import Path

import pytest

from kernel_optimizer.models.candidate_artifact import AcceptedTransform, ParameterApplicationRequest
from kernel_optimizer.models.contract_common import CallableRef
from kernel_optimizer.paramspace.project_materializer import ProjectMaterializer
from kernel_optimizer.tasks.project_application import project_namespace
from kernel_optimizer.tasks.project_artifacts import ProjectSpec, materialize_project, snapshot_project
from kernel_optimizer.tasks.project_content import ProjectArtifactError, SnapshotDocument
from tests.test_v5_project_artifact_edges import project, run_value
from tests.test_v5_project_artifacts import APPLICATION


def test_root_package_initialization_and_resource(tmp_path: Path) -> None:
    # Given
    store, root, spec = project(tmp_path, "from . import VALUE\ndef run():\n    return VALUE\n")
    _ = (root / "__init__.py").write_text(
        "from pathlib import Path\nVALUE = int(Path(__file__).with_name('value.txt').read_text())\n", encoding="utf-8")
    _ = (root / "value.txt").write_text("17", encoding="utf-8")
    artifact = snapshot_project(store, root, ProjectSpec(paths=("__init__.py", "pipeline.py", "value.txt"),
        backend_type=spec.backend_type, entrypoint=spec.entrypoint, parameter_application=spec.entrypoint))
    output = materialize_project(store, artifact, tmp_path / "out")
    # When
    with project_namespace(output) as name:
        value = run_value(importlib.import_module(name + ".pipeline"))
    # Then
    assert value == 17
    assert name not in sys.modules


def test_application_preserves_explicit_transform_history(tmp_path: Path) -> None:
    # Given
    source = '''
from kernel_optimizer.tasks.project_application import application_context
def run(): return 3
def apply_parameters(request):
    return application_context().artifact.model_copy(update={"accepted_transforms": ()})
'''
    store, root, spec = project(tmp_path, source)
    parent = snapshot_project(store, root, ProjectSpec(paths=spec.paths, backend_type=spec.backend_type,
        entrypoint=spec.entrypoint, parameter_application=spec.parameter_application,
        accepted_transforms=(AcceptedTransform(id="kept", status="active", validation_ref="v:kept"),)))
    ref = store.put_artifact(parent.model_dump_json())
    # When / Then
    with pytest.raises(ProjectArtifactError):
        _ = ProjectMaterializer(store, tmp_path / "work").materialize(ParameterApplicationRequest(candidate_ref=ref, params={}))


def test_parameter_application_uses_each_trees_relative_imports(tmp_path: Path) -> None:
    # Given
    source = "from .seed import SEED\nfrom .helper import VALUE\ndef run(): return VALUE\n" + APPLICATION.replace(
        'repr(request.params["value"])', 'repr(request.params["value"] + SEED)')
    store, root, spec = project(tmp_path, source)
    _ = (root / "seed.py").write_text("SEED = 3\n", encoding="utf-8")
    _ = (root / "helper.py").write_text("VALUE = 0\n", encoding="utf-8")
    spec = ProjectSpec(paths=("pipeline.py", "seed.py", "helper.py"), backend_type=spec.backend_type,
                       entrypoint=spec.entrypoint, parameter_application=spec.parameter_application)
    first = snapshot_project(store, root, spec)
    _ = (root / "seed.py").write_text("SEED = 8\n", encoding="utf-8")
    second = snapshot_project(store, root, spec)
    refs = tuple(store.put_artifact(a.model_dump_json()) for a in (first, second, first))
    materializer = ProjectMaterializer(store, tmp_path / "work")
    # When
    values: list[int] = []
    for index, ref in enumerate(refs):
        artifact = materializer.materialize(ParameterApplicationRequest(candidate_ref=ref, params={"value": 10}))
        output = materialize_project(store, artifact, tmp_path / str(index))
        with project_namespace(output) as name:
            values.append(run_value(importlib.import_module(name + ".pipeline")))
    # Then
    assert values == [13, 18, 13]
    assert list((tmp_path / "work").iterdir()) == []


@pytest.mark.parametrize("symbol", ["absent", "value"])
def test_application_symbol_must_be_callable(tmp_path: Path, symbol: str) -> None:
    # Given
    store, root, spec = project(tmp_path, "value = 4\ndef run(): return 2\n")
    artifact = snapshot_project(store, root, ProjectSpec(paths=spec.paths, backend_type=spec.backend_type,
        entrypoint=spec.entrypoint, parameter_application=CallableRef(module="pipeline", callable=symbol)))
    ref = store.put_artifact(artifact.model_dump_json())
    # When / Then
    with pytest.raises(ProjectArtifactError):
        _ = ProjectMaterializer(store, tmp_path / "work").materialize(ParameterApplicationRequest(candidate_ref=ref, params={}))


def test_parameter_values_remain_bound_when_files_are_unchanged(tmp_path: Path) -> None:
    # Given
    source = '''
from kernel_optimizer.tasks.project_application import application_context
def run(): return 2
def apply_parameters(request): return application_context().artifact
'''
    store, root, spec = project(tmp_path, source)
    artifact = snapshot_project(store, root, spec)
    ref = store.put_artifact(artifact.model_dump_json())
    materializer = ProjectMaterializer(store, tmp_path / "work")
    # When
    first = materializer.materialize(ParameterApplicationRequest(candidate_ref=ref, params={"mode": "a"}))
    second = materializer.materialize(ParameterApplicationRequest(candidate_ref=ref, params={"mode": "b"}))
    # Then
    assert first.files == second.files
    assert first.snapshot_ref != second.snapshot_ref


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("mutates", [False, True], ids=["unchanged", "mutated"])
def test_requested_provenance_survives_local_parameter_mutation(
    tmp_path: Path, nested: bool, mutates: bool,
) -> None:
    # Given
    target = 'request.params["config"]["value"]' if nested else 'request.params["value"]'
    operation = f"{target} = 999" if mutates else "pass"
    source = (
        "from kernel_optimizer.tasks.project_application import application_context\n"
        "def run(): return 7\n"
        "def apply_parameters(request):\n"
        f"    {operation}\n"
        "    print('CALLBACK_LOCAL', request.model_dump_json())\n"
        "    return application_context().artifact\n"
    )
    store, root, spec = project(tmp_path, source)
    artifact = snapshot_project(store, root, spec)
    ref = store.put_artifact(artifact.model_dump_json())
    requests = tuple(ParameterApplicationRequest(candidate_ref=ref,
        params={"config": {"value": value}} if nested else {"value": value}) for value in (11, 22))
    materializer = ProjectMaterializer(store, tmp_path / "work")
    # When
    results = tuple(materializer.materialize(request) for request in requests)
    # Then
    documents = tuple(SnapshotDocument.model_validate_json(store.get_artifact(result.snapshot_ref)) for result in results)
    candidate_refs = tuple(store.put_artifact(result.model_dump_json()) for result in results)
    print("SNAPSHOT_REFS", tuple(result.snapshot_ref for result in results))
    print("CANDIDATE_REFS", candidate_refs)
    for requested_value, request, result, document in zip((11, 22), requests, results, documents, strict=True):
        expected = {"config": {"value": requested_value}} if nested else {"value": requested_value}
        print("REQUESTED", requested_value, "RECORDED", document.model_dump_json())
        assert request.params == expected
        assert document.application is not None
        assert document.application.candidate_ref == ref
        assert document.application.params == expected
        assert result.files == artifact.files
    assert results[0].snapshot_ref != results[1].snapshot_ref
    assert candidate_refs[0] != candidate_refs[1]
