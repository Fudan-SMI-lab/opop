"""Real-store project snapshot and application behavior."""

from pathlib import Path
import subprocess
import sys

import pytest

from kernel_optimizer.models.candidate_artifact import ParameterApplicationRequest
from kernel_optimizer.models.contract_common import CallableRef

from kernel_optimizer.models.core import ParamSet
from kernel_optimizer.paramspace.materializer import extract_defaults, materialize
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.tasks import project_artifacts as api


def test_existing_store_text_roundtrip(tmp_path: Path) -> None:
    # Given
    store = RunStore.create(tmp_path, "run", {})
    text = "hello\nUnicode: \u03bb\n"
    # When
    restored = store.get_artifact(store.put_artifact(text))
    # Then
    assert restored == text.encode("utf-8")


def test_existing_legacy_params_behavior() -> None:
    # Given
    source = "# prefix\nPARAMS = {'amount': 2}\ndef run():\n    return PARAMS['amount']\n"
    # When
    result = materialize(source, ParamSet(values={"amount": 7}))
    # Then
    assert extract_defaults(result) == {"amount": 7}
    assert result.startswith("# prefix\nPARAMS = ")
    assert result.endswith("\ndef run():\n    return PARAMS['amount']\n")


def test_snapshot_add_replace_delete_and_apply(tmp_path: Path) -> None:
    # Given
    store = RunStore.create(tmp_path, "run", {})
    root = tmp_path / "original"
    root.mkdir()
    _ = (root / "pipeline.py").write_text("def run():\n    return 2\n", encoding="utf-8")
    _ = (root / "obsolete.py").write_text("OLD = 1\n", encoding="utf-8")
    spec = api.ProjectSpec(paths=("pipeline.py", "obsolete.py"), backend_type="custom-cpu",
                           entrypoint=CallableRef(module="pipeline", callable="run"),
                           parameter_application=CallableRef(module="pipeline", callable="run"))
    original = api.snapshot_project(store, root, spec)
    _ = (root / "pipeline.py").write_text(
        "from .helper import VALUE\ndef run():\n    return VALUE + 3\n" + APPLICATION,
        encoding="utf-8",
    )
    _ = (root / "helper.py").write_text("VALUE = 4\n", encoding="utf-8")
    (root / "obsolete.py").unlink()
    changed = api.snapshot_project(store, root, api.ProjectSpec(
        paths=("pipeline.py", "helper.py"), backend_type="custom-cpu",
        entrypoint=spec.entrypoint,
        parameter_application=CallableRef(module="pipeline", callable="apply_parameters"),
        parent=original,
    ))
    ref = store.put_artifact(changed.model_dump_json())
    _ = root.rename(tmp_path / "renamed")
    # When
    result = api.apply_project_parameters(store, ParameterApplicationRequest(
        candidate_ref=ref, params={"value": 11}), tmp_path / "application")
    _ = api.materialize_project(store, result, tmp_path / "output")
    child = subprocess.run([sys.executable, "-B", "-c",
                            "from output.pipeline import run; print(run())"],
                           cwd=tmp_path, capture_output=True, text=True, check=False)
    # Then
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "14"
    assert {(c.operation, c.path) for c in changed.changes} == {
        ("replace", "pipeline.py"), ("add", "helper.py"), ("delete", "obsolete.py")}
    assert not (tmp_path / "output" / "obsolete.py").exists()


APPLICATION = '''
from kernel_optimizer.tasks.project_application import application_context
from kernel_optimizer.tasks.project_artifacts import ProjectSpec, snapshot_project
def apply_parameters(request):
    context = application_context()
    (context.root / "helper.py").write_text("VALUE = " + repr(request.params["value"]) + "\\n")
    return snapshot_project(context.store, context.root, ProjectSpec(
        paths=tuple(f.path for f in context.artifact.files),
        backend_type=context.artifact.backend_type,
        entrypoint=context.artifact.entrypoint,
        parameter_application=context.artifact.parameter_application,
        parent=context.artifact,
        accepted_transforms=context.artifact.accepted_transforms,
    ))
'''


@pytest.mark.parametrize("fault", ["missing", "corrupt", "bad-return"])
def test_missing_file_or_bad_application_rejected(tmp_path: Path, fault: str) -> None:
    # Given
    store = RunStore.create(tmp_path, "run", {})
    root = tmp_path / "source"
    root.mkdir()
    _ = (root / "pipeline.py").write_text("def run(request=None):\n    return 42\n", encoding="utf-8")
    artifact = api.snapshot_project(store, root, api.ProjectSpec(
        paths=("pipeline.py",), backend_type="cpu",
        entrypoint=CallableRef(module="pipeline", callable="run"),
        parameter_application=CallableRef(module="pipeline", callable="run"),
    ))
    ref = store.put_artifact(artifact.model_dump_json())
    blob = store.artifacts_dir / artifact.files[0].content_ref.removeprefix("sha256:")
    if fault == "missing":
        blob.unlink()
    if fault == "corrupt":
        _ = blob.write_bytes(b"corrupt")
    # When / Then
    with pytest.raises(api.ProjectArtifactError):
        _ = api.apply_project_parameters(store, ParameterApplicationRequest(candidate_ref=ref, params={}),
                                     tmp_path / "output")
