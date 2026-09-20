from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from kernel_optimizer import wiring
from kernel_optimizer.agents.task_rewriter import TaskRewriteInputs, TaskRewriterAgent
from kernel_optimizer.config import AppConfig
from kernel_optimizer.store.run_store import RunStore
from tests.c2_contract_capture import Captured, RecordingProvider, load_anchor, request_at, structured_files

pytest_plugins = ["tests.c3_search_fakes"]


@dataclass(frozen=True, slots=True)
class ProfileCase:
    runtime: wiring.Runtime
    store: RunStore
    provider: RecordingProvider
    inputs: TaskRewriteInputs


@pytest.fixture
def profile_case(tmp_path: Path) -> Iterator[ProfileCase]:
    inputs = request_at(tmp_path / "case")
    with closing(RecordingProvider(inputs.project_root)) as provider:
        runtime = wiring.Runtime(AppConfig())
        runtime.client = provider
        yield ProfileCase(runtime, RunStore(tmp_path / "run"), provider, inputs)


def operator_inputs(inputs: TaskRewriteInputs) -> TaskRewriteInputs:
    context: dict[str, JsonValue] = {
        "framework_revision": "F0-fixture", "parent_bundle_sha256": "parent-fixture",
        "task_facet": {"contract_sha256": "contract-fixture", "quality_limits": {"local_rtol": 0.02}},
        "operator_brief": {"site_id": "scale", "module_paths": ["model.scale"],
            "reference_source": "def reference(x): return x * 2\n",
            "callable_interface": "replace(site, call, params)",
            "numerical_semantics": ["preserve input/output float32"],
            "representative_calls": [{"phase": "prefill", "tensors": [
                {"shape": [2, 4], "stride": [4, 1], "dtype": "float32", "device": "cuda:0"}]}]},
        "heldout": {"input_ids": [999]}, "frozen_contract": {"heldout_prompts": ["sealed"]},
        "stage_budget": {"local_evaluations_remaining": 4, "model_evaluations_remaining": 2},
    }
    return inputs.model_copy(update={"context": context,
        "bundle_sources": {"operators.py": inputs.candidate_path.read_text(encoding="utf-8")},
        "bundle_document": {"entry": "operators.py", "files": ["operators.py"], "helpers": []}})


def test_cda_contract_when_direct_compat_profile_is_selected(profile_case: ProfileCase) -> None:
    # Given
    case = profile_case
    agent = wiring.build_task_rewriter(case.runtime.cfg, case.store, case.runtime, execution_profile="c2_direct_compat")
    # When
    with pytest.raises(Captured):
        agent.invoke(case.inputs)
    # Then: exact old schema includes the required nonempty TaskSpace.params definition.
    actual = case.provider.requests[0]
    expected = load_anchor("cda1130-direct")
    assert actual.output_schema == expected.output_schema
    assert structured_files(actual) == structured_files(expected)
    assert set(actual.files) == set(expected.files)
    assert agent.execution_profile == "c2_direct_compat" and agent.task_kind == "kernelbench"


@pytest.mark.parametrize("space", [{}, {"params": []}])
def test_old_space_constraint_when_compat_output_is_parsed(profile_case: ProfileCase, space: dict[str, JsonValue]) -> None:
    # Given
    case = profile_case
    agent = wiring.build_task_rewriter(case.runtime.cfg, case.store, case.runtime, execution_profile="c2_direct_compat")
    # When / Then
    with pytest.raises(ValidationError):
        agent.output_model.model_validate({"candidate_file": "child.py", "space": space})


@pytest.mark.parametrize("goal", ["TTFT", "single throughput", "multi throughput", "Minimize arbitrary J"])
def test_operator_schema_when_goal_changes_without_changing_profile(profile_case: ProfileCase, goal: str) -> None:
    # Given
    case = profile_case
    inputs = operator_inputs(case.inputs).model_copy(update={"goal": goal})
    agent = wiring.build_task_rewriter(case.runtime.cfg, case.store, case.runtime, execution_profile="model_operator")
    # When
    with pytest.raises(Captured):
        agent.invoke(inputs)
    # Then: machine-visible obligations are C3-only; arbitrary context and project files are not seeded.
    request = case.provider.requests[0]
    assert request.output_schema["title"] == "ModelOperatorRewriteResult"
    assert "device_kernels" in request.output_schema["required"]
    payload = structured_files(request)["analysis/task_response.json"]
    assert isinstance(payload, dict) and payload["goal"] == goal
    context = payload["context"]
    assert isinstance(context, dict) and set(context) == {
        "framework_revision", "parent_bundle_sha256", "task_facet", "operator_brief", "stage_budget", "repair_feedback"}
    assert payload["source_paths"] == [] and "project/reference.py" not in request.files
    assert agent.execution_profile == "model_operator" and agent.task_kind == "model_project_operator"
    profile = structured_files(request)["task/execution_profile.json"]
    assert isinstance(profile, dict) and profile["device_proof_status"] == "not_run"


@pytest.mark.parametrize("declarations", [[], [{"backend": "torch", "source_file": "op.py", "entry": "f"}]])
def test_operator_output_rejected_when_declarations_are_missing_or_host_only(
    profile_case: ProfileCase, declarations: list[dict[str, str]],
) -> None:
    # Given
    case = profile_case
    agent = wiring.build_task_rewriter(case.runtime.cfg, case.store, case.runtime, execution_profile="model_operator")
    # When / Then
    with pytest.raises(ValidationError):
        agent.output_model.model_validate({"candidate_file": "op.py", "bundle_file": "bundle.json",
            "site_groups": {"scale": ["model.scale"]}, "device_kernels": declarations})


def test_generic_contract_when_explicit_existing_profile_is_selected(profile_case: ProfileCase) -> None:
    # Given
    case = profile_case
    # When
    agent = wiring.build_task_rewriter(case.runtime.cfg, case.store, case.runtime, execution_profile="existing_generic")
    # Then
    assert type(agent) is TaskRewriterAgent
    assert agent.output_model.model_json_schema() == load_anchor("8526a13-generic").output_schema


def test_profile_rejected_when_selector_is_unknown(profile_case: ProfileCase) -> None:
    # Given
    case = profile_case
    # When / Then: runtime boundary rejects untyped CLI/config input too.
    with pytest.raises(ValidationError):
        wiring.build_task_rewriter(case.runtime.cfg, case.store, case.runtime, execution_profile="unknown")
