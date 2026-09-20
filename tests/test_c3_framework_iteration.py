"""Fast workflow transitions use durable quotas and explicit, isolated execution profiles."""

import importlib
from pathlib import Path

import pytest

from examples.c3_qwen3.device_records import StageIdentity
from examples.c3_qwen3.fast_prepare import FastAccess, prepare_fast
from examples.c3_qwen3.runner_records import RunnerError
from kernel_optimizer.config import AppConfig

SETUP = Path(__file__).parents[1] / "results/c3-fast-kernel-iteration/setup"


def api(name):
    path = Path(__file__).parents[1] / "examples/c3_qwen3" / f"{name}.py"
    assert path.is_file(), f"missing fast composition: {name}"
    return importlib.import_module(f"examples.c3_qwen3.{name}")


def test_dev_budget_resume_preserves_D_and_every_failed_admission(tmp_path):
    # Given: a single development ledger shared across framework revisions.
    module = api("fast_budget")
    stage = StageIdentity(profile="c3_fast_device", stage="development", framework_id="F0")
    spec = module.BudgetSpec(lane="development", stage=stage, started_unix_s=1000, deadline_unix_s=6400)
    clock = [1001.0]
    ledger = module.StageBudget(tmp_path / "budget.json", spec, now=lambda: clock[0])
    assert ledger.admit(stage, "local") == 0
    assert ledger.admit(stage, "model") == 0
    # When: reconstruct after framework engineering/wait, without starting a new clock.
    clock[0] = 2000
    resumed = module.StageBudget(tmp_path / "budget.json", spec, now=lambda: clock[0])
    next_stage = stage.model_copy(update={"framework_id": "F1"})
    resumed.activate(next_stage)
    # Then
    assert resumed.remaining("local") == 11 and resumed.remaining("model") == 3
    assert resumed.spec.deadline_unix_s == 6400
    assert resumed.admit(stage, "local") is None
    assert resumed.admit(next_stage, "local") == 1
    clock[0] = 6401
    assert resumed.admit(next_stage, "local") is None
    with pytest.raises(RunnerError):
        module.StageBudget(tmp_path / "budget.json", spec.model_copy(update={"deadline_unix_s": 9000}), now=lambda: clock[0])


def test_stage_caps_are_separate_and_preparation_cannot_be_free(tmp_path):
    # Given: setup and formal lanes have distinct declared accounting vectors.
    module = api("fast_budget")
    setup = StageIdentity(profile="c3_fast_device", stage="setup", framework_id="F0")
    ledger = module.StageBudget(tmp_path / "setup.json", module.BudgetSpec(lane="setup", stage=setup,
        started_unix_s=1000, deadline_unix_s=2000), now=lambda: 1001)
    # When / Then
    assert [ledger.admit(setup, "profile") for _ in range(4)] == [0, 1, 2, None]
    assert [ledger.admit(setup, "local") for _ in range(4)] == [0, 1, 2, None]
    formal = setup.model_copy(update={"stage": "formal_search"})
    lane = module.StageBudget(tmp_path / "formal.json", module.BudgetSpec(lane="formal_search", stage=formal,
        started_unix_s=1000, deadline_unix_s=2000), now=lambda: 1001)
    assert lane.admit(formal, "capture") == 0
    assert lane.remaining("local") == 7
    assert [lane.admit_agent(formal) for _ in range(3)] == [0, 1, None]
    assert lane.remaining("model") == 4


def test_fast_config_changes_actual_timeout_without_mutating_C2_config():
    # Given: a shared AppConfig must not become a global fast-workflow setting.
    cfg = AppConfig()
    original = cfg.model_dump()
    # When
    scoped = api("fast_workflow").scoped_config(cfg)
    # Then
    assert scoped.opencode.request_timeout_s == 480
    assert scoped.agents.rewriter.max_retries == scoped.agents.rewriter.max_transport_retries == 0
    assert cfg.model_dump() == original


def test_new_fast_phases_do_not_route_through_old_manual_pins():
    # Given / When: the actual existing input boundary, not a replacement corpus.
    dev = prepare_fast(SETUP / "contract.json", SETUP / "preflight.json",
        FastAccess(profile="c3_fast_device", phase="development"))
    formal = prepare_fast(SETUP / "contract.json", SETUP / "preflight.json",
        FastAccess(profile="c3_fast_device", phase="formal_search"))
    # Then
    assert len(dev.task.prompts_by_id) == len(formal.task.prompts_by_id) == 16
    assert all(p.split != "heldout" for p in dev.task.prompts_by_id.values())
    assert dev.task.contract_sha256.startswith("2c0e2fc")
    assert not any(g.final_prompt_groups for g in formal.task.contract.goals)


def test_development_go_uses_real_local_and_model_pipeline_then_stops(tmp_path):
    # Given: numerical device observations replace only the physical GPU boundary.
    from contextlib import closing
    from tests.c3_framework_fakes import workflow_case
    generator, provider, _, _ = workflow_case(tmp_path / "F0")
    # When
    with closing(provider):
        result = api("fast_development").develop(generator, tmp_path / "dev.json", tmp_path / "F0/result")
    # Then
    assert result.status == "DEV_GO", result.model_dump_json()
    assert len(provider.requests) == 1 and result.candidates[-1].artifact.author == "agent"
    assert result.candidates[-1].locals[-1].official_model_score is None
    assert result.candidates[-1].model.valid
    assert generator.evaluator.budget.state.operations["model"] == 2
    with pytest.raises(RunnerError):
        api("fast_development").develop(generator, tmp_path / "dev.json", tmp_path / "repeat")
    assert not any("heldout" in str(k).lower() for k in provider.requests[0]["context"].keys())
    generator.evaluator.runtime.runner.close()


def test_ordinary_artifact_error_gets_one_fixed_repair_not_new_framework(tmp_path):
    # Given: the first numerical operator is incorrect; the fixed repair computes the reference.
    from contextlib import closing
    from tests.c3_framework_fakes import workflow_case
    generator, provider, frame, _ = workflow_case(tmp_path / "F0", mode="wrong_then_repair")
    # When
    with closing(provider):
        result = api("fast_development").develop(generator, tmp_path / "dev.json", tmp_path / "F0/result")
    # Then
    assert result.status == "DEV_GO" and len(provider.requests) == 2
    assert result.framework == frame and result.candidates[0].status == "local_failed"
    assert provider.requests[0]["context"]["repair_feedback"] is None
    assert provider.requests[1]["context"]["repair_feedback"] is not None
    assert len(set(provider.sessions)) == 2
    assert all(r["bundle_document"]["sites"] == [] for r in provider.requests)
    generator.evaluator.runtime.runner.close()


def test_transport_failure_is_not_redrawn(tmp_path):
    # Given / When
    from contextlib import closing
    from tests.c3_framework_fakes import workflow_case
    generator, provider, _, _ = workflow_case(tmp_path / "F0", mode="transport")
    with closing(provider):
        result = api("fast_development").develop(generator, tmp_path / "dev.json", tmp_path / "F0/result")
    # Then
    assert result.status == "PARENT_TRIAGE" and len(provider.requests) == 1
    assert result.candidates[0].status == "transport_failed"
    generator.evaluator.runtime.runner.close()


def test_develop_cli_composes_real_runtime_config_and_stops(tmp_path, monkeypatch):
    # Given: real CLI/factory composition, with only native load/device/provider boundaries fake.
    import json
    from contextlib import closing
    from kernel_optimizer.agents.runtime import OpencodeClient, OpencodeServer
    from tests.c3_framework_fakes import workflow_case
    from examples.c3_qwen3 import fast_runtime
    generator, provider, frame, target = workflow_case(tmp_path / "prepared")
    engine = generator.evaluator
    config, framework, target_path = (tmp_path / n for n in ("config.json", "framework.json", "target.json"))
    config.write_text(AppConfig().model_dump_json())
    framework.write_text(frame.model_dump_json())
    target_path.write_text(target.model_dump_json())
    prepared = generator.context.prepared.task
    spec = fast_runtime.RunInputs(contract=prepared.asset_spec.contract_path, assets_manifest=prepared.asset_spec.assets_manifest,
        agent_config=config, framework=framework, target=target_path, output=tmp_path / "F0",
        budget=engine.budget.path, goal_text="Same observed operator, full-model objective.",
        oracle_refs=engine.files.oracle, development_state=tmp_path / "dev.json", device="cpu")
    request = tmp_path / "run.json"
    request.write_text(spec.model_dump_json())
    timeouts = []
    def start(self):
        timeouts.append(self.cfg.request_timeout_s)
        return "http://127.0.0.1:0"
    monkeypatch.setattr(fast_runtime, "load_fast", lambda *args: engine.runtime.runner)
    monkeypatch.setattr(fast_runtime, "TorchLocalDevice", lambda *args: engine.runtime.device)
    monkeypatch.setattr(OpencodeServer, "start", start)
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: None)
    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", lambda self, *args, **kwargs: provider.prompt(*args, **kwargs))
    # When
    with closing(provider):
        code = api("fast_cli").main(["develop", "--inputs", str(request)])
    # Then
    assert code == 0 and timeouts == [480]
    result = json.loads((tmp_path / "F0/result.json").read_text())
    assert result["phase"] == "development" and result["status"] == "DEV_GO"
    assert len(provider.requests) == 1 and engine.runtime.runner.closed
    assert json.loads((tmp_path / "dev.json").read_text())["status"] == "DEV_GO"
