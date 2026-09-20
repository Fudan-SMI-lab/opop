import json
from contextlib import closing
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from kernel_optimizer.agents.model_operator_rewriter import ModelOperatorRewriteResult
from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient
from kernel_optimizer.agents.task_rewriter import TaskRewriterAgent
from kernel_optimizer.config import AppConfig
from kernel_optimizer.models.core import sha256_text
from kernel_optimizer.store.run_store import RunStore
from kernel_optimizer.wiring import ExecutionProfile, Runtime, build_task_rewriter
from tests.c2_contract_capture import SOURCE, request_at
from tests.test_c3_rewriter_parent_snapshot import EditingProvider
from tests.test_task_agent_profiles import operator_inputs

pytest_plugins = ["tests.c3_search_fakes"]
CHILD: Final = "PARAMS = {'tile': 2}\ndef run(x): return x + x\n"


def agent_for(root: Path, provider: OpencodeClient, profile: ExecutionProfile) -> TaskRewriterAgent:
    cfg = AppConfig()
    cfg.agents.rewriter.max_retries = 0
    cfg.agents.rewriter.max_transport_retries = 0
    runtime = Runtime(cfg)
    runtime.client = provider
    return build_task_rewriter(cfg, RunStore(root / "run"), runtime, execution_profile=profile)


@pytest.mark.parametrize("changed", [False, True])
def test_direct_compat_snapshot_when_provider_edits_parent_mirror(tmp_path: Path, changed: bool) -> None:
    # Given
    inputs = request_at(tmp_path / "case")
    source = CHILD if changed else SOURCE

    def emit(folder: Path) -> dict[str, JsonValue]:
        (folder / "child.py").write_text(source, encoding="utf-8")
        (folder / "candidate/current.py").write_text("def run(x): return x - 99\n", encoding="utf-8")
        return {"candidate_file": "child.py"}

    with closing(EditingProvider(emit)) as provider:
        agent = agent_for(tmp_path, provider, "c2_direct_compat")
        # When / Then: old model-facing schema still uses the fixed original-parent authority.
        if changed:
            outcome = agent.invoke(inputs)
            assert outcome.output.candidate_file == "child.py" and outcome.attempts == 1
        else:
            with pytest.raises(AgentCallError):
                agent.invoke(inputs)


@pytest.mark.parametrize("source_file", ["operators.py", "missing.py", "../outside.py"])
def test_operator_declaration_when_validated_against_declared_bundle(tmp_path: Path, source_file: str) -> None:
    # Given: these CPU fixture sources test bookkeeping only, not Triton execution.
    inputs = operator_inputs(request_at(tmp_path / "case"))

    def emit(folder: Path) -> dict[str, JsonValue]:
        (folder / "operators.py").write_text(CHILD, encoding="utf-8")
        (folder / "bundle.json").write_text(json.dumps({"entry": "operators.py", "files": ["operators.py"]}), encoding="utf-8")
        (folder / "task/execution_profile.json").write_text('{"goal":"provider-edited mirror"}', encoding="utf-8")
        return {"candidate_file": "operators.py", "bundle_file": "bundle.json",
            "site_groups": {"scale": ["model.scale"]},
            "device_kernels": [{"backend": "triton", "source_file": source_file, "entry": "run"}]}

    with closing(EditingProvider(emit)) as provider:
        agent = agent_for(tmp_path, provider, "model_operator")
        # When / Then: a declaration produces NOT_RUN evidence, never device PASS.
        if source_file == "operators.py":
            outcome = agent.invoke(inputs)
            assert isinstance(outcome.output, ModelOperatorRewriteResult)
            artifact = json.loads(outcome.sandbox.read_output("analysis/operator_declaration.json"))
            assert artifact["device_proof_status"] == "not_run"
            assert artifact["goal"] == inputs.goal
            assert artifact["source_sha256"] == {"operators.py": sha256_text(CHILD)}
            assert artifact["device_kernels"][0]["entry"] == "run"
        else:
            with pytest.raises(AgentCallError):
                agent.invoke(inputs)


def test_operator_scope_rejected_when_candidate_selects_another_group(tmp_path: Path) -> None:
    # Given
    inputs = operator_inputs(request_at(tmp_path / "case"))

    def emit(folder: Path) -> dict[str, JsonValue]:
        (folder / "operators.py").write_text(CHILD, encoding="utf-8")
        (folder / "bundle.json").write_text(json.dumps({"entry": "operators.py", "files": ["operators.py"]}), encoding="utf-8")
        return {"candidate_file": "operators.py", "bundle_file": "bundle.json",
            "site_groups": {"other": ["model.decoder"]},
            "device_kernels": [{"backend": "cuda", "source_file": "operators.py", "entry": "run"}]}

    with closing(EditingProvider(emit)) as provider:
        agent = agent_for(tmp_path, provider, "model_operator")
        # When / Then
        with pytest.raises(AgentCallError):
            agent.invoke(inputs)
