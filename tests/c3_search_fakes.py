"""Real task agent and resident evaluator; only provider and numerical device timing are fake."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from time import time

import pytest

from kernel_optimizer.agents import sandbox
from kernel_optimizer.agents.runtime import OpencodeClient, PromptResult
from kernel_optimizer.agents.task_rewriter import TaskRewriterAgent
from kernel_optimizer.config import AgentModuleConfig
from kernel_optimizer.store.run_store import RunStore
from tests.test_c3_model_runner import runner


@pytest.fixture(autouse=True)
def no_sandbox_git(monkeypatch):
    monkeypatch.setattr(sandbox, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
        TimeoutExpired=subprocess.TimeoutExpired))


class BundleProvider(OpencodeClient):
    def __init__(self, selftests=0, wrong=False):
        super().__init__("http://127.0.0.1:0")
        self.requests = []
        self.selftests = selftests
        self.wrong = wrong
        self.helper_results = []

    def create_session(self, directory, title):
        return title

    def prompt(self, session_id, text, **kwargs):
        folder = kwargs["directory"]
        inputs = json.loads((folder / "analysis/task_response.json").read_text())
        self.requests.append(inputs)
        context = inputs["context"]
        parent = inputs["bundle_document"]
        number = context["opportunity"]
        repair = context.get("repair_feedback")
        cost = 2 if number == 1 else 12
        sources = inputs["bundle_sources"]
        entry = "from .helper import body\nPARAMS={'speed': 2}\ndef replace(site, call, params): return body(call.args[0])\n"
        if parent["sites"]:
            entry = sources["operators.py"]
        (folder / "operators.py").write_text(entry)
        body = "x * 2" if number == 1 else "x + x"
        if self.wrong and not repair:
            body = "x + 999"
        (folder / "helper.py").write_text(f"def body(x): return {body}\n")
        space = {"params": [{"name": "speed", "kind": "int", "choices": list(range(cost, cost + 10))}]}
        bundle = {**parent, "entry": "operators.py", "files": ["operators.py"], "helpers": ["helper.py"],
            "sites": [{"site_id": "scale", "replacement_callable": "replace"}], "params": {"speed": cost},
            "space": space, "parent_bundle": context["parent_bundle_sha256"]}
        (folder / "bundle.json").write_text(json.dumps(bundle))
        if self.selftests and not repair:
            from examples.c3_qwen3.search_helper import request_test
            endpoint = json.loads((folder / "task/operator-helper.json").read_text())
            for _ in range(self.selftests):
                self.helper_results.append(request_test(endpoint, {"bundle": str(folder / "bundle.json"),
                    "params": {"values": {"speed": cost}}, "site_groups": {"scale": ["scale"]}}))
        return PromptResult(text="", session_id=session_id, structured={"candidate_file": "operators.py",
            "bundle_file": "bundle.json", "space": space, "site_groups": {"scale": ["scale"]},
            "recommended_configs": [{"values": {"speed": cost + 7}}, {"values": {"speed": cost + 8}}]})


def search_case(tmp_path, goal_id="ttft", selftests=0, wrong=False):
    from examples.c3_qwen3.search_records import SearchClock, SearchInputs
    from examples.c3_qwen3.search_session import OperatorSession
    resident, backend, prepared = runner(tmp_path / "raw")
    original_forward = backend.forward
    def forward(rows):
        before = backend.now
        result = original_forward(rows)
        if resident.binding.active:
            backend.now = before + (backend.now - before) / resident.binding.bundle.params["speed"]
        return result
    backend.forward = forward
    oracle = resident.create_oracles(("calibration-calibration-00", "calibration-calibration-01"))
    goal = next(g for g in prepared.contract.goals if g.id == goal_id)
    profile = resident.profile(goal, goal.search_prompt_ids)
    started = time() - 1
    inputs = SearchInputs(contract=prepared.asset_spec.contract_path, assets_manifest=prepared.asset_spec.assets_manifest,
        oracle_refs=oracle, profile=profile, goal_id=goal_id, goal=f"user goal {goal_id} fixture", output=tmp_path / "search",
        clock=SearchClock(campaign_started_unix_s=started, search_cutoff_unix_s=started + 9000,
                          final_deadline_unix_s=started + 10800))
    session = OperatorSession(resident, goal, oracle, tmp_path / "session")
    provider = BundleProvider(selftests, wrong)
    store = RunStore.create(tmp_path, "agent", {})
    agent = TaskRewriterAgent(provider, sandbox.SandboxFactory(store.run_dir / "sandboxes"), store,
        AgentModuleConfig(max_retries=0, max_transport_retries=0))
    return session, provider, agent, inputs
