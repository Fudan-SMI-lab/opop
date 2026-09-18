"""Real opportunity consumers with only provider and worker boundaries replaced."""

import json

import pytest

from kernel_optimizer.agents.runtime import OpencodeClient
from kernel_optimizer.agents.modules import BottleneckAnalystAgent, StructureRewriterAgent
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.paramspace.materializer import extract_defaults
from kernel_optimizer.store.run_store import RunStore
from scripts.experiments.c2_opportunity_inputs import CampaignRun, OpportunityInputs
from scripts.experiments.c2_opportunity_program import run_opportunity
from tests.c2_opportunity_fakes import campaign, prepared


@pytest.mark.parametrize("slot", ["A0", "A1"])
def test_targeted_order_and_g0_compatibility(campaign, monkeypatch, slot):
    # Given: one late-axis request; defaults differ from selected parameters.
    inputs = OpportunityInputs.model_validate({**campaign.inputs(1, slot).model_dump(), "probe_strategy": "targeted"})
    original_prompt, original_worker = OpencodeClient.prompt, WslGpuWorker.run_job
    sequence, endpoints = [], []
    analyst_seed, rewrite_seed = BottleneckAnalystAgent.seed_sandbox, StructureRewriterAgent.seed_sandbox
    flags = []

    def seed_analyst(self, inputs, sandbox):
        flags.append((inputs.plan_probes, inputs.probe_budget))
        return analyst_seed(self, inputs, sandbox)

    def seed_rewriter(self, inputs, sandbox):
        flags.append(inputs.report_precedes_responses)
        return rewrite_seed(self, inputs, sandbox)

    def prompt(self, session_id, text, **kwargs):
        title = kwargs["schema"]["title"]
        sequence.append(title)
        answer = original_prompt(self, session_id, text, **kwargs)
        if title == "BottleneckReport":
            answer.structured["probe_requests"] = [{"axis": "k5", "a_value": 0, "b_value": 2}]
        recommendations = [{"values": {"x": 42}}, {"values": {"x": 43}}]
        if title == "RewriteResult":
            answer.structured["candidates"][0]["recommended_configs"] = recommendations
        if title == "ParameterizationResult":
            assert json.loads((kwargs["directory"] / "analysis/recommended_configs.json").read_text()) == recommendations
            answer.structured["recommended_configs"] = recommendations
        if title == "RewriteResult" and slot == "A1":
            folder = kwargs["directory"]
            report = json.loads((folder / "analysis/bottleneck.json").read_text())
            assert report["probe_requests"][0]["axis"] == "k5"
            rows = [json.loads(line) for line in (folder / "analysis/conditional_responses.md").read_text().splitlines()]
            assert rows[0]["a_params"]["values"]["k5"] == 0
            assert rows[0]["b_params"]["values"]["k5"] == 2
        return answer

    def worker(self, job, timeout_s, tag, **kwargs):
        if job["job_type"] != "static_check" and "acquisition" in self.jobs_dir.parts:
            from pathlib import Path
            sequence.append("probe")
            endpoints.append(extract_defaults(Path(job["kernel_src_path"]).read_text()))
        return original_worker(self, job, timeout_s, tag, **kwargs)

    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    monkeypatch.setattr(BottleneckAnalystAgent, "seed_sandbox", seed_analyst)
    monkeypatch.setattr(StructureRewriterAgent, "seed_sandbox", seed_rewriter)
    # When
    result = run_opportunity(inputs, CampaignRun(campaign.cfg, campaign.root / "targeted", 15400,
        now=lambda: campaign.clock[0], clock=lambda: 900000 + campaign.clock[0]))
    # Then
    assert result.status == "complete"
    assert sequence.count("BottleneckReport") == sequence.count("RewriteResult") == 1
    assert sequence[0] == "BottleneckReport"
    assert result.information.asked == 80
    assert flags == [(slot == "A1", 12), slot == "A1"]
    native = RunStore.open(campaign.root / "targeted/information/retune")
    queued = [e.payload["params"]["values"] for e in native.iter_events() if e.type == "RECOMMENDED_CONFIG_QUEUED"]
    assert {"x": 42} in queued and {"x": 43} in queued
    assert result.acquisition_calls == (12 if slot == "A1" else 0)
    if slot == "A1":
        assert sequence[1:13] == ["probe"] * 12 and sequence[13] == "RewriteResult"
        assert [p["k5"] for p in endpoints[:2]] == [0, 2]
        assert campaign.roles[:2] == [("BottleneckReport", False, False), ("RewriteResult", False, True)]
        envelope = json.loads(result.responses.read_text())
        assert len(envelope["evidence"]["observations"]) == 12
        assert envelope["preliminary_report"]["probe_requests"][0]["axis"] == "k5"
    else:
        assert not endpoints and not any(fresh for _, _, fresh in campaign.roles)


def test_targeted_cutoff_retains_report_and_partial_acquisition(campaign, monkeypatch):
    # Given: the deadline expires after the first admitted endpoint.
    inputs = OpportunityInputs.model_validate({**campaign.inputs(1, "A1").model_dump(), "probe_strategy": "targeted"})
    original = WslGpuWorker.run_job
    def worker(self, job, timeout_s, tag, **kwargs):
        result = original(self, job, timeout_s, tag, **kwargs)
        if job["job_type"] != "static_check" and "acquisition" in self.jobs_dir.parts:
            campaign.clock[0] = 15401
        return result
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    # When
    result = run_opportunity(inputs, CampaignRun(campaign.cfg, campaign.root / "cutoff", 15400,
        now=lambda: campaign.clock[0], clock=lambda: 900000 + campaign.clock[0]))
    # Then
    assert result.status == "censored" and result.acquisition_calls == 1
    assert [title for title, _, _ in campaign.roles] == ["BottleneckReport"]
    envelope = json.loads(result.responses.read_text())
    assert envelope["preliminary_report"]["summary"] == "fixture"
    assert envelope["probe_calls"] == len(envelope["evidence"]["observations"]) == 1
    assert envelope["responses"][0]["b"] is None
