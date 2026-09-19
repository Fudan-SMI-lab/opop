"""Admission/drain and optional helper cutoff propagation through real runner boundaries."""

import json

import pytest

from kernel_optimizer.agents import self_test, self_test_context
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from scripts.experiments.c2_opportunity_inputs import CampaignRun
from scripts.experiments.c2_local_inputs import InputError
from scripts.experiments.c2_opportunity_program import run_opportunity
from tests.c2_opportunity_fakes import campaign, prepared
from tests.test_agent_self_test import seeded
from tests.test_c2_helper_pilot import pilot_clock


def test_opportunity_records_admitted_drain_and_stops_new_work(campaign, monkeypatch):
    # Given: the first admitted parent measurement drains past both pilot deadlines.
    original = WslGpuWorker.run_job
    def worker(self, job, timeout_s, tag, **kwargs):
        raw = original(self, job, timeout_s, tag, **kwargs)
        if job["job_type"] != "static_check":
            campaign.clock[0] = 19010
        return raw
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    run = CampaignRun(campaign.cfg, campaign.root / "drain", 19000, pilot_clock=pilot_clock(),
        now=lambda: campaign.clock[0], clock=lambda: campaign.clock[0])
    # When
    result = run_opportunity(campaign.inputs(1, "A0"), run)
    # Then
    assert result.status == "censored" and len(campaign.jobs) == 1 and not campaign.roles
    assert result.drain_s == 1210 and result.final_drain_s == 10


@pytest.mark.parametrize("cutoff,expected_calls,drain", [(None, 2, 0), (999, 0, 0), (1001, 1, 1)])
def test_helper_stops_new_submissions_but_records_admitted_drain(seeded, monkeypatch, cutoff, expected_calls, drain):
    # Given: a real helper pipeline with only the external worker replaced.
    context = json.loads(seeded.read_text())
    if cutoff is not None:
        context["formal_cutoff_unix_s"] = cutoff
    seeded.write_text(json.dumps(context))
    root = seeded.parent.parent
    candidate = root / "candidate.py"
    candidate.write_text("PARAMS={'x': 1}\n")
    clock, calls = [1000.0], []
    monkeypatch.setattr(self_test.time, "time", lambda: clock[0])
    def worker(self, job, timeout_s, tag, lock_mode="exclusive"):
        calls.append(job)
        clock[0] = 1002
        return {"ok": True, "compiled": True, "correct": True,
                "latency_ms": {"mean": 2.0, "median": 2.0}}
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    output = root / "result"
    # When
    code = self_test.main(["--context", str(seeded), "--candidate", str(candidate),
                           "--mode", "quick", "--output", str(output)])
    # Then
    assert len(calls) == expected_calls
    result = json.loads((output / "result.json").read_text())
    assert result["drain_s"] == drain and result["formal_cutoff_unix_s"] == cutoff
    assert code == (0 if cutoff is None else 1)
    if cutoff is not None:
        assert result["score_ms"] is None


def test_helper_cutoff_scope_resets_after_exception():
    # Given / When: a process-local context scope, not an environment variable.
    with pytest.raises(InputError):
        with self_test_context.formal_helper_cutoff(17800):
            raise InputError("fixture")
    # Then
    assert self_test_context.FORMAL_HELPER_CUTOFF.get() is None


@pytest.mark.parametrize("drains", [False, True])
def test_heldout_uses_final_reserve_and_records_final_drain(campaign, monkeypatch, drains):
    # Given: two real resolved opportunities with the common final identity.
    from scripts.experiments.c2_opportunity_heldout import HeldoutInputs, run_heldout
    clock = pilot_clock()
    for slot in ("A0", "A1"):
        run_opportunity(campaign.inputs(1, slot), CampaignRun(campaign.cfg, campaign.root / slot, 19000,
            now=lambda: campaign.clock[0], clock=lambda: campaign.clock[0], pilot_clock=clock))
    campaign.clock[0] = 17801
    jobs = []
    def worker(self, job, timeout_s, tag, **kwargs):
        if job["job_type"] == "static_check":
            return {"ok": True}
        jobs.append(job)
        assert job["num_perf_trials"] == 100 and job["num_correct_trials"] == 5
        if drains:
            campaign.clock[0] = 19010
        return {"ok": True, "latency_ms": {"mean": 100, "median": 100, "min": 100,
                "max": 100, "std": 0, "n": 100, "samples": [100] * 100}}
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    inputs = HeldoutInputs(**campaign.inputs(1, "A0").model_dump(),
        g0=campaign.root / "A0/result.json", c2=campaign.root / "A1/result.json")
    # When: heldout starts after the work cutoff, while its own deadline remains open.
    result = run_heldout(inputs, CampaignRun(campaign.cfg, campaign.root / "heldout", 19000,
        now=lambda: campaign.clock[0], clock=lambda: campaign.clock[0], pilot_clock=clock))
    # Then
    assert result.status == ("censored" if drains else "complete")
    assert len(jobs) == (1 if drains else 9)
    assert result.drain_s == (10 if drains else 0)
    assert result.deadline_unix_s == result.admission_deadline_unix_s == 19000
    assert result.pilot_clock == clock


def test_summary_rejects_missing_pilot_identity(campaign):
    # Given: an expired pilot row and a row falsely claiming the legacy clock.
    from scripts.experiments.c2_opportunity_summary import summarize
    campaign.clock[0] = 17801
    result = run_opportunity(campaign.inputs(1, "A0"), CampaignRun(campaign.cfg, campaign.root / "expired", 19000,
        now=lambda: campaign.clock[0], clock=lambda: campaign.clock[0], pilot_clock=pilot_clock()))
    other = result.model_copy(update={"slot": "A1", "pilot_clock": None})
    # When / Then
    with pytest.raises(InputError, match="pilot clock"):
        summarize([result, other], [])


def test_controller_process_drain_does_not_reset_next_wave(campaign):
    # Given: a heldout subprocess admitted on time but observed terminal after final cutoff.
    from scripts.experiments.c2_helper_pilot import run_host
    from tests.test_c2_helper_pilot import cpu_launcher, host_spec
    clock = [1000.0]
    launched = []
    def launch(job):
        launched.append(job.inputs.stem)
        process = cpu_launcher(job)
        if job.mode == "heldout":
            clock[0] = 19001
        return process
    # When
    receipt = run_host(host_spec(campaign, "A"), launch, now=lambda: clock[0])
    # Then
    assert receipt.pairs[0].drain_s == receipt.pairs[0].final_drain_s == 1
    assert launched == ["wave1-A0", "wave1-A1", "wave1-heldout"]
    assert all(row.status == "censored" for row in receipt.opportunities[2:] + receipt.pairs[1:])
