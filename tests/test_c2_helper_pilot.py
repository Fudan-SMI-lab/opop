"""Pilot clock and host-local dispatch tests; all external execution is CPU-only."""

import importlib
import json
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import pytest

from kernel_optimizer.config import AppConfig
from scripts.experiments import c2_opportunity_inputs as inputs_module
from scripts.experiments.c2_local_inputs import InputError
from scripts.experiments.c2_opportunity_inputs import CampaignRun
from tests.c2_opportunity_fakes import campaign, prepared


def pilot_clock():
    return inputs_module.PilotClock(campaign_started_unix_s=1000,
        work_cutoff_unix_s=17800, final_deadline_unix_s=19000)


def test_pilot_identity_and_admission_are_separate(tmp_path):
    # Given: a five-hour identity with a twenty-minute heldout reserve.
    run = CampaignRun(AppConfig(), tmp_path, 19000, now=lambda: 1000, clock=lambda: 50,
                      pilot_clock=pilot_clock())
    # When
    _, work = run.start()
    _, heldout = run.start(heldout=True)
    # Then
    assert work.remaining() == 16800 and heldout.remaining() == 18000
    assert run.deadline_unix_s == 19000 and run.campaign_started_unix_s == 1000


def test_four_hour_default_still_rejects_five_hours(tmp_path):
    # Given / When / Then
    with pytest.raises(InputError):
        CampaignRun(AppConfig(), tmp_path, 19000, now=lambda: 1000).start()
    run = CampaignRun(AppConfig(), tmp_path, 15400, now=lambda: 1000, clock=lambda: 50)
    assert run.start()[1].remaining() == 14400


@pytest.mark.parametrize("change", [{"work_cutoff_unix_s": 17801}, {"final_deadline_unix_s": 19001},
                                    {"campaign_started_unix_s": float("nan")}])
def test_clock_cannot_reset_or_change_reserve(change):
    # Given / When / Then
    with pytest.raises(ValueError):
        inputs_module.PilotClock.model_validate({"campaign_started_unix_s": 1000,
            "work_cutoff_unix_s": 17800, "final_deadline_unix_s": 19000, **change})


def test_opportunity_cutoff_keeps_final_identity_without_work(campaign):
    # Given: work has expired, but the final heldout reserve remains.
    from scripts.experiments.c2_opportunity_program import run_opportunity
    campaign.clock[0] = 17801
    run = CampaignRun(campaign.cfg, campaign.root / "expired", 19000,
        now=lambda: campaign.clock[0], clock=lambda: campaign.clock[0], pilot_clock=pilot_clock())
    # When
    result = run_opportunity(campaign.inputs(1, "A1"), run)
    # Then
    assert result.status == "censored" and result.deadline_unix_s == 19000
    assert result.pilot_clock == pilot_clock() and result.admission_deadline_unix_s == 17800
    assert result.late_start_s == 1 and result.drain_s == 0
    assert not campaign.roles and not campaign.jobs
    manifest = json.loads((run.output / "manifest.json").read_text())
    assert manifest["campaign_started_unix_s"] == 1000


def host_spec(campaign, host):
    module = importlib.import_module("scripts.experiments.c2_helper_pilot")
    configs = []
    for gpu in (0, 1):
        path = campaign.root / f"{host}-{gpu}.json"
        cfg = campaign.cfg.model_copy(deep=True)
        cfg.opencode.port = 42510 + gpu
        path.write_text(cfg.model_dump_json())
        configs.append(path)
    return module.HostPilot(host=host, clock=pilot_clock(), output=campaign.root / host,
        gpu0_config=configs[0], gpu1_config=configs[1],
        task21={"shared": campaign.inputs(1, "B0").shared, "reference": campaign.inputs(1, "B0").reference},
        task43={"shared": campaign.inputs(1, "A0").shared, "reference": campaign.inputs(1, "A0").reference})


def test_expired_cli_keeps_every_scheduled_row_without_worker_launch(campaign):
    # Given: a historical clock, so this real CLI invocation is safe without any provider/worker.
    spec = host_spec(campaign, "A")
    path = campaign.root / "host.json"
    path.write_text(spec.model_dump_json())
    # When
    result = subprocess.run([sys.executable, "-m", "scripts.experiments.c2_helper_pilot",
        "host", "--inputs", str(path)], capture_output=True, text=True, timeout=30)
    # Then
    assert result.returncode == 1, result.stderr
    receipt = json.loads((spec.output / "controller.json").read_text())
    assert len(receipt["opportunities"]) == 4 and len(receipt["pairs"]) == 2
    assert all(row["exit_code"] is None for row in receipt["opportunities"])
    assert all(row["status"] == "censored" for row in receipt["opportunities"] + receipt["pairs"])
    assert not list(spec.output.rglob("jobs"))


def test_cli_help_is_cpu_safe():
    # Given / When
    result = subprocess.run([sys.executable, "-m", "scripts.experiments.c2_helper_pilot", "--help"],
                            capture_output=True, text=True, timeout=30)
    # Then
    assert result.returncode == 0 and "host" in result.stdout and "worker" in result.stdout


def cpu_launcher(job, *, block=False, mode="normal"):
    path = job.inputs.with_suffix(".job.json")
    path.write_text(job.model_dump_json())
    env = {**os.environ, "CPU_PILOT_MODE": mode, "CUDA_VISIBLE_DEVICES": str(job.gpu)}
    if block:
        env["CPU_PILOT_BLOCK"] = "1"
    with job.inputs.with_suffix(".log").open("wb") as log:
        return subprocess.Popen([sys.executable, "-m", "tests.c2_helper_pilot_cpu", str(path)],
            stdin=subprocess.PIPE if block else subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)


def test_fast_host_finishes_heldout_while_other_host_is_blocked(campaign, monkeypatch):
    # Given: one slow opportunity waits on an OS pipe; there is no timing-based sleep or polling.
    from scripts.experiments.c2_helper_pilot import run_host
    a, b = host_spec(campaign, "A"), host_spec(campaign, "B")
    blocked = Queue()
    launches = []
    def polling_forbidden(self):
        raise AssertionError("controller must use blocking process completion")
    monkeypatch.setattr(subprocess.Popen, "poll", polling_forbidden)
    def launch(job):
        is_blocked = job.mode == "opportunity" and job.inputs.stem == "wave1-B1"
        launches.append((job.inputs.stem, job.mode, job.gpu))
        process = cpu_launcher(job, block=is_blocked)
        if is_blocked:
            blocked.put(process)
        return process
    # When: independent controllers own their local processes, not a global wave.
    with ThreadPoolExecutor(max_workers=2) as hosts:
        slow = hosts.submit(run_host, b, launch, now=lambda: 1000)
        process = blocked.get(timeout=30)
        fast = hosts.submit(run_host, a, launch, now=lambda: 1000)
        try:
            first = fast.result(timeout=90)
            assert not slow.done()
            assert all(pair.status == "complete" for pair in first.pairs)
        finally:
            process.stdin.write(b"x")
            process.stdin.flush()
            process.stdin.close()
        second = slow.result(timeout=90)
    # Then: eight original-parent opportunities and four immediate nine-block heldouts, without a barrier.
    assert len(first.opportunities + second.opportunities) == 8
    assert len(first.pairs + second.pairs) == 4
    assert all(row.status == "complete" for row in first.opportunities + second.opportunities + second.pairs)
    for host in (a, b):
        observations = [json.loads(p.read_text()) for p in (host.output / "requests").glob("cpu-*/observed.json")]
        assert sum(len([j for j in o["jobs"] if j[0] in {"parent", "G0", "C2"}]) for o in observations) == 18
        opps = [o for o in observations if o["strategy"] is not None]
        assert sorted(o["strategy"] for o in opps) == ["provided", "provided", "targeted", "targeted"]
        assert sorted(o["gpu"] for o in opps) == ["0", "0", "1", "1"]
    assert all(p.ready_to_heldout_start_s == 0 for p in first.pairs + second.pairs)


@pytest.mark.parametrize("mode,expected", [("missing", "censored"), ("corrupt", "censored"),
                                           ("retained", "failed"), ("nonzero_valid", "complete")])
def test_failed_process_is_not_replaced_and_retained_parent_can_be_measured(campaign, mode, expected):
    # Given: only the first C2 subprocess has the designated technical/scientific outcome.
    from scripts.experiments.c2_helper_pilot import run_host
    spec = host_spec(campaign, "A")
    launches = []
    def launch(job):
        launches.append(job.inputs.stem)
        return cpu_launcher(job, mode=mode if job.inputs.stem == "wave1-A1" else "normal")
    # When
    receipt = run_host(spec, launch, now=lambda: 1000)
    # Then
    assert receipt.opportunities[1].status == expected
    assert len(launches) == 6 and len(set(launches)) == 6
    assert receipt.pairs[0].status == ("censored" if expected == "censored" else "complete")
    assert receipt.pairs[1].status == "complete"
    assert launches == ["wave1-A0", "wave1-A1", "wave1-heldout", "wave2-A0", "wave2-A1", "wave2-heldout"]
    if mode == "nonzero_valid":
        assert receipt.opportunities[1].exit_code == 7
    if mode == "retained":
        result = json.loads(receipt.opportunities[1].result.read_text())
        assert result["incumbent"]["selected"] == "parent"


def test_worker_cli_expired_is_safe_and_preserves_final_identity(campaign):
    # Given: an expired job routed through the same worker CLI used by the controller.
    from scripts.experiments.c2_helper_pilot import WorkerJob
    spec = host_spec(campaign, "A")
    inputs = campaign.root / "slot.json"
    inputs.write_text(spec.opportunity(1, "A1").model_dump_json())
    job = WorkerJob(mode="opportunity", config=spec.gpu1_config, inputs=inputs,
                    output=campaign.root / "expired-worker", gpu=1, clock=pilot_clock())
    path = campaign.root / "job.json"
    path.write_text(job.model_dump_json())
    # When
    process = subprocess.run([sys.executable, "-m", "scripts.experiments.c2_helper_pilot",
        "worker", "--inputs", str(path)], capture_output=True, text=True, timeout=30)
    # Then
    assert process.returncode == 1, process.stderr
    result = json.loads((job.output / "result.json").read_text())
    assert result["deadline_unix_s"] == 19000 and result["admission_deadline_unix_s"] == 17800
    assert result["status"] == "censored" and result["model_calls_started"] == 0
    assert not list(job.output.rglob("*.out.json"))


def test_corrupt_heldout_does_not_invent_start_or_replace_pair(campaign):
    # Given: the first heldout process leaves corrupt bytes rather than a valid result.
    from scripts.experiments.c2_helper_pilot import run_host
    spec = host_spec(campaign, "A")
    def launch(job):
        return cpu_launcher(job, mode="corrupt" if job.inputs.stem == "wave1-heldout" else "normal")
    # When
    receipt = run_host(spec, launch, now=lambda: 1000)
    # Then
    assert receipt.pairs[0].status == "censored" and receipt.pairs[0].heldout_started_unix_s is None
    assert receipt.pairs[0].ready_to_heldout_start_s is None
    assert receipt.pairs[1].status == "complete"
    assert (spec.output / "heldout/wave1/result.json").read_text() == "{broken"
