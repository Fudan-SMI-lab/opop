"""Subprocess test harness: real runners, fake provider/worker, and no sandbox Git invocation."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_optimizer.agents import sandbox
from kernel_optimizer.agents.runtime import OpencodeClient
from scripts.experiments import c2_helper_pilot as pilot
from scripts.experiments.c2_opportunity_inputs import CampaignRun, OpportunityInputs
from tests.c2_opportunity_fakes import campaign
from tests.test_c2_information_inputs import prepared


@pytest.fixture(autouse=True)
def sandbox_without_git(monkeypatch):
    monkeypatch.setattr(sandbox, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
        TimeoutExpired=subprocess.TimeoutExpired))


def main() -> int:
    job = pilot.WorkerJob.model_validate_json(Path(sys.argv[1]).read_text())
    root = job.inputs.parent / f"cpu-{job.inputs.stem}"
    root.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        sandbox_without_git.__wrapped__(patch)
        fixture = campaign.__wrapped__(root, patch, prepared.__wrapped__())
        patch.setattr(pilot, "CampaignRun", lambda cfg, output, deadline, **kwargs: CampaignRun(
            cfg, output, deadline, now=lambda: fixture.clock[0], clock=lambda: 900000 + fixture.clock[0], **kwargs))
        inputs = OpportunityInputs.model_validate_json(job.inputs.read_text()) if job.mode == "opportunity" else None
        mode = os.environ.get("CPU_PILOT_MODE", "normal")
        if os.environ.get("CPU_PILOT_BLOCK"):
            sys.stdin.buffer.read(1)
        if mode == "missing":
            return 7
        if mode == "corrupt":
            job.output.mkdir(parents=True)
            (job.output / "result.json").write_text("{broken")
            return 0
        if mode == "retained":
            fixture.mode[0] = "service_error"
        original = OpencodeClient.prompt
        def prompt(self, session_id, text, **kwargs):
            context = json.loads((kwargs["directory"] / "task/self_test.json").read_text())
            assert context["formal_cutoff_unix_s"] == job.clock.work_cutoff_unix_s
            return original(self, session_id, text, **kwargs)
        patch.setattr(OpencodeClient, "prompt", prompt)
        code = pilot.run_worker(job)
        (root / "observed.json").write_text(json.dumps({"jobs": fixture.jobs, "roles": fixture.roles,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "strategy": inputs.probe_strategy if inputs else None}))
        return 7 if mode == "nonzero_valid" else code


if __name__ == "__main__":
    raise SystemExit(main())
