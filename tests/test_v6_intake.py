import argparse

import pytest

from kernel_optimizer.project.intake import IntakeProposal, resolve_request
from kernel_optimizer.project_cli import add_project_parser


def test_plain_entry_and_original_request_are_separate(tmp_path):
    parser = argparse.ArgumentParser()
    add_project_parser(parser)
    old = parser.parse_args(["--request", str(tmp_path/"request.json"), "--output", str(tmp_path/"out")])
    assert old.project is None and old.rounds == 2
    new = parser.parse_args(["--project", str(tmp_path), "--goal", "reduce latency", "--output", str(tmp_path/"out")])
    assert new.request is None and new.goal == "reduce latency"


def test_resolved_description_keeps_task_semantics_and_uses_deployment(tmp_path):
    (tmp_path/"entry.py").write_text("def run(): return 1\n")
    proposal = IntakeProposal(status="ready", description={
        "entry": "entry.py", "files": ["entry.py"],
        "objective": {"name": "elapsed_ms", "unit": "ms", "direction": "minimize"},
        "workload": {"items": 12}, "quality": {"atol": 0, "rtol": 0}, "device_count": 1,
    }, evidence={"workload": ["entry.py"]}, assumptions=[], questions=[])
    environment = {"devices": ["GPU-A", "GPU-B"], "lease_dir": str(tmp_path/"leases"),
                   "tools": {}, "profiling": {}, "worker_timeout_s": 600}
    request = resolve_request(proposal, tmp_path, "reduce elapsed time", environment)
    assert request.workload == {"items": 12} and request.devices == ["GPU-A"]
    assert request.binding_backend == "legacy" and request.distributed_backend is None
    proposal.description.device_count = 3
    with pytest.raises(ValueError, match="more GPUs"):
        resolve_request(proposal, tmp_path, "reduce elapsed time", environment)


def test_unresolved_semantics_do_not_create_an_executable_request(tmp_path):
    proposal = IntakeProposal(status="needs_input", description=None, evidence={}, assumptions=[],
                              questions=["Which dataset and completion condition define this workload?"])
    assert resolve_request(proposal, tmp_path, "optimize", {}) is None
