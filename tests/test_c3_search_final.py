"""CPU composition for operator readiness and fresh same-device final rows."""

import importlib
import json
import subprocess
import sys
import io
from pathlib import Path
from time import time

import pytest

from tests.test_c3_model_runner import runner


def api(name):
    path = Path(__file__).parents[1] / "examples/c3_qwen3" / f"{name}.py"
    assert path.is_file(), f"missing operator composition: {name}"
    return importlib.import_module(f"examples.c3_qwen3.{name}")


def test_readiness_generates_independent_oracles_and_actual_baseline_checks(tmp_path):
    # Given: one resident numerical model and the exact frozen 58-prompt corpus.
    resident, backend, prepared = runner(tmp_path / "raw")
    # When
    ready = api("search_readiness").prepare_readiness(resident, tmp_path / "ready")
    # Then
    assert ready.valid and len(ready.profiles) == 3 and len(ready.oracles) == 10
    assert all(len(checks) == 2 and all(r.valid for r in checks) for checks in ready.checks.values())
    assert ready.contract_sha256 == prepared.contract_sha256 and backend.loads == 1
    assert len(json.loads(ready.oracles["calibration"].read_text())["prompts"]) == 2
    assert len(json.loads(ready.oracles["multi:0"].read_text())["prompts"]) == 8
    assert not resident.binding.sites
    resident.close()


def test_thirty_six_fresh_blocks_include_duplicate_fallback_rows(tmp_path):
    # Given: all three trajectories retained baseline; still four independently measured rows.
    from examples.c3_qwen3.search_records import SearchClock, Selection
    from examples.c3_qwen3.model_binding import load_bundle
    from examples.c3_qwen3.search_bundle import export_selection
    from kernel_optimizer.models.core import ParamSet
    resident, backend, prepared = runner(tmp_path / "raw")
    ready = api("search_readiness").prepare_readiness(resident, tmp_path / "ready")
    bundle = load_bundle(ready.baseline_bundle, {})
    selected = Selection(bundle=bundle.path, bundle_sha256=bundle.bundle_sha256, source_hashes=dict(bundle.source_hashes),
        params=ParamSet(values={}), params_sha256=bundle.params_sha256, site_groups={},
        evaluation=ready.checks["ttft"][0], contract_sha256=prepared.contract_sha256,
        model_revision=prepared.asset_spec.revision, baseline_fallback=True)
    paths = {}
    for goal in ("ttft", "single", "multi"):
        export_selection(selected, tmp_path / goal)
        paths[goal] = tmp_path / goal / "selection.json"
    before = backend.forward_calls
    started = time() - 1
    clock = SearchClock(campaign_started_unix_s=started, search_cutoff_unix_s=started + 9000,
                        final_deadline_unix_s=started + 10800)
    # When
    result = api("search_matrix").run_matrix(resident, ready, paths, clock=clock, output=tmp_path / "matrix")
    # Then
    assert len(result.cells) == 36 and all(c.evaluation.valid for c in result.cells)
    assert result.slots_used == 36 and backend.forward_calls > before
    assert [c.row for c in result.cells[:12]] == ["baseline", "ttft", "single", "multi", "multi", "single", "ttft", "baseline", "single", "multi", "baseline", "ttft"]
    assert backend.loads == 1 and result.selection_unchanged
    resident.close()


def test_operator_cli_help_does_not_load_gpu_libraries():
    # Given / When: the actual module surface in the local environment without Torch.
    result = subprocess.run([sys.executable, "-B", "-m", "examples.c3_qwen3.operator_cli", "--help"],
                            capture_output=True, text=True, timeout=30)
    # Then
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in ("readiness", "optimize-goal", "heldout", "analyze"))


def test_expired_heldout_cli_keeps_all_cells_without_model(tmp_path):
    # Given: expired inputs, deliberately without any prepared model files.
    payload = {"readiness": str(tmp_path / "absent.json"), "selections": {}, "output": str(tmp_path / "result"),
        "clock": {"campaign_started_unix_s": 1, "search_cutoff_unix_s": 9001, "final_deadline_unix_s": 10801}}
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(payload))
    # When
    process = subprocess.run([sys.executable, "-B", "-m", "examples.c3_qwen3.operator_cli", "heldout",
        "--inputs", str(path)], capture_output=True, text=True, timeout=30)
    # Then
    assert process.returncode == 1
    cells = json.loads((tmp_path / "result/matrix.json").read_text())["cells"]
    assert len(cells) == 36 and all(c["status"] == "censored" and c["evaluation"]["score"] is None for c in cells)


@pytest.mark.parametrize("stdin_clock", [False, True])
def test_real_optimize_cli_uses_configured_agent_and_one_resident(tmp_path, monkeypatch, stdin_clock):
    # Given: actual CLI composition with only loading/provider service boundaries replaced.
    from contextlib import closing
    from tests.c3_search_fakes import search_case
    from kernel_optimizer.agents.runtime import OpencodeClient, OpencodeServer
    from kernel_optimizer.config import AppConfig
    cli = api("operator_cli")
    unused, provider, _, inputs = search_case(tmp_path)
    unused.evaluator.close()
    resident = unused.runner
    loaded = []
    def load(*args):
        loaded.append(True)
        return resident
    monkeypatch.setattr(cli, "load", load)
    monkeypatch.setattr(OpencodeServer, "start", lambda self: "http://127.0.0.1:0")
    monkeypatch.setattr(OpencodeServer, "stop", lambda self: None)
    monkeypatch.setattr(OpencodeClient, "create_session", lambda self, directory, title: title)
    monkeypatch.setattr(OpencodeClient, "prompt", lambda self, *args, **kwargs: provider.prompt(*args, **kwargs))
    config = tmp_path / "config.json"
    config.write_text(AppConfig().model_dump_json())
    request = tmp_path / "search.json"
    request.write_text(inputs.model_dump_json())
    class ClockInput(io.StringIO):
        def readline(self, *args):
            assert loaded == [True]
            return super().readline(*args)
    if stdin_clock:
        monkeypatch.setattr(sys, "stdin", ClockInput(inputs.clock.model_dump_json() + "\n"))
    # When
    with closing(provider):
        code = cli.main(["optimize-goal", "--inputs", str(request), "--config", str(config), "--device", "cpu",
                        *(["--clock-from-stdin"] if stdin_clock else [])])
    # Then
    result = json.loads((inputs.output / "result.json").read_text())
    assert code == 0 and len(provider.requests) == 2
    assert result["slots_reconciled"] and resident.backend.loads == 1 and resident.closed
    assert result["selected"]["model_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"


def test_analysis_keeps_missing_scores_and_records_private_gpu_bypass(tmp_path):
    # Given: censored fresh cells and an operator-supplied audited tool-use record.
    from examples.c3_qwen3.search_matrix import expired_matrix
    from examples.c3_qwen3.search_records import SearchClock
    from examples.c3_qwen3.search_analysis import analyze_matrix
    expired_matrix(SearchClock(campaign_started_unix_s=1, search_cutoff_unix_s=9001,
                               final_deadline_unix_s=10801), tmp_path / "matrix")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps([{"session_id": "fixture", "command": "python private_gpu_test.py",
                                  "gpu_execution": True, "via_resident_helper": False}]))
    # When
    result = analyze_matrix(tmp_path / "matrix/matrix.json", audit)
    # Then
    assert not result.complete and all(c.median is None for c in result.cells)
    assert len(result.protocol_violations) == 1 and result.private_gpu_audit == "reviewed"
