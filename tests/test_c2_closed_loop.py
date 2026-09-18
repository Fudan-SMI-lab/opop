"""Two real information/native-retune rounds with external provider/worker CPU fakes."""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

from kernel_optimizer.agents.runtime import AgentCallError, OpencodeClient, OpencodeServer, PromptResult
from kernel_optimizer.gpu.worker_client import WslGpuWorker
from kernel_optimizer.paramspace.materializer import extract_defaults
from scripts.experiments.c2_information import run_information
from scripts.experiments.c2_local_execution import acquire
from scripts.experiments.c2_local_inputs import Responses, Shared
from tests.test_c2_information_inputs import prepared as existing_prepared

prepared = existing_prepared


def loop_module():
    assert (Path(__file__).parents[1] / "scripts/experiments/c2_closed_loop.py").is_file(), "closed-loop entry is missing"
    return importlib.import_module("scripts.experiments.c2_closed_loop")


def round_number(path: Path) -> int:
    return next(int(part.removeprefix("round-")) for part in path.parts if part in {"round-1", "round-2"})


@pytest.mark.parametrize(("group", "mode"), [
    ("G0", "improve"), ("G2", "improve"), ("G2", "keep"),
    ("G0", "generation"), ("G2", "witness"), ("G0", "no_best"),
    ("G2", "child_final_invalid"), ("G0", "parent_invalid"), ("G2", "second_parent_invalid"),
])
def test_two_rounds_update_real_parent_and_preserve_elapsed_history(tmp_path, monkeypatch, prepared, group, mode):
    # Given: the original P1, not a first-stage winning child.
    module = loop_module()
    initial, _, cfg = prepared
    cfg.run.seed = 51
    for name in ("analyst", "rewriter", "parameterizer"):
        getattr(cfg.agents, name).max_retries = 0
        getattr(cfg.agents, name).max_transport_retries = 0
    original = initial.model_dump()
    active = []
    roles = []
    analyst_inputs = {}
    rewrite_history = {}
    acquisition_calls = []
    probe_params = {1: [], 2: []}
    opportunities = []

    def start(self):
        active.append(True)
        return "http://unused.invalid"

    def stop(self):
        active.pop()

    def session(self, directory, title):
        return title

    def prompt(self, session_id, text, **kwargs):
        assert active
        directory = kwargs["directory"]
        number = round_number(directory)
        title = kwargs["schema"]["title"]
        roles.append((number, title))
        if title == "BottleneckReport":
            analyst_inputs[number] = (
                extract_defaults((directory / "candidate/source.py").read_text()),
                list(csv.DictReader((directory / "tuning/trials.csv").read_text().splitlines())),
            )
        if title == "RewriteResult":
            rewrite_history[number] = json.loads((directory / "history/failed_hypotheses.json").read_text())
        if mode == "generation" and number == 1:
            raise AgentCallError("CPU fixture failed first opportunity")
        key, default = ("z", 7) if number == 1 else ("w", 9)
        source = f"import triton\nPARAMS={{'{key}': {default}, 'partner': 4}}\n@triton.jit\ndef work(): return {number + 1}\n"
        (directory / "child.py").write_text(source)
        answers = {
            "BottleneckReport": {"summary": "CPU fixture", "hypotheses": []},
            "RewriteResult": {"candidates": [{"file": "child.py", "backend": "triton",
                                               "hypothesis_id": "H1", "change_summary": "fixture"}]},
            "ParameterizationResult": {"file": "child.py", "space": {"params": [
                {"name": key, "kind": "int", "choices": list(range(80))},
                {"name": "partner", "kind": "int", "choices": [2, 4]}], "constraints": []}},
        }
        return PromptResult(text="", structured=answers[title], session_id=session_id, cost=0.1)

    def worker(self, job, timeout_s, tag, **kwargs):
        assert bool(active) == ("retune" in self.jobs_dir.parts)
        number = round_number(self.jobs_dir)
        if job["job_type"] == "static_check":
            return {"ok": True}
        if job["job_type"] == "compile_probe":
            paths = [job["kernel_src_path"], *job.get("extra_kernel_src_paths", [])]
            return {"ok": True, "results": {p: {"ok": True, "max_shared": 0} for p in paths}}
        assert job["seed"] == 0
        params = extract_defaults(Path(job["kernel_src_path"]).read_text())
        is_probe = "acquisition" in self.jobs_dir.parts
        is_parent = "parent" in self.jobs_dir.parts
        if is_probe:
            probe_params[number].append(params)
            value = 100.0 + sum(int(v) for v in params.values())
        elif is_parent:
            if mode == "parent_invalid" or (mode == "second_parent_invalid" and number == 2):
                return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "invalid parent"}
            value = 2.0 if number == 1 else 40.0
        else:
            if number == 1 and mode == "no_best" and "-wit-" in tag:
                return {"ok": True}
            invalid = number == 1 and (mode in {"witness", "no_best"}
                      or (mode == "child_final_invalid" and job["num_perf_trials"] == 100))
            if invalid:
                return {"ok": False, "failure_kind": "correctness_mismatch", "log_tail": "invalid child"}
            key, default = ("z", 7) if number == 1 else ("w", 9)
            value = (4.0 if number == 1 else 3.0) if params == {key: default, "partner": 4} else 10.0
            if mode == "keep" and number == 1:
                value += 5.0
            if job["num_perf_trials"] == 100:
                value = 20.0 if number == 1 else 30.0
        return {"ok": True, "latency_ms": {"mean": value, "median": value, "min": value, "max": value,
                "std": 0.0, "n": job["num_perf_trials"], "samples": [value] * job["num_perf_trials"]}}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is acquire.__code__:
            acquisition_calls.append((frame.f_locals["shared"].model_copy(deep=True), frame.f_locals["probe_budget"]))
        if event == "return" and frame.f_code is run_information.__code__:
            opportunities.append(arg)

    monkeypatch.setattr(OpencodeServer, "start", start)
    monkeypatch.setattr(OpencodeServer, "stop", stop)
    monkeypatch.setattr(OpencodeClient, "create_session", session)
    monkeypatch.setattr(OpencodeClient, "prompt", prompt)
    monkeypatch.setattr(WslGpuWorker, "run_job", worker)
    previous_profile = sys.getprofile()
    # When: the wrapper composes the unchanged information API for exactly two planned opportunities.
    try:
        sys.setprofile(profile)
        result = module.run_closed_loop(initial, module.LoopRun(cfg, group, tmp_path / "run"))
    finally:
        sys.setprofile(previous_profile)
    # Then: native selections update state; failures cost time but never earn an extra opportunity.
    stopped = mode in {"child_final_invalid", "parent_invalid", "second_parent_invalid"}
    expected_rounds = 1 if mode in {"child_final_invalid", "parent_invalid"} else 2
    assert len(result.rounds) == len(opportunities) == expected_rounds
    assert bool(result.terminal_error) == stopped
    assert initial.model_dump() == original and not active
    assert cfg.run.seed == 51
    assert result.initial_parent_finals == opportunities[0].parent_finals
    if mode != "parent_invalid":
        assert all(t.latency_ms.robust_ms == 2.0 for t in result.initial_parent_finals)
        assert analyst_inputs[1][0] == {"x": 1}
    assert result.elapsed_total_s >= result.rounds[-1].elapsed_started_s
    for number, row in enumerate(result.rounds, 1):
        assert (tmp_path / "run" / row.information_result).is_file()
        envelope = Responses.model_validate_json((tmp_path / "run" / f"round-{number}" / "responses.json").read_text())
        if group == "G0":
            assert envelope.probe_calls == envelope.costs.worker_attempts == 0
            assert all(r.a is None and r.b is None for r in envelope.responses)
        else:
            assert envelope.probe_calls == len(probe_params[number])
        if row.elapsed_ready_s is not None:
            assert row.elapsed_ready_s >= row.elapsed_started_s + opportunities[number - 1].wall_s
            assert extract_defaults((tmp_path / "run" / row.artifact).read_text()) == row.incumbent.params.values
    assert len(acquisition_calls) == (expected_rounds if group == "G2" else 0)
    assert all(budget == 12 for _, budget in acquisition_calls)
    if stopped:
        assert result.rounds[-1].elapsed_ready_s is None
    if expected_rounds == 2:
        assert result.rounds[1].elapsed_started_s >= result.rounds[0].elapsed_ready_s
        next_shared = Shared.model_validate_json((tmp_path / "run/round-2/shared.json").read_text())
        changed = mode in {"improve", "second_parent_invalid"}
        if changed:
            assert next_shared.parent.params.values == {"z": 7, "partner": 4}
            native_parent = min((t for t in opportunities[0].child.trials
                                 if t.status == "complete" and t.params == next_shared.parent.params),
                                key=lambda t: t.latency_ms.robust_ms)
            assert next_shared.parent == native_parent
            assert len(next_shared.trials) == 40
            assert {t.candidate_id for t in next_shared.trials} == {next_shared.parent.candidate_id}
            assert {t.space_id for t in next_shared.trials} == {next_shared.parent.space_id}
            assert next_shared.space.params[0].choices == list(range(80))
            assert next_shared.source == (tmp_path / "run" / result.rounds[0].artifact).read_text()
            assert sum(t.failure_detail.startswith("[reused_measurement=true]") for t in next_shared.trials) == 2
            assert all(t.latency_ms.robust_ms < 100 for t in next_shared.trials if t.latency_ms)
            assert all(t.latency_ms.robust_ms == 20 for t in result.rounds[0].fresh_finals)
            if group == "G2":
                assert probe_params[2] == [{"z": 0, "partner": 4}, {"z": 79, "partner": 4},
                                           {"z": 7, "partner": 2}, {"z": 7, "partner": 4}]
        else:
            assert next_shared.model_dump(exclude={"failed_hypotheses"}) == initial.model_dump(exclude={"failed_hypotheses"})
            expected_history = 0 if mode == "generation" else 1
            assert len(next_shared.failed_hypotheses) == expected_history
            if expected_history:
                assert next_shared.failed_hypotheses[0]["id"] == "H1"
                assert next_shared.failed_hypotheses[0]["parent_candidate_id"] == initial.parent.candidate_id
                assert rewrite_history[2] == next_shared.failed_hypotheses
            if mode == "keep":
                assert next_shared.failed_hypotheses[0]["outcome"] == "valid_but_not_faster"
                child_ms = next_shared.failed_hypotheses[0]["child_best_ms"]
                assert isinstance(child_ms, (int, float)) and next_shared.parent.latency_ms is not None
                assert child_ms >= next_shared.parent.latency_ms.robust_ms
        for field in ("task", "reference_source", "evaluation", "semantics", "device"):
            assert getattr(next_shared, field) == getattr(initial, field)
        if mode != "second_parent_invalid":
            assert analyst_inputs[2][0] == next_shared.parent.params.values
            assert len(analyst_inputs[2][1]) == len(next_shared.trials)
            assert result.rounds[1].incumbent.params.values == {"w": 9, "partner": 4}
    expected_calls = 0 if mode == "parent_invalid" else (3 if mode in {"child_final_invalid", "second_parent_invalid"}
                     else 4 if mode == "generation" else 6)
    assert len(roles) == expected_calls


def test_closed_loop_cli_help_is_cpu_only():
    with pytest.raises(SystemExit) as exc:
        loop_module().main(["--help"])
    assert exc.value.code == 0
