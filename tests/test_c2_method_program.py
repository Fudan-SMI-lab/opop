"""CPU-only boundary and preregistered decision tests."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_c2_information_inputs import prepared


def method_module(name: str = "protocol"):
    # Given: the requested implementation must exist, rather than skip coverage.
    path = Path(__file__).parents[1] / "scripts/experiments" / f"c2_method_{name}.py"
    assert path.is_file()
    return importlib.import_module(f"scripts.experiments.c2_method_{name}")


def test_fixed_slots_and_rotations():
    # Given / When: read the executable preregistration.
    module = method_module()
    slots = module.SLOTS
    # Then: all four cells and exact arm orders survive, including H-before-P.
    assert [(s.task, s.rep, s.order) for s in slots.values()] == [
        ("level3:43", 0, ("G0", "L", "P", "H")),
        ("level3:21", 0, ("L", "H", "G0", "P")),
        ("level3:21", 1, ("P", "G0", "H", "L")),
        ("level3:43", 1, ("H", "P", "L", "G0")),
    ]
    for slot in slots:
        first, second, third = module.final_orders(slot)
        assert first == ("parent", *slots[slot].order)
        assert second == first[::-1]
        assert third == first[1:] + first[:1]


@pytest.mark.parametrize(("h", "expected"), [(98.0, "pass"), (98.001, "fail"), (None, "inconclusive")])
def test_gate_boundary_and_missing_finals(h: float | None, expected: str):
    # Given: exact 2% boundary, with nonoverlapping full blocks.
    module = method_module("gates")
    # When: apply the deterministic investment threshold.
    comparison = module.compare_blocks([100.0] * 3, [] if h is None else [h] * 3)
    # Then: missing data cannot pass and the boundary is inclusive.
    assert comparison.status == expected


def test_overlap_is_not_a_pass():
    # Given / When: medians improve but block ranges overlap.
    result = method_module("gates").compare_blocks([99.0, 100.0, 101.0], [90.0, 95.0, 99.0])
    # Then: strict nonoverlap is independently required.
    assert result.status == "fail"


@pytest.mark.parametrize("argv,code", [(["--help"], 0), (["--phase", "wrong"], 2),
                                      (["--slot", "A2"], 2)])
def test_cli_parser_exits_without_services(argv: list[str], code: int):
    # Given / When: only parser-level inputs, no model or GPU setup.
    with pytest.raises(SystemExit) as exc:
        method_module("program").main(argv)
    # Then: argparse-compatible help and bad arguments have stable exit codes.
    assert exc.value.code == code


def test_expired_core_keeps_all_opportunities_without_services(tmp_path, monkeypatch, prepared):
    # Given: the entire scheduling allowance elapsed before admission.
    module, protocol = method_module("program"), method_module()
    shared, _, cfg = prepared
    reference, snapshot = tmp_path / "reference.py", tmp_path / "shared.json"
    reference.write_text(shared.reference_source)
    snapshot.write_text(shared.model_dump_json())
    inputs = protocol.MethodInputs(slot="A1", shared=snapshot, reference=reference)
    starts = []
    monkeypatch.setattr(module.Runtime, "__enter__", lambda self: starts.append(True))
    # When: invoke the real scheduler with a deterministic expired clock.
    result = module.run_core(inputs, cfg, tmp_path / "run", deadline=protocol.Deadline(0, lambda: 14401))
    # Then: no model/worker start, four censored rows, separately reported drain.
    assert not starts
    assert len(result.opportunities) == 4
    assert all(row.status == "censored" and row.asked == 0 for row in result.opportunities)
    assert result.drain_s == 1


@pytest.mark.parametrize("mismatch", ["slot", "task"])
def test_cli_rejects_identity_before_config_or_services(tmp_path, prepared, mismatch):
    # Given: nonexistent config proves validation happens first.
    protocol = method_module()
    shared, _, _ = prepared
    reference, snapshot = tmp_path / "reference.py", tmp_path / "shared.json"
    reference.write_text(shared.reference_source)
    snapshot.write_text(shared.model_dump_json())
    inputs = protocol.MethodInputs(slot="A1" if mismatch == "slot" else "A0", shared=snapshot, reference=reference)
    wrapper = tmp_path / "inputs.json"
    wrapper.write_text(inputs.model_dump_json())
    # When: task21 is presented to task43's slot or wrapper slot disagrees.
    completed = subprocess.run([sys.executable, "-m", "scripts.experiments.c2_method_program",
        "--phase", "core", "--slot", "A0", "--config", str(tmp_path / "absent.yaml"),
        "--inputs", str(wrapper), "--output", str(tmp_path / "run")],
        capture_output=True, text=True, check=False, timeout=30)
    # Then: no output/service directory is created.
    assert completed.returncode == 1 and not (tmp_path / "run").exists()
    assert completed.stderr.partition(":")[0] == "InputError"


def test_summary_retains_missing_cells_and_does_not_trigger_a():
    # Given / When: no fresh core results are available.
    summary = method_module("gates").summarize_core([])
    # Then: all four remain denominator rows, without manufactured finals.
    assert summary.status == "inconclusive"
    assert len(summary.cells) == 4
    assert summary.branch.phase is None


def core_rows():
    protocol = method_module()
    rows = []
    for slot, cell in protocol.SLOTS.items():
        def finals(value: float):
            return [{"trial_id": str(i), "candidate_id": "fresh", "space_id": "fresh", "params": {"values": {}},
                     "status": "complete", "latency_ms": {"mean": value, "median": value, "std": 0,
                     "min": value, "max": value, "n_samples": 100}} for i in range(3)]
        rows.append(protocol.CellResult.model_validate({"slot": slot, "task": cell.task, "rep": cell.rep,
            "shared_id": cell.task, "parent_finals": finals(100), "opportunities": [
                {"arm": arm, "artifact": "fresh.py", "params": {"values": {}}, "backend": "triton",
                 "asked": 40, "status": "complete", "finals": finals(98 if arm == "H" else 100)}
                for arm in cell.order]}))
    return rows


@pytest.mark.parametrize("censor", [False, True])
def test_global_gate_censors_any_unfinished_core_work(censor: bool):
    # Given: three clear primary wins, one known invalid H, optionally missing secondary finals.
    rows = core_rows()
    rows[3] = rows[3].model_copy(update={"opportunities": [
        r.model_copy(update={"status": "invalid"}) if r.arm == "H" else r for r in rows[3].opportunities]})
    if censor:
        rows[0] = rows[0].model_copy(update={"opportunities": [
            r.model_copy(update={"finals": []}) if r.arm == "L" else r for r in rows[0].opportunities]})
    # When: summarize all four cells, not only the winners.
    summary = method_module("gates").summarize_core(rows)
    # Then: known invalidity stays in denominator; any missing measurement prevents A.
    assert summary.status == ("inconclusive" if censor else "pass")
    assert summary.branch.phase == (None if censor else "A")
    assert summary.branch.study_opportunities == (0 if censor else 8)


def test_b_priority_uses_original_valid_h_eligibility():
    # Given: a censored core plus one resolved native-eligible H, without latency filtering.
    rows = core_rows()
    rows[0] = rows[0].model_copy(update={"opportunities": [
        r.model_copy(update={"native_expansion_eligible": True}) if r.arm == "H" else r for r in rows[0].opportunities]})
    # When: only this cell is supplied; other cells remain censored.
    summary = method_module("gates").summarize_core(rows[:1])
    # Then: B is subset-only and cannot manufacture four eligible cells.
    assert summary.status == "inconclusive"
    assert summary.branch.phase == "B" and summary.branch.eligible_slots == ("A0",)
    assert summary.branch.study_opportunities == 2 and not summary.branch.ready


def test_relative_wrapper_paths_are_rejected():
    # Given / When: relative paths would change meaning during common-input staging.
    protocol = method_module()
    with pytest.raises(ValueError):
        protocol.MethodInputs(slot="A0", shared=Path("shared.json"), reference=Path("reference.py"))


def test_external_submission_admission_restores_original_methods(monkeypatch):
    # Given: one admitted external operation advances beyond the four-hour scheduling cap.
    from kernel_optimizer.agents.runtime import OpencodeClient
    from kernel_optimizer.gpu.worker_client import WslGpuWorker
    protocol, guard = method_module(), method_module("admission")
    calls = []
    now = [0.0]
    def external():
        calls.append(True)
        now[0] = 14401.0
    monkeypatch.setattr(OpencodeClient, "prompt", external)
    monkeypatch.setattr(WslGpuWorker, "run_job", external)
    # When: the second submission is made after the first has drained past cutoff.
    with guard.admission(protocol.Deadline(0, lambda: now[0])):
        OpencodeClient.prompt()
        with pytest.raises(protocol.SchedulingExpired):
            WslGpuWorker.run_job()
    # Then: no second expensive call and no process-wide interception leaks.
    assert calls == [True]
    assert OpencodeClient.prompt is external and WslGpuWorker.run_job is external


def test_c_dispatch_requires_native_request_and_verified_bridge():
    # Given: valid original H and no native B eligibility; core itself is incomplete.
    protocol, gates = method_module(), method_module("gates")
    evidence = protocol.ConditionalReadiness(planner_requested_slots=("A0",), bridge_verified=True,
                                            worker_budget_verified=True)
    # When: future T7 supplies native planner eligibility, not manually selected axes.
    summary = gates.summarize_core(core_rows()[:1], evidence)
    # Then: only the original eligible cell is planned, never executed or marked ready.
    assert summary.branch.phase == "C" and summary.branch.eligible_slots == ("A0",)
    assert summary.branch.study_opportunities == 2 and not summary.branch.ready


@pytest.mark.parametrize("asked,expected", [(0, "censored"), (39, "censored"), (40, "complete")])
def test_b40_completion_uses_native_asked_count(asked: int, expected: str):
    # Given: helper reports success without promising the native asked count.
    from scripts.experiments.c2_retune import RetuneResult
    native = method_module("native")
    # When: classify the opportunity from native study accounting.
    status = native.native_status(RetuneResult(status="complete"), asked)
    # Then: partial or missing accounting cannot be called a completed B40.
    assert status == expected


def test_failed_extra_final_is_not_filtered_out_of_gate():
    # Given: three good blocks alongside an additional failed block.
    rows = core_rows()
    h = next(r for r in rows[0].opportunities if r.arm == "H")
    h.finals.append(h.finals[0].model_copy(update={"status": "fail"}))
    # When: summarize the malformed final evidence without selecting good blocks.
    summary = method_module("gates").summarize_core(rows)
    # Then: no positive global gate can be manufactured by dropping the failure.
    assert summary.status == "inconclusive" and summary.branch.phase != "A"
