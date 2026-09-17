"""Terminal-only A measurements never add studies or rewrite ready traces."""

import importlib
from pathlib import Path

import pytest

from scripts.experiments.c2_method_protocol import Deadline
from tests.c2_method_a_fakes import a_fixture, prepared
from tests.test_c2_method_a import a_module


def terminal_module():
    assert (Path(__file__).parents[1] / "scripts/experiments/c2_method_a_terminal.py").is_file()
    return importlib.import_module("scripts.experiments.c2_method_a_terminal")


@pytest.mark.parametrize("mode", ["complete", "expired", "mid_deadline"])
def test_terminal_only_measures_frozen_endpoints_with_original_deadline(a_fixture, mode):
    # Given: two real completed A trajectories and their measured-source exports.
    a, terminal = a_module(), terminal_module()
    fixture = a_fixture
    fixture.mode[0] = "fresh_baseline"
    results = []
    for slot in ("A0", "B1"):
        inputs = a.AInputs.model_validate_json(fixture.wrapper(slot).read_text())
        results.append(a.run_a(inputs, a.ARun(fixture.cfg, fixture.root / slot,
            Deadline(0, lambda: fixture.clock[0], 9000), 1000)))
    records = [fixture.root / slot / "result.json" for slot in ("A0", "B1")]
    original_bytes = [p.read_bytes() for p in records]
    artifact_inputs = [terminal.TerminalArm(result=record, source=r.exported_source,
        helper_root=r.exported_helper_root, helpers=r.exported_helpers) for r, record in zip(results, records, strict=True)]
    if mode == "expired":
        artifact_inputs = [spec.model_copy(update={"source": fixture.root / "not-copied.py",
                                                 "helpers": ()}) for spec in artifact_inputs]
    inputs = terminal.TerminalInputs(slot="A0", original=fixture.root / "A0-inputs.json",
                                    g0=artifact_inputs[0], h=artifact_inputs[1])
    before_calls, before_jobs = len(fixture.roles), len(fixture.jobs)
    if mode == "mid_deadline":
        fixture.mode[0] = "deadline"
    # When: terminal finals share the original 150-minute cap, including transfer/wait.
    outcome = terminal.run_terminal(inputs, terminal.TerminalRun(fixture.cfg, fixture.root / "terminal",
        now=lambda: 10001 if mode == "expired" else 2000 + fixture.clock[0], clock=lambda: fixture.clock[0]))
    # Then: either six full blocks or none, no model/study, no backdated ready trace.
    assert len(fixture.roles) == before_calls
    assert [p.read_bytes() for p in records] == original_bytes
    assert len(fixture.jobs) - before_jobs == {"complete": 6, "expired": 0, "mid_deadline": 1}[mode]
    assert outcome.status == ("valid" if mode == "complete" else "censored")
    assert outcome.deadline_unix_s == 10000
    if mode == "complete":
        assert len(outcome.g0_finals) == len(outcome.h_finals) == 3
        assert outcome.normalized_g0.ratio == outcome.normalized_h.ratio == 1.5
        assert outcome.baseline == [2.0] * 3
        from kernel_optimizer.store.run_store import RunStore
        phases = [e.payload["phase"] for e in RunStore.open(fixture.root / "terminal").iter_events()
                  if e.type == "LOCAL_EVAL_STARTED"]
        assert phases == ["terminal_G0_0", "terminal_H_0", "terminal_H_1", "terminal_G0_1", "terminal_G0_2", "terminal_H_2"]


@pytest.mark.parametrize("late", [False, True])
def test_ready_cutoff_uses_own_card_baselines_not_late_terminal(a_fixture, late):
    # Given: H's improved endpoint arrives after the common cutoff.
    assert (Path(__file__).parents[1] / "scripts/experiments/c2_method_a_compare.py").is_file()
    a = a_module()
    fixture = a_fixture
    inputs = a.AInputs.model_validate_json(fixture.wrapper("A0").read_text())
    g0 = a.run_a(inputs, a.ARun(fixture.cfg, fixture.root / "A0", Deadline(0, lambda: 0, 9000), 1000))
    g0 = g0.model_copy(update={"loop": g0.loop.model_copy(update={"elapsed_total_s": 10})})
    h = g0.model_copy(update={"slot": "B1", "arm": "H", "loop": g0.loop.model_copy(update={
        "elapsed_total_s": 20,
        "initial_parent_finals": [t.model_copy(update={"latency_ms": t.latency_ms.model_copy(update={"median": 50})})
                                  for t in g0.loop.initial_parent_finals],
        "rounds": [r.model_copy(update={"elapsed_ready_s": (15 if late else 0) + r.number,
            "fresh_finals": [t.model_copy(update={"latency_ms": t.latency_ms.model_copy(update={"median": 20})})
                             for t in r.fresh_finals]}) for r in g0.loop.rounds]})})
    # When: the CPU ready comparator picks only states available by H_time.
    comparison = importlib.import_module("scripts.experiments.c2_method_a_compare").compare_ready(g0, h)
    # Then: H falls back to its original parent, not its late endpoint.
    assert comparison.h_time_s == 10 and comparison.h.ratio == (1 if late else 0.4)
    assert comparison.g0.ratio == 0.6 and comparison.status == ("fail" if late else "pass")
