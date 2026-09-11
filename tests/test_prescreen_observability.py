"""The prescreen's own cost must be visible in the log, not inferred from an idle GPU.

Found by measurement, not by reading code. Box 3 sat 15 minutes with the GPU at 0% while `ptxas`
chewed 99.5% of one core on 152,185 lines of PTX. Two `SPACE_PRESCREENED` entries sat at 1200.5 s and
1201.0 s -- exactly `build_timeout_s` -- and each reported `infeasible: 0`, which is BYTE-IDENTICAL to
a fast batch that legitimately found nothing to reject. A 20-minute batch that answered nothing looked
the same as a 0.5-second one.

`prescreen_batch` swallows its own timeout by design (a screen must never be a verdict), so the
`PRESCREEN_FAILED` path above it cannot fire for a timeout. This event is the only place the
difference can be recorded.

The distinction that matters is ANSWERED vs INFEASIBLE:
  * `infeasible: 0, answered: 40` -- the batch ran and nothing was rejectable. Nothing is wrong.
  * `infeasible: 0, answered: 0`  -- the batch produced no information at all. That is the timeout.

These tests drive the REAL `_prescreen_space`, including its own sampling, materialization and file
writing; only the evaluator and the clock are substituted. A test that recomputed `answered` itself
would pass on an orchestrator that never emitted the field -- the recorded
`a-test-that-copies-the-loop-does-not-test-it` failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kernel_optimizer.config import AppConfig
from kernel_optimizer.models.core import Candidate, ParamDomain, ParameterSpace, TaskSpec

_SOURCE = '''
PARAMS = {"BLOCK_M": 32}

import torch


class ModelNew(torch.nn.Module):
    def forward(self, x):
        return x * PARAMS["BLOCK_M"]
'''


class _Store:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def append(self, type_: str, payload: dict) -> None:
        self.events.append((type_, payload))

    def of(self, type_: str) -> list[dict]:
        return [p for t, p in self.events if t == type_]


class _Evaluator:
    """Behaves like the real evaluator on this path: the batch is silent on failure, and verdicts come
    only from the cache. `by_index` keys the verdict on the config's ordinal so a test can say "the
    first two were answered, the rest were not" without knowing what the sampler produced."""

    def __init__(self, by_index: dict[int, Any] | None = None, answer_all: Any = None) -> None:
        self.by_index = by_index or {}
        self.answer_all = answer_all
        self.batches = 0
        self._order: list[str] = []

    def prescreen_batch(self, task, paths, tag, backend) -> None:  # noqa: ANN001, ARG002
        self.batches += 1
        self._order = [Path(p).read_text(encoding="utf-8") for p in paths]

    def cached_shared_verdict(self, src: str, backend: str, cap: int):  # noqa: ANN001, ARG002
        if self.answer_all is not None:
            return self.answer_all
        try:
            return self.by_index.get(self._order.index(src))
        except ValueError:
            return None


def _orch(evaluator: _Evaluator, store: _Store, monkeypatch, elapsed: float):
    """A real Orchestrator carrying only the collaborators this path touches.

    The clock is advanced by patching `time.time` in the orchestrator's own module, so the recorded
    `elapsed_s` is the production code measuring a controlled clock rather than a real sleep.
    """
    from kernel_optimizer.control import orchestrator as orch_mod

    cfg = AppConfig()
    cfg.evaluation.build_timeout_s = 1200.0
    cfg.budgets.trials_per_space = 16          # keep the sampler's work small
    cfg.gpu.compile_screen_enabled = True

    calls = {"n": 0}

    def fake_time() -> float:
        calls["n"] += 1
        return 1000.0 if calls["n"] == 1 else 1000.0 + elapsed

    monkeypatch.setattr(orch_mod.time, "time", fake_time)

    o = object.__new__(orch_mod.Orchestrator)
    o.cfg = cfg
    o.store = store
    o.deps = type("D", (), {"evaluator": evaluator})()
    o.task = TaskSpec(task_id="level3:48", level=3, problem_id=48, name="t",
                      ref_path=Path("ref.py"))
    return o


def _crun(tmp_path: Path):
    """A CandidateRun with one two-value knob, so the sampler finds exactly 2 legal configurations."""
    from kernel_optimizer.control.orchestrator import CandidateRun

    cand = Candidate(candidate_id="cand-1", family_id="fam-1", origin="seed", backend="triton",
                     source_sha="0" * 64, structural_signature="s", approach_summary="a")
    space = ParameterSpace(
        space_id="sp-1", candidate_id="cand-1", version=1, source_sha="0" * 64,
        domains=[ParamDomain(name="BLOCK_M", kind="int", choices=[32, 64])],
        constraints=[],
    )
    crun = object.__new__(CandidateRun)
    crun.candidate = cand
    crun.space = space
    crun.source = _SOURCE
    return crun


def _run(o, tmp_path: Path) -> None:
    d = tmp_path / "trials"
    d.mkdir(exist_ok=True)
    o._prescreen_space(_crun(tmp_path), d)


def test_a_prescreen_that_answers_nothing_is_distinguishable_from_one_that_finds_nothing(
        tmp_path, monkeypatch):
    """The whole point. `infeasible: 0` alone cannot tell a 1200 s timeout from a 0.5 s clean batch,
    and on box 3 it did not: the 1200.5/1201.0 s entries were indistinguishable from a 490 s entry
    that found 17.
    """
    store, ev = _Store(), _Evaluator(answer_all=None)      # nothing cached, as after a timeout
    _run(_orch(ev, store, monkeypatch, elapsed=1200.5), tmp_path)

    got = store.of("SPACE_PRESCREENED")
    assert len(got) == 1, store.events
    p = got[0]
    assert p["configs_probed"] == 2, p
    assert p["infeasible"] == 0
    assert p["answered"] == 0, "a batch that cached no verdict must report answered=0"
    assert p["timed_out"] is True, (
        "1200.5 s against build_timeout_s=1200 with no answers must read as a timeout: %r" % p)
    assert p["elapsed_s"] == pytest.approx(1200.5, abs=0.2)


def test_a_fast_batch_that_finds_nothing_is_NOT_flagged_as_a_timeout(tmp_path, monkeypatch):
    """The false-positive direction, which matters more than the true positive: most batches
    legitimately find nothing to reject, and flagging those would make the field useless."""
    store, ev = _Store(), _Evaluator(answer_all=True)      # every config answered FEASIBLE
    _run(_orch(ev, store, monkeypatch, elapsed=0.4), tmp_path)

    p = store.of("SPACE_PRESCREENED")[0]
    assert p["infeasible"] == 0
    assert p["answered"] == 2, "both configs were answered feasible: %r" % p
    assert p["timed_out"] is False
    assert p["elapsed_s"] == pytest.approx(0.4, abs=0.2)


def test_answered_is_not_the_same_count_as_infeasible(tmp_path, monkeypatch):
    """`answered` must not collapse into `infeasible`: a config answered FEASIBLE is answered and not
    infeasible. Box 3's real seq-104 batch was 40 probed / 17 infeasible."""
    store, ev = _Store(), _Evaluator(by_index={0: False, 1: True})
    _run(_orch(ev, store, monkeypatch, elapsed=490.6), tmp_path)

    p = store.of("SPACE_PRESCREENED")[0]
    assert p["configs_probed"] == 2
    assert p["infeasible"] == 1, "only the False verdict is infeasible: %r" % p
    assert p["answered"] == 2, "False and True are both answers: %r" % p
    assert p["timed_out"] is False, "answered>0 means it produced information, however slow"


def test_a_slow_batch_that_DID_answer_is_not_a_timeout(tmp_path, monkeypatch):
    """Whether `timed_out` means anything: elapsed alone is not enough. A batch can take the full
    timeout and still have cached its answers before the deadline."""
    store, ev = _Store(), _Evaluator(by_index={0: False})
    _run(_orch(ev, store, monkeypatch, elapsed=1201.0), tmp_path)

    p = store.of("SPACE_PRESCREENED")[0]
    assert p["answered"] == 1
    assert p["timed_out"] is False, (
        "a slow batch that produced answers is expensive, not uninformative: %r" % p)


def test_the_screen_can_still_be_switched_off(tmp_path, monkeypatch):
    """The disable path must emit nothing at all -- an event claiming 0 probed configs would put a
    zero in the corpus that reads as a failed screen rather than an absent one."""
    store, ev = _Store(), _Evaluator(answer_all=True)
    o = _orch(ev, store, monkeypatch, elapsed=0.1)
    o.cfg.gpu.compile_screen_enabled = False
    _run(o, tmp_path)
    assert store.of("SPACE_PRESCREENED") == []
    assert ev.batches == 0
