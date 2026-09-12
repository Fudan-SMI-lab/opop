"""D8 step 1. Every GPU job records how long it took and whether it was killed.

WHY THIS IS THE FIRST STEP AND NOT THE CHEAPEST ONE. The plan was to derive a per-compile gate
from the existing `compile_s`. That field reads **p50 0.3 s / max 1.6 s** over 198 profiles, while
`ptxas` on the same run was observed running **25:35**. The reason: `compile_s` is written by the
worker when the job RETURNS, and the six trials that paid 20-50 minutes all hit the deadline and
wrote no `out.json` -- so they have no profile at all. On-disk confirmation: every trial without a
profile is a failure, and that candidate's are exactly 6 timeout + 1 runtime_error + 7 infeasible.

**The metric is censored precisely on the samples that would price it**, so any threshold derived
from it is both too low and suspiciously clean. Hence the instrumentation must land first, and it
must be stamped by the HOST at every exit -- including the timeout -- because a killed job is the
one case guaranteed to have no worker-side record.

WHAT IT MAKES VISIBLE. "One candidate ate 50.8% of the budget" currently needs a throwaway script
against `events.jsonl`. With `job_wall_s` on every trial it is a column in `trials.csv`.

NOT A JUDGEMENT INPUT. Nothing in ranking, allocation or acceptance may read these keys -- "cheap
but slow" must never become a reason to reject a correct kernel. Guarded below by a grep test, the
same discipline `CONVERSION_RATES` is held to.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from kernel_optimizer.config import AppConfig
from kernel_optimizer.gpu import worker_client as wc_mod
from kernel_optimizer.gpu.worker_client import JOB_WALL_KEYS, WslGpuWorker, _with_wall


# --------------------------------------------------------------------- the helper itself


def test_both_keys_are_stamped_with_a_plausible_duration():
    t0 = time.monotonic() - 2.5
    out = _with_wall({"ok": True}, t0, False)

    for k in JOB_WALL_KEYS:
        assert k in out, (k, out)
    assert 2.0 <= out["job_wall_s"] <= 4.0, out["job_wall_s"]
    assert out["job_timed_out"] is False
    assert out["ok"] is True, "the original result must survive intact"


def test_a_worker_supplied_figure_SURVIVES():
    """`setdefault`, not assignment. A worker-reported `job_wall_s` excludes this process's own
    bookkeeping and is therefore closer to the truth; overwriting it would replace a better number
    with a worse one, silently."""
    out = _with_wall({"ok": True, "job_wall_s": 7.5, "job_timed_out": True},
                     time.monotonic() - 100.0, False)

    assert out["job_wall_s"] == 7.5, out
    assert out["job_timed_out"] is True, out


def test_the_input_dict_is_not_mutated():
    """The result dict comes from `json.loads` of the worker's output and is also written to disk;
    stamping must not edit the caller's object."""
    src = {"ok": True}
    out = _with_wall(src, time.monotonic(), False)

    assert src == {"ok": True}, src
    assert out is not src


# --------------------------------------------------------------------- all four exits


def _worker(tmp_path: Path) -> WslGpuWorker:
    cfg = AppConfig()
    cfg.gpu.concurrency.enabled = False
    cfg.gpu.concurrency.timing_cooldown_s = 0.0
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    w = object.__new__(WslGpuWorker)
    w.cfg = cfg.wsl
    w.conc = cfg.gpu.concurrency
    w.jobs_dir = jobs
    w.worker_main_path = tmp_path / "worker_main.py"
    w.worker_main_path.write_text("# stub\n", encoding="utf-8")

    class _Lock:
        def acquire(self, mode):  # noqa: ANN001, ARG002
            return None

        def release(self, mode):  # noqa: ANN001, ARG002
            return None

    w.lock = _Lock()
    return w


class _Proc:
    """A `Popen` stand-in that reproduces one of the four exits."""

    returncode = 3

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.pid = 4242
        self._reaped = False

    def communicate(self, timeout=None):  # noqa: ANN001, ARG002
        if self.mode == "timeout" and not self._reaped:
            self._reaped = True
            raise subprocess.TimeoutExpired(cmd="stub", timeout=timeout or 0)
        return b"", b"stderr tail here"


def _run(tmp_path: Path, monkeypatch, mode: str, out_content: str | None,
         timeout_s: float = 5.0) -> dict[str, Any]:
    w = _worker(tmp_path)
    monkeypatch.setattr(wc_mod.subprocess, "Popen", lambda *a, **k: _Proc(mode))
    monkeypatch.setattr(w, "_kill_job", lambda proc: None)
    monkeypatch.setattr(w, "_build_command", lambda j, o: (["true"], {}))
    if out_content is not None:
        # `run_job` derives the out path from the job id, which contains a uuid -- so intercept the
        # write by pre-creating it from inside a patched `_build_command`. Simpler: patch Path.exists
        # and read_text for the one file it will look at.
        real_exists = Path.exists
        real_read = Path.read_text

        def fake_exists(self):  # noqa: ANN001
            if self.name.endswith(".out.json"):
                return True
            return real_exists(self)

        def fake_read(self, *a, **k):  # noqa: ANN001
            if self.name.endswith(".out.json"):
                return out_content
            return real_read(self, *a, **k)

        monkeypatch.setattr(Path, "exists", fake_exists)
        monkeypatch.setattr(Path, "read_text", fake_read)
    return w.run_job({"kind": "eval"}, timeout_s, "tag")


def test_exit_timeout_is_stamped_AND_flagged(tmp_path, monkeypatch):
    """THE REASON THE WHOLE CHANGE EXISTS. A killed job has no worker-side record at all, so if
    only this exit were missed the instrumentation would be blind to exactly the population it was
    built for. This test must fail on the unfixed code."""
    got = _run(tmp_path, monkeypatch, "timeout", None, timeout_s=5.0)

    assert got["failure_kind"] == "timeout", got
    assert got["job_timed_out"] is True, (
        "a job killed at its deadline must SAY so: %r" % got)
    assert got["job_wall_s"] is not None and got["job_wall_s"] >= 0.0, got


def test_exit_no_out_json_is_stamped(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch, "normal", None)

    assert got["failure_kind"] == "worker_crash", got
    assert got["job_timed_out"] is False, got
    assert got["job_wall_s"] is not None, got


def test_exit_normal_is_stamped(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch, "normal", '{"ok": true, "correct": true}')

    assert got["ok"] is True, got
    assert got["job_timed_out"] is False, got
    assert got["job_wall_s"] is not None, got


def test_exit_unparseable_is_stamped(tmp_path, monkeypatch):
    got = _run(tmp_path, monkeypatch, "normal", "{not json")

    assert got["failure_kind"] == "worker_crash", got
    assert "unparseable" in got["log_tail"], got
    assert got["job_timed_out"] is False, got
    assert got["job_wall_s"] is not None, got


def test_every_run_job_exit_goes_through_the_stamp():
    """A structural guard, because "I applied it at four exits" is exactly the kind of claim that
    silently becomes three when a fifth exit is added. Scan `run_job`'s body for `return`
    statements and require each to be a `_with_wall(...)` call."""
    src = Path(wc_mod.__file__).read_text(encoding="utf-8")
    body = src.split("def run_job(", 1)[1].split("\n    def ", 1)[0]
    returns = re.findall(r"^\s+return (.*)$", body, re.M)
    assert returns, "no returns found in run_job -- the scan is broken"
    bad = [r for r in returns if not r.startswith("_with_wall(")]
    assert not bad, "every run_job exit must be stamped; these are not: %r" % bad
    assert len(returns) == 4, (
        "four exits are expected (timeout / no out.json / normal / unparseable); found %d -- "
        "if an exit was added it needs the stamp too: %r" % (len(returns), returns))


# --------------------------------------------------------------------- the model round-trip


def test_the_field_SURVIVES_a_real_TrialRecord_round_trip():
    """THE IDENTIFIED TRAP. `ProfileRecord` sets only `frozen=True`, so it inherits pydantic's
    default `extra="ignore"` and would drop an unknown key **silently**. `TrialRecord` is a plain
    `BaseModel` with the same default. So the stamp must be an EXPLICIT field, and the guard has to
    be a round-trip through the real model rather than a dict assertion."""
    from kernel_optimizer.models.core import ParamSet, TrialRecord

    rec = TrialRecord(trial_id="t1", candidate_id="c1", space_id="s1",
                      params=ParamSet(values={"BLOCK_M": 32}), status="fail",
                      failure_kind="timeout", job_wall_s=1503.25, job_timed_out=True)

    back = TrialRecord.model_validate(rec.model_dump())
    assert back.job_wall_s == pytest.approx(1503.25)
    assert back.job_timed_out is True
    assert "job_wall_s" in rec.model_dump(), "it must reach the event log, not just memory"


def test_an_unknown_key_would_have_been_dropped_silently():
    """The control for the test above: prove the trap is real, so the explicit field is justified
    rather than defensive. `extra="ignore"` accepts and discards -- no error anywhere."""
    from kernel_optimizer.models.core import ParamSet, TrialRecord

    rec = TrialRecord.model_validate({
        "trial_id": "t1", "candidate_id": "c1", "space_id": "s1",
        "params": ParamSet(values={"BLOCK_M": 32}).model_dump(), "status": "fail",
        "not_a_declared_field": 99.0,
    })

    assert not hasattr(rec, "not_a_declared_field")
    assert "not_a_declared_field" not in rec.model_dump(), (
        "pydantic silently dropped it -- which is why job_wall_s is a declared field")


def test_the_two_paths_that_never_launched_a_job_report_None_not_zero():
    """"Not measured" and "0 seconds" must stay distinguishable. A materialize error and a screen
    refusal genuinely cost no GPU time; reporting 0.0 would be right by accident and wrong in
    shape, and would pollute any distribution computed over the column."""
    from kernel_optimizer.models.core import ParamSet, TrialRecord

    rec = TrialRecord(trial_id="t1", candidate_id="c1", space_id="s1",
                      params=ParamSet(values={"BLOCK_M": 32}), status="fail",
                      failure_kind="materialize_error")

    assert rec.job_wall_s is None
    assert rec.job_timed_out is None


# --------------------------------------------------------------------- not a judgement input


def test_no_selection_path_reads_the_cost_fields():
    """The discipline, enforced structurally. Ranking, allocation and acceptance must not read
    these keys, or a slow compile becomes a reason to reject a correct kernel. Same guard shape as
    `CONVERSION_RATES`'s.

    Allowed readers: the stamp itself, the model declaration, the trial constructor that copies it
    out of the result dict, and the reporting layer.
    """
    root = Path(wc_mod.__file__).resolve().parents[1]
    allowed = {
        "gpu/worker_client.py",        # writes it
        "models/core.py",              # declares it
        "control/orchestrator.py",     # copies result -> TrialRecord
        "reporting/report.py",         # reports it
        "reporting/completeness.py",   # reports it
    }
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in allowed or "__pycache__" in rel:
            continue
        text = path.read_text(encoding="utf-8")
        for key in JOB_WALL_KEYS:
            if key in text:
                offenders.append(f"{rel}: {key}")
    assert not offenders, (
        "these read a cost field outside the write/report paths: %r" % offenders)


def test_the_orchestrator_only_copies_it_and_never_compares_it():
    """The one production consumer is a field copy. A comparison there would be the actual
    failure -- it is the module that ranks trials, so `job_wall_s < x` anywhere in it is a
    selection rule wearing a measurement's clothes.

    Reads the file rather than importing the module: `orchestrator` imports optuna, which the
    Windows host does not have by design, and an `importorskip` here would turn the guard into a
    skip on the machine where this change is written -- and `a-skip-is-not-a-verdict`. Source text
    is the right level for this assertion anyway, since the claim is about what the code says.
    """
    path = Path(wc_mod.__file__).resolve().parents[1] / "control" / "orchestrator.py"
    src = path.read_text(encoding="utf-8")
    bad: list[str] = []
    for key in JOB_WALL_KEYS:
        for m in re.finditer(re.escape(key), src):
            start = src.rfind("\n", 0, m.start()) + 1
            line = src[start:src.find("\n", m.end())].strip()
            if line.startswith("#"):
                continue                                   # a comment explaining it
            if f'result.get("{key}")' in line:
                continue                                   # the copy into TrialRecord
            bad.append(line)
    assert not bad, (
        "the orchestrator may only COPY these fields out of a result dict, never act on them: %r"
        % bad)
