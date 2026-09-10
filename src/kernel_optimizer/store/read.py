"""The authoritative reader for a finished run's events, sources and kernel metadata (G21).

WHY THIS EXISTS. Every probe and audit script hand-wrote its own field paths into `events.jsonl`, and
a wrong path does not raise -- it reads `None`, the code takes the empty branch, and the script prints
a clean, plausible, WRONG table. "I read it wrong" and "there is nothing there" are indistinguishable.

Four instances in a single day, all of this shape:

    guessed                          actual                              symptom
    payload.status                   payload.trial.status                0 of 680 trials found
    latency_ms.median_ms             latency_ms.median                   every latency read as None
    candidates/*.py                  candidates/<id>/source.py           10 sources read as absent
    compiled.shared                  compiled.metadata.shared            a probe reported a healthy
                                                                         box as blind to Tier-1

The second one is the clearest: a run holding 680 TRIAL_DONE events, 371 of them complete, reported
"no complete trial with a latency in this run". The fourth is the most dangerous, because its output
was a defect that did not exist -- it invented work and cast doubt on a working machine.

THE FIX IS NOT THE WRAPPING. Collecting these paths into functions only moves the hazard if the
functions still return `[]`. The actual fix is that AN EMPTY RESULT RAISES. Every one of the four
failures was an empty result accepted as an answer, so emptiness must be a decision the caller writes
down (`allow_empty=True`) rather than a silence they inherit. That is the whole point of this module;
the shared field paths are a side benefit.

SCOPE. New code should use this. The ~75 existing scripts are deliberately NOT migrated: most of
their conclusions are already recorded in docs, re-running them buys nothing, and a mass edit risks
more than it fixes. Migrate one when you next need to trust its output -- which is exactly when you
want to know whether it was reading correctly.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any


class EmptyResult(RuntimeError):
    """A read that matched nothing.

    Raised instead of returning an empty container, because the failure this module exists to prevent
    is precisely an empty container being mistaken for a finding. The message names what was searched
    and what was found, so the reader can tell a bad path from a genuinely empty run.
    """


class ReadError(RuntimeError):
    """The run directory or a file inside it is not readable/parseable."""


# ---------------------------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------------------------


def read_events(run_dir: Path | str) -> list[dict]:
    """Every event in a run, in order. Raises if the log is missing or holds no events."""
    run = Path(run_dir)
    path = run / "events.jsonl"
    if not path.is_file():
        raise ReadError(f"no events.jsonl in {run}")
    events: list[dict] = []
    bad = 0
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial last line is normal for a run still in flight; a log full of them is
                # not, and the count is reported rather than hidden.
                bad += 1
    if not events:
        raise EmptyResult(f"{path} parsed to zero events ({bad} unparseable lines)")
    return events


def events_of_type(run_dir: Path | str, event_type: str, *, allow_empty: bool = False) -> list[dict]:
    """All events of one type.

    `allow_empty` must be passed explicitly to accept none. Without it, a type that never occurred
    raises -- which is what turns "Loop D never ran" from an empty list into a statement.
    """
    events = read_events(run_dir)
    hits = [e for e in events if e.get("type") == event_type]
    if not hits and not allow_empty:
        present = sorted({e.get("type", "?") for e in events})
        raise EmptyResult(
            f"no {event_type!r} events in {run_dir} ({len(events)} events present, types: "
            f"{present}). Pass allow_empty=True if their absence is the finding."
        )
    return hits


# ---------------------------------------------------------------------------------------------
# trials
# ---------------------------------------------------------------------------------------------

# The one place that knows where a trial's fields live. `TRIAL_DONE` nests the record under
# `payload.trial`; reading `payload.status` finds nothing and silently reports zero trials.
_TRIAL_KEY = "trial"

# The latency keys as actually stored. Verified against a real run: the stored dict is
# {max, mean, median, min, n_samples, samples, std} -- there is NO `_ms` suffix on any of them, and
# no `robust_ms`, which is a name from the in-memory model rather than the log.
_LATENCY_PREFERENCE = ("median", "mean")


def trial_of(event: dict) -> dict:
    """The trial record inside a TRIAL_DONE event."""
    return (event.get("payload") or {}).get(_TRIAL_KEY) or {}


def latency_ms_of(trial: dict) -> float | None:
    """One trial's representative latency in ms, or None if it has none.

    Median first, by project decision: measured rank-correctness 93.2% for median against 64.8% for
    mean over 20 samples, and `min` must never be used -- it judges backwards. Returning None is
    legitimate here (a failed trial has no latency), which is why this one does not raise; the
    raising happens at the collection level, where "no trial has a latency" IS suspicious.
    """
    lat = trial.get("latency_ms") or {}
    for key in _LATENCY_PREFERENCE:
        v = lat.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def completed_trials(run_dir: Path | str, *, allow_empty: bool = False) -> list[dict]:
    """Every trial that completed, as trial records (not events).

    Raises when a run holds TRIAL_DONE events but none of them read as complete -- the exact
    signature of a wrong field path, and the failure that reported 0 of 680.
    """
    done = events_of_type(run_dir, "TRIAL_DONE", allow_empty=allow_empty)
    trials = [trial_of(e) for e in done]
    complete = [t for t in trials if t.get("status") == "complete"]
    if not complete and not allow_empty:
        statuses = sorted({str(t.get("status")) for t in trials})
        raise EmptyResult(
            f"{len(done)} TRIAL_DONE events in {run_dir} but none with status=='complete' "
            f"(statuses seen: {statuses}). If every trial genuinely failed, pass allow_empty=True; "
            f"otherwise the field path is wrong -- the record is nested under payload.{_TRIAL_KEY}."
        )
    return complete


def trials_with_latency(run_dir: Path | str, *, allow_empty: bool = False) -> list[dict]:
    """Completed trials that carry a latency, each with `latency_ms_value` attached."""
    out = []
    for t in completed_trials(run_dir, allow_empty=allow_empty):
        ms = latency_ms_of(t)
        if ms is not None:
            out.append({**t, "latency_ms_value": ms})
    if not out and not allow_empty:
        raise EmptyResult(
            f"no completed trial in {run_dir} carries a latency. The stored keys are "
            f"{_LATENCY_PREFERENCE!r} -- NOT median_ms/robust_ms, which are model names and read as "
            f"None from the log."
        )
    return out


def best_trial(run_dir: Path | str) -> dict:
    """The completed trial with the lowest latency. Raises if there is none."""
    return min(trials_with_latency(run_dir), key=lambda t: t["latency_ms_value"])


# ---------------------------------------------------------------------------------------------
# candidate sources
# ---------------------------------------------------------------------------------------------


def candidate_ids(run_dir: Path | str, *, allow_empty: bool = False) -> list[str]:
    """Candidate ids that have a directory on disk."""
    cdir = Path(run_dir) / "candidates"
    if not cdir.is_dir():
        raise ReadError(f"no candidates/ directory in {run_dir}")
    ids = sorted(p.name for p in cdir.iterdir() if p.is_dir())
    if not ids and not allow_empty:
        raise EmptyResult(f"{cdir} contains no candidate directories")
    return ids


def candidate_source(run_dir: Path | str, candidate_id: str) -> str:
    """One candidate's source text.

    The layout is `candidates/<candidate_id>/source.py` -- a DIRECTORY per candidate, holding the
    source plus its two witnesses and its trials. A `candidates/*.py` glob matches nothing and reads
    as "no source on disk" on a run that has ten of them.
    """
    path = Path(run_dir) / "candidates" / candidate_id / "source.py"
    if not path.is_file():
        avail = candidate_ids(run_dir, allow_empty=True)
        raise ReadError(
            f"no source.py for {candidate_id!r} at {path}. Candidates present: {avail}. "
            f"The layout is candidates/<id>/source.py, not candidates/<id>.py."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# kernel metadata
# ---------------------------------------------------------------------------------------------

# Which object carries which field. Split across two objects on a compiled Triton kernel, and
# guessing wrong produced a probe that declared a healthy box blind to Tier-1 signals.
_ON_KERNEL = ("n_regs", "n_spills")
_ON_METADATA = ("shared", "num_warps", "num_stages")


def kernel_metadata(compiled: Any) -> dict:
    """Resource metadata off a compiled Triton kernel, from WHICHEVER object carries it.

    `n_regs`/`n_spills` hang off the kernel; `shared`/`num_warps`/`num_stages` live on
    `compiled.metadata`. Both objects are searched for every field, so a Triton version that moves
    one does not produce a false "unavailable".

    Returns a dict plus `_missing`: the fields found on NEITHER object. Only that list justifies
    saying a box is blind -- the probe that reported the A800 blind had simply read the wrong object,
    and reporting a working machine as broken is worse than reporting nothing.

    `shared == 0` is a real value (a kernel that allocates no shared memory), so absence is
    represented by None and never conflated with zero.
    """
    meta_obj = getattr(compiled, "metadata", None)
    out: dict[str, Any] = {}
    missing: list[str] = []
    for name in (*_ON_KERNEL, *_ON_METADATA):
        v = getattr(compiled, name, None)
        if v is None and meta_obj is not None:
            v = getattr(meta_obj, name, None)
        # Triton has used both `n_spills` and `num_spills` across versions.
        if v is None and name == "n_spills":
            v = getattr(compiled, "num_spills", None)
            if v is None and meta_obj is not None:
                v = getattr(meta_obj, "num_spills", None)
        out[name] = v
        if v is None:
            missing.append(name)
    out["_missing"] = missing
    return out
