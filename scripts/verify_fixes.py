"""Check every fix landed since the last experiment against a finished run's events.

Reads only on-disk events.jsonl + jobs/ mtimes (never notification text). Each check
prints PASS / FAIL / N/A with the evidence it used, so a claim in the write-up can be
traced back to a specific event.

Usage:  python scripts/verify_fixes.py runs/run-l3-48-20260905-003307 [baseline_run]
"""

from __future__ import annotations

import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path


def load(run: Path) -> list[dict]:
    path = run / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def job_wall_times(run: Path) -> dict[str, list[float]]:
    """Wall time per GPU job, from the mtime gap between the job spec and its result."""
    out: dict[str, list[float]] = collections.defaultdict(list)
    jobs = run / "jobs"
    if not jobs.is_dir():
        return out
    for spec in jobs.glob("*.json"):
        if spec.name.endswith(".out.json"):
            continue
        res = spec.with_suffix("").with_suffix(".out.json")
        if not res.exists():
            res = jobs / (spec.stem + ".out.json")
        if not res.exists():
            continue
        dt = os.path.getmtime(res) - os.path.getmtime(spec)
        if 0 < dt < 3600:
            kind = spec.stem.rsplit("-", 1)[0]
            out[kind].append(dt)
    return out


def say(name: str, ok: bool | None, detail: str) -> None:
    tag = "PASS" if ok else ("FAIL" if ok is False else " N/A")
    print(f"[{tag}] {name}\n       {detail}")


def check_speedup(run: Path, base: Path | None) -> None:
    """Fix A: venv on ext4 instead of the 9p /mnt mount."""
    new = job_wall_times(run)
    if not new:
        say("A  ext4 venv (per-job wall time)", None, "no job files yet")
        return
    allt = [t for v in new.values() for t in v]
    med = statistics.median(allt)
    detail = f"{len(allt)} jobs, median {med:.1f}s"
    if base is not None and (base / "jobs").is_dir():
        old = job_wall_times(base)
        shared = sorted(set(new) & set(old))
        parts = [
            f"{k}: {statistics.median(old[k]):.1f}s -> {statistics.median(new[k]):.1f}s"
            for k in shared[:6]
        ]
        if parts:
            detail += " | vs baseline run -- " + "; ".join(parts)
    say("A  ext4 venv (per-job wall time)", med < 12.0, detail)


def check_prefetch(rows: list[dict]) -> None:
    """Fix B1: parameterizer calls prefetched onto a background thread must overlap GPU work."""
    # One span per call_id. A call with retries emits AGENT_CALL_FAILED per attempt and
    # then AGENT_CALL_FINISHED, all sharing one start; pairing every terminal event with
    # that start would fabricate self-overlap, so keep only the LAST terminal event.
    starts = {e["payload"]["call_id"]: e["ts"] for e in rows if e["type"] == "AGENT_CALL_STARTED"}
    ends: dict[str, tuple[float, str]] = {}
    for e in rows:
        if e["type"] in ("AGENT_CALL_FINISHED", "AGENT_CALL_FAILED"):
            cid = e["payload"].get("call_id")
            if cid in starts:
                ends[cid] = (e["ts"], e["payload"].get("module", "?"))
    spans = [(starts[c], t, m) for c, (t, m) in ends.items()]
    if not spans:
        say("B1 agent/GPU overlap (prefetch)", None, "no completed agent calls yet")
        return
    spans.sort()
    overlap = 0.0
    pairs = 0
    for i, (s1, e1, m1) in enumerate(spans):
        for s2, e2, m2 in spans[i + 1:]:
            if s2 >= e1:
                break
            ov = min(e1, e2) - s2
            if ov > 0:
                overlap += ov
                pairs += 1
    say(
        "B1 agent/GPU overlap (prefetch)",
        pairs > 0,
        f"{len(spans)} agent calls, {pairs} overlapping pairs, {overlap / 60:.1f} min concurrent"
        " (was 0.00h before B1)",
    )


def check_k_bugs(rows: list[dict]) -> None:
    """K expansion bugs: degenerate_domain and constraint_invalid must no longer appear."""
    bad = collections.Counter()
    for e in rows:
        if e["type"] in ("SPACE_REJECTED", "SPACE_EXPANSION_REJECTED"):
            bad[e["payload"].get("reason", "?")] += 1
    hits = bad.get("degenerate_domain", 0) + bad.get("constraint_invalid", 0)
    exp = sum(1 for e in rows if e["type"] == "SPACE_EXPANDED")
    say(
        "K  no degenerate_domain / constraint_invalid",
        hits == 0,
        f"{exp} expansions; rejection reasons = {dict(bad) or 'none'}",
    )


def check_expansion_regression(rows: list[dict]) -> None:
    """K re-tune must never lose the prior optimum (L3:43 cand-0c3b5820: 20.0 -> 22.6ms)."""
    per_cand: dict[str, list[float]] = collections.defaultdict(list)
    for e in rows:
        if e["type"] == "TUNING_DONE":
            p = e["payload"]
            cid = p.get("candidate_id") or p.get("space", {}).get("candidate_id")
            ms = p.get("best_ms") or p.get("best_latency_ms")
            if cid and ms:
                per_cand[cid].append(float(ms))
    regressions = {c: v for c, v in per_cand.items() if len(v) > 1 and v[-1] > min(v) * 1.001}
    say(
        "K  expansion keeps prior optimum",
        not regressions,
        f"{len(per_cand)} candidates tuned; regressions = {regressions or 'none'}",
    )


def check_early_pruning(rows: list[dict]) -> None:
    """Anti-early-pruning: no active family may be frozen with 0 rewrite rounds."""
    fin = next((e for e in rows if e["type"] == "RUN_FINISHED"), None)
    if not fin:
        say("F  no family frozen with 0 rewrite rounds", None, "run not finished")
        return
    fams = fin["payload"]["summary"]["families"]
    rounds = collections.Counter()
    for e in rows:
        if e["type"] == "FAMILY_ROUND_RECORDED":
            rounds[e["payload"]["family_id"]] += 1
    unexplored = [f for f in fams if rounds[f] == 0]
    detail = "; ".join(f"{f}: {rounds[f]} rounds, {fams[f]['status']}" for f in fams)
    say("F  no family frozen with 0 rewrite rounds", not unexplored, detail)


def check_antihack(rows: list[dict]) -> None:
    """The excessive_speedup guard must flag, not reject, a verified-correct kernel.

    A trial FAILED as excessive_speedup means the old semantics were in force (or the
    candidate genuinely failed correctness too). Since worker jobs are one-shot
    subprocesses that re-import worker_main, a mid-run fix changes semantics partway
    through, so report when the last such failure occurred rather than just the count.
    """
    fin = next((e for e in rows if e["type"] == "RUN_FINISHED"), None)
    kinds: collections.Counter = collections.Counter()
    last_xs = None
    for e in rows:
        if e["type"] == "TRIAL_DONE":
            fk = e["payload"]["trial"].get("failure_kind")
            if fk:
                kinds[fk] += 1
            if fk == "excessive_speedup":
                last_xs = e["ts"]
    xs = kinds.get("excessive_speedup", 0)
    detail = f"trial failure kinds = {dict(kinds) or 'none'}"
    verdict: bool | None = None
    if xs:
        when = time.strftime("%H:%M:%S", time.localtime(last_xs)) if last_xs else "?"
        detail += (f"; {xs} trial(s) HARD-FAILED for speed (last at {when}) -- expected"
                   " only from jobs that ran before the guard fix, or that also failed"
                   " correctness")
        verdict = False  # a hard fail for speed alone is the bug this check watches
    else:
        detail += ("; no trial hard-failed for speed. NOTE this is vacuous unless the"
                   " run actually produced a >=10x candidate -- check for"
                   " excessive_speedup_note in jobs/*.out.json to confirm the flag path"
                   " ran")
    if fin:
        flag = fin["payload"]["summary"]["best"].get("excessive_speedup_flag")
        detail += f"; best.excessive_speedup_flag={flag}"
        verdict = verdict if verdict is not None else True
    say("G  speed guard flags rather than rejects", verdict, detail)


def check_reeval_gap(rows: list[dict]) -> None:
    """Honest verdict must come from final_reeval_ms, not tuned_ms."""
    fin = next((e for e in rows if e["type"] == "RUN_FINISHED"), None)
    if not fin:
        say("H  honest verdict uses final_reeval_ms", None, "run not finished")
        return
    b = fin["payload"]["summary"]["best"]
    tuned, re_ms = b.get("tuned_ms"), b.get("final_reeval_ms")
    hv = b.get("honest_verdict") or {}
    gap = (re_ms - tuned) / tuned * 100 if tuned and re_ms else None
    say(
        "H  honest verdict uses final_reeval_ms",
        hv.get("same_precision_speedup") is not None,
        f"tuned={tuned}ms reeval={re_ms}ms (gap {gap:+.1f}%); "
        f"{hv.get('candidate_precision')} vs {hv.get('compared_against')} = "
        f"{hv.get('same_precision_speedup')}x, beats={hv.get('beats_same_precision_baseline')}",
    )


def check_error_excerpt(run: Path, rows: list[dict]) -> None:
    """Repair feedback must keep the traceback TAIL (the diagnosis), not the head."""
    tails = [
        e["payload"].get("detail", "")
        for e in rows
        if e["type"] in ("SPACE_REJECTED", "SPACE_EXPANSION_REJECTED")
    ]
    if not tails:
        say("E  error excerpt keeps traceback tail", None, "no rejections in this run")
        return
    long = [t for t in tails if len(t) > 1200]
    elided = [t for t in tails if "chars elided" in t]
    say(
        "E  error excerpt keeps traceback tail",
        not long or bool(elided),
        f"{len(tails)} rejection details, {len(elided)} elided-with-tail, "
        f"max len {max(len(t) for t in tails)}",
    )


def check_reused_journalled(rows: list[dict]) -> None:
    """Reused measurements (witness anchors, the carried-over pre-expansion optimum)
    must appear as TRIAL_DONE, or replay re-runs them and the report cannot see them."""
    reused = [e for e in rows if e["type"] == "TRIAL_DONE"
              and e["payload"].get("reused_measurement")]
    expansions = sum(1 for e in rows if e["type"] == "SPACE_EXPANDED")
    published = sum(1 for e in rows if e["type"] == "SPACE_PUBLISHED")
    if published == 0:
        say("J  reused measurements journalled", None, "no spaces published yet")
        return
    # Every published space anchors its two witnesses, so a run that tuned at all
    # should show reused records once the fix is in force.
    say(
        "J  reused measurements journalled",
        bool(reused),
        f"{len(reused)} reused-measurement trials across {published} space(s), "
        f"{expansions} expansion(s)"
        + ("" if reused else " -- none present: driver predates the fix (orchestrator"
                             " changes need a fresh run, unlike worker-side fixes)"),
    )


def main() -> None:
    run = Path(sys.argv[1])
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    rows = load(run)
    print(f"=== {run.name} : {len(rows)} events ===\n")
    check_speedup(run, base)
    check_prefetch(rows)
    check_k_bugs(rows)
    check_expansion_regression(rows)
    check_early_pruning(rows)
    check_antihack(rows)
    check_reeval_gap(rows)
    check_error_excerpt(run, rows)
    check_reused_journalled(rows)


if __name__ == "__main__":
    main()
