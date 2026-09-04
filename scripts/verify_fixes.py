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
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(run: Path) -> list[dict]:
    path = run / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fix_epoch(needle: str, pathspec: str) -> float | None:
    """Commit epoch of the change that introduced `needle` into `pathspec`.

    A check that ignores this turns a propagation window into a false FAIL. Which
    window applies depends on WHERE the fix lives:

      * worker-side (gpu/worker_main.py) reaches a RUNNING experiment immediately --
        every GPU job is a one-shot subprocess that re-imports it. Compare the fix
        epoch against each EVENT's timestamp.
      * driver-side (orchestrator/agents/config) is loaded once at startup, so it
        cannot apply to a run that was already going. Compare against the RUN's start.

    Returns None if git cannot answer, so the caller degrades to N/A rather than
    inventing a verdict.
    """
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "-S", needle, "--", pathspec],
            cwd=REPO, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()
    return float(line[0]) if out.returncode == 0 and line else None


def hhmm(ts: float | None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "?"


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


def check_antihack(run: Path, rows: list[dict]) -> None:
    """The excessive_speedup guard must flag, not reject, a verified-correct kernel.

    Worker-side fix, so it takes effect DURING a run: jobs that started before the
    commit ran the old reject-for-speed semantics and jobs after it ran the new
    accept-and-flag semantics. Counting hard-fails over the whole run therefore
    convicts the fix of its own predecessor's behaviour. Only a hard-fail AFTER the
    commit epoch is evidence against it.
    """
    fin = next((e for e in rows if e["type"] == "RUN_FINISHED"), None)
    fix = fix_epoch("excessive_speedup_note", "src/kernel_optimizer/gpu/worker_main.py")
    kinds: collections.Counter = collections.Counter()
    before, after = [], []
    for e in rows:
        if e["type"] == "TRIAL_DONE":
            fk = e["payload"]["trial"].get("failure_kind")
            if fk:
                kinds[fk] += 1
            if fk == "excessive_speedup":
                (after if fix is not None and e["ts"] >= fix else before).append(e["ts"])
    detail = f"trial failure kinds = {dict(kinds) or 'none'}"
    verdict: bool | None = None
    if fix is None and (before or after):
        detail += "; hard-fails present but git could not date the fix -- undetermined"
    elif after:
        detail += (f"; {len(after)} trial(s) HARD-FAILED for speed AFTER the fix landed"
                   f" at {hhmm(fix)} (last {hhmm(after[-1])}) -- the guard is still"
                   " rejecting, or those candidates also failed correctness")
        verdict = False
    elif before:
        # The expected shape of a mid-run worker fix.
        detail += (f"; {len(before)} hard-fail(s), all BEFORE the fix landed at"
                   f" {hhmm(fix)} (last {hhmm(before[-1])}); 0 after")
        verdict = True
    # The accept path lives in the job output, not in any event: a flagged-but-accepted
    # job produces an ordinary complete trial. Without this the no-hard-fail case is
    # vacuous -- it cannot tell "guard fixed" from "no fast kernel ever appeared".
    notes = 0
    jobs = run / "jobs"
    if jobs.is_dir():
        for f in jobs.glob("*.out.json"):
            try:
                if "excessive_speedup_note" in f.read_text(encoding="utf-8", errors="replace"):
                    notes += 1
            except OSError:
                pass
    if notes:
        detail += f"; {notes} job(s) accepted-and-flagged (excessive_speedup_note)"
        verdict = bool(verdict) if verdict is False else True
    else:
        detail += ("; NO job carries excessive_speedup_note -- the flag path never ran,"
                   " so a clean result here is vacuous")
        if verdict is None:
            verdict = None
    if fin:
        flag = fin["payload"]["summary"]["best"].get("excessive_speedup_flag")
        detail += f"; best.excessive_speedup_flag={flag}"
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


def check_metrics_crash(run: Path, rows: list[dict]) -> None:
    """The relaxed-metrics reporter must not crash the path it explains.

    torch.quantile refuses >~16M elements and this task's output is 134M, so the p99
    line raised RuntimeError and REPLACED the correctness verdict with a traceback --
    the repair agent then diagnosed my diagnostic instead of the kernel. Worker-side,
    so datable against individual events.
    """
    fix = fix_epoch("1_000_000", "src/kernel_optimizer/gpu/worker_main.py")
    hits = [e["ts"] for e in rows
            if e["type"] in ("SPACE_REJECTED", "SPACE_EXPANSION_REJECTED")
            and "quantile" in e["payload"].get("detail", "")]
    after = [t for t in hits if fix is not None and t >= fix]
    if fix is None:
        say("I  metrics reporter does not crash", None, "git could not date the fix")
        return
    detail = f"fix landed {hhmm(fix)}; {len(hits)} rejection(s) mention quantile"
    if after:
        detail += f", {len(after)} AFTER the fix (last {hhmm(after[-1])})"
    elif hits:
        detail += f", all BEFORE it (last {hhmm(hits[-1])}); 0 after"
    else:
        detail += " -- none at all"
    say("I  metrics reporter does not crash", not after, detail)


def check_cosine_overflow(run: Path, rows: list[dict]) -> None:
    """fp32 cosine overflowed to nan when outputs exceed ~1.8e19, silently vetoing
    correct kernels. The fix normalises by max-abs in float64 before the dot product.

    Distinguishing "the overflow bug" from "the kernel really did emit NaN" needs the
    NON_FINITE_OUTPUT line, which landed in a LATER commit than the cosine fix itself.
    Before that line existed a nan cosine is genuinely ambiguous, so only rejections
    after BOTH commits can be scored -- otherwise this check would convict the cosine
    fix whenever a candidate legitimately produced NaN.
    """
    cos_fix = fix_epoch("_cosine_similarity", "src/kernel_optimizer/gpu/worker_main.py")
    nf_fix = fix_epoch("NON_FINITE_OUTPUT", "src/kernel_optimizer/gpu/worker_main.py")
    if cos_fix is None or nf_fix is None:
        say("L  cosine does not overflow to nan", None, "git could not date the fixes")
        return
    scorable = max(cos_fix, nf_fix)
    nan_after, real_after, ambiguous = [], [], []
    for e in rows:
        if e["type"] not in ("SPACE_REJECTED", "SPACE_EXPANSION_REJECTED"):
            continue
        d = e["payload"].get("detail", "")
        if "cosine" not in d:
            continue
        if e["ts"] < scorable:
            if e["ts"] >= cos_fix and "'cosine': 'nan'" in d:
                ambiguous.append(e["ts"])
            continue
        # A nan cosine alongside NON_FINITE_OUTPUT is the candidate's own fault, not
        # the overflow bug -- the kernel really did emit NaN.
        if "'cosine': 'nan'" in d and "NON_FINITE_OUTPUT" not in d:
            nan_after.append(e["ts"])
        elif "'cosine': 0." in d or "'cosine': 1." in d:
            real_after.append(e["ts"])
    detail = (f"scorable after {hhmm(scorable)} (cosine fix {hhmm(cos_fix)},"
              f" non-finite reporting {hhmm(nf_fix)}): {len(real_after)} rejection(s)"
              f" report a real cosine on 1e22-scale output,"
              f" {len(nan_after)} report nan with no non-finite output")
    if ambiguous:
        detail += (f"; {len(ambiguous)} earlier nan(s) not scorable -- no NON_FINITE"
                   " line yet, so overflow and a genuinely-NaN kernel are"
                   " indistinguishable there")
    if not real_after and not nan_after:
        say("L  cosine does not overflow to nan", None,
            detail + "; no scorable rejection, so a clean result here is vacuous")
        return
    say("L  cosine does not overflow to nan", not nan_after, detail)
def check_reused_journalled(rows: list[dict]) -> None:
    """Reused measurements (witness anchors, the carried-over pre-expansion optimum)
    must appear as TRIAL_DONE, or replay re-runs them and the report cannot see them.

    Driver-side fix: the orchestrator is imported once at startup, so a run that began
    before the commit CANNOT exhibit it however long it goes on. Reporting that as FAIL
    blames the fix for a propagation window, so date the run against the commit and
    return N/A when the run predates it.
    """
    reused = [e for e in rows if e["type"] == "TRIAL_DONE"
              and e["payload"].get("reused_measurement")]
    expansions = sum(1 for e in rows if e["type"] == "SPACE_EXPANDED")
    published = sum(1 for e in rows if e["type"] == "SPACE_PUBLISHED")
    if published == 0:
        say("J  reused measurements journalled", None, "no spaces published yet")
        return
    fix = fix_epoch("reused_measurement",
                    "src/kernel_optimizer/control/orchestrator.py")
    run_start = rows[0]["ts"]
    if reused:
        say("J  reused measurements journalled", True,
            f"{len(reused)} reused-measurement trials across {published} space(s), "
            f"{expansions} expansion(s)")
        return
    if fix is not None and run_start < fix:
        # Not a failure: the driver in memory predates the fix. The reuse still HAPPENED
        # (tuning saw its full budget) -- it is only invisible in the event log, which
        # is exactly what the fix addresses for subsequent runs.
        asked = sum(e["payload"]["snapshot"]["asked"]
                    for e in rows if e["type"] == "TUNING_DONE")
        logged = sum(1 for e in rows if e["type"] == "TRIAL_DONE")
        say("J  reused measurements journalled", None,
            f"run started {hhmm(run_start)}, fix committed {hhmm(fix)} -- driver-side,"
            f" so it cannot apply to this run; expect the gap: TPE asked {asked},"
            f" {logged} TRIAL_DONE logged, {asked - logged} reused-but-unjournalled"
            f" ({published} space(s) x 2 witnesses + {expansions} carried prior-best)")
        return
    say("J  reused measurements journalled", False,
        f"0 reused-measurement trials across {published} space(s), {expansions}"
        f" expansion(s), but the fix landed {hhmm(fix)} BEFORE this run started"
        f" {hhmm(run_start)} -- it should have fired")


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
    check_antihack(run, rows)
    check_reeval_gap(rows)
    check_error_excerpt(run, rows)
    check_metrics_crash(run, rows)
    check_cosine_overflow(run, rows)
    check_reused_journalled(rows)


if __name__ == "__main__":
    main()
