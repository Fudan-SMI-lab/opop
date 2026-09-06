"""Cross-check a finished run's report against its own events.jsonl.

Written for the L3:43 close-out but task-agnostic: every number a report claims should be
derivable from the event log, and this prints them side by side so a mismatch is visible rather
than assumed away. Reads only; touches nothing.

Usage: python scripts/verify_report_against_events.py runs/<run-id>
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ATTENTION_HINTS = ("attention", "flash", "score", "softmax", "probab")


def load(run: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in (run / "events.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = pathlib.Path(sys.argv[1])
    ev = load(run)
    print(f"run      : {run.name}")
    print(f"events   : {len(ev)}")
    print(f"elapsed  : {(ev[-1]['ts'] - ev[0]['ts']) / 3600:.2f} h")
    finished = [e for e in ev if e["type"] == "RUN_FINISHED"]
    print(f"finished : {bool(finished)}")

    # --- budget the run actually used (manifest, NOT the current config on disk) -----------
    man = run / "manifest.json"
    if man.exists():
        b = (json.loads(man.read_text(encoding="utf-8")).get("config") or {}).get("budgets") or {}
        keys = ("rewrite_rounds_per_family", "no_improve_rounds", "min_improvement_pct",
                "trials_per_space", "max_families_total", "max_seed_candidates",
                "wall_clock_hours")
        print("\nbudget in force for THIS run (from manifest.json):")
        for k in keys:
            if k in b:
                print(f"  {k:28s} {b[k]}")

    # --- families ------------------------------------------------------------------------
    fam: dict[str, list[float]] = defaultdict(list)
    state: dict[str, tuple] = {}
    for e in ev:
        p = e.get("payload") or {}
        if e["type"] == "FAMILY_ROUND_RECORDED":
            fam[p["family_id"]].append(p.get("best_ms"))
        if e["type"] == "CONVERGENCE_DECIDED":
            d = p.get("decision") or {}
            if d.get("scope") == "family" and p.get("family_id"):
                state[p["family_id"]] = (d.get("verdict"), d.get("stop_kind"),
                                         (d.get("evidence") or {}).get("rewrite_rounds_used"))
    print("\nfamilies (from events):")
    cut_off = []
    for f, h in sorted(fam.items(), key=lambda kv: min(x for x in kv[1] if x)):
        ms = [x for x in h if x]
        verdict, kind, rounds = state.get(f, ("?", None, None))
        tag = f"FROZEN({kind})" if verdict == "freeze" else "ACTIVE"
        print(f"  {f}  best={min(ms):6.2f}  rounds={rounds}  "
              f"hist={[round(x, 1) for x in ms]}  {tag}")
        if kind == "budget_exhausted" and len(ms) >= 2 and ms[-2]:
            gain = (ms[-2] - ms[-1]) / ms[-2] * 100
            if gain >= 2.0:
                cut_off.append((f, round(gain, 1)))
    if cut_off:
        print("  !! frozen on budget while STILL IMPROVING >=2% (defect 0b):")
        for f, g in cut_off:
            print(f"       {f}  final-round gain {g}%")

    # --- attribution: which candidates wrote their own attention kernel? ------------------
    best: dict[str, tuple] = {}
    for e in ev:
        if e["type"] != "TRIAL_DONE":
            continue
        t = e["payload"]["trial"]
        if t.get("status") != "complete":
            continue
        c, m = t.get("candidate_id"), (t.get("latency_ms") or {}).get("mean")
        if not m:
            continue
        if c not in best or m < best[c][0]:
            best[c] = (m, (t.get("profile") or {}).get("kernel_names") or [],
                       (t.get("params") or {}).get("values") or {})
    def own_attention(kn: list[str]) -> bool:
        return any(any(h in k.lower() for h in ATTENTION_HINTS) for k in kn)
    hand = [(m, c) for c, (m, kn, _) in best.items() if own_attention(kn)]
    dele = [(m, c) for c, (m, kn, _) in best.items() if not own_attention(kn)]
    print("\nattribution (kernel_names of each candidate's best trial):")
    if hand:
        print(f"  best FULLY hand-written : {min(hand)[0]:6.2f} ms  ({min(hand)[1]})")
    if dele:
        print(f"  best delegating         : {min(dele)[0]:6.2f} ms  ({min(dele)[1]})")
        print("  ^ a candidate with no attention kernel handed that work to PyTorch; report both")

    # --- trials ---------------------------------------------------------------------------
    st, kinds = Counter(), Counter()
    for e in ev:
        if e["type"] != "TRIAL_DONE":
            continue
        t = e["payload"]["trial"]
        st[t.get("status")] += 1
        if t.get("status") == "fail":
            kinds[t.get("failure_kind")] += 1
    n = sum(st.values())
    if n:
        print(f"\ntrials: {n}  complete={st['complete']}  fail={st['fail']} "
              f"({st['fail'] / n * 100:.1f}%)")
        print(f"  failure kinds: {dict(kinds)}")

    # --- losses ---------------------------------------------------------------------------
    loss = Counter()
    for e in ev:
        p = e.get("payload") or {}
        if e["type"] == "SPACE_REJECTED":
            loss[str(p.get("reason"))] += 1
        if e["type"] == "SPACE_EXPANSION_REJECTED":
            loss["expansion:" + str(p.get("reason"))] += 1
        if e["type"] == "AGENT_CALL_FAILED" and p.get("final"):
            loss["agent_failed:" + str(p.get("module"))] += 1
    if loss:
        print("\nlosses:")
        for k, v in loss.most_common():
            print(f"  {k:34s} {v}")

    # --- what the report claims -----------------------------------------------------------
    md = run / "report" / "report.md"
    if not md.exists():
        print("\n(no report/report.md yet)")
        return 0
    t = md.read_text(encoding="utf-8", errors="replace")
    g = lambda p, d="-": (re.search(p, t).group(1) if re.search(p, t) else d)  # noqa: E731
    print("\nreport.md claims:")
    print(f"  tuned            : {g(r'tuned latency: ([\d.]+) ms')} ms")
    print(f"  final re-eval    : {g(r're-eval: (\w+) at [\d.]+ ms')} at "
          f"{g(r're-eval: \w+ at ([\d.]+) ms')} ms")
    print(f"  precision        : {g(r'arithmetic precision: \*\*(\w+)\*\*')}")
    print(f"  vs torch_compile : {g(r'vs `torch_compile`: \*\*([\d.]+)x')}x")
    print(f"  vs tc_tf32       : {g(r'vs `torch_compile_tf32`: \*\*([\d.]+)x')}x")
    honest = "beats" if "✅ beats" in t else ("FAILS" if "same-precision" in t else "-")
    print(f"  honest verdict   : {honest}")
    if "excessive speedup" in t:
        print("  !! flagged: excessive speedup -- inspect the kernel before trusting")

    tuned = g(r"tuned latency: ([\d.]+) ms")
    reeval = g(r"re-eval: \w+ at ([\d.]+) ms")
    if tuned != "-" and reeval != "-":
        d = (float(reeval) - float(tuned)) / float(tuned) * 100
        print(f"\n  tuned -> re-eval drift: {d:+.1f}%  "
              f"(publish the re-eval; tuned runs 1.5-6.7% optimistic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
