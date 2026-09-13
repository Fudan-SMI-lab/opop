"""What would multi-knob ablation COST? Count the evaluations, before proposing it.

WHY. §4.5 of docs/next-round-changes.md proposes backing a refused config toward theta* one
dimension at a time to find the minimal set of knobs that makes it fit. That is a combinatorial
search, and today's single-knob probe costs 0.49 s per wall. Before anyone implements it, the
number of evaluations has to be known -- and it can be counted offline from the record, with no
GPU time at all.

THE CHEAPEST VERSION FIRST. A greedy backward pass: from the refused config, try reverting each
still-differing knob to its theta* value, keep the single revert that reduces shared_bytes most,
repeat until the config fits. That needs sum(d, d-1, ..., 1) = d(d+1)/2 evaluations in the worst
case for d differing knobs, and d*k in practice where k is how many reverts it takes to fit.
With d = 10 (the live BLOCK_N case) the worst case is 55 evaluations at ~0.49 s = ~27 s per wall.

BUT THE REAL QUESTION IS WHETHER THE EVALUATION IS FREE. shared_bytes for an arbitrary config is
not in the record unless that config was tried. So this probe reports, per refused config:
  * d, the number of knobs differing from theta*
  * the worst-case and greedy evaluation counts
  * HOW MANY of the single-knob reverts are ALREADY MEASURED in the record (those are free)
The last number decides whether this is a compile-bound search or mostly a lookup.

WHAT THE EVALUATION ACTUALLY COSTS -- measured, not assumed. The operation needed is the
COMPILE-ONLY SCREEN, which already reports `max_shared` and `limit` (CONFIG_SCREENED_INFEASIBLE
carries exactly `candidate_id, kernel, limit, max_shared, params`), so no timing run and no full
cubin is required. `correctness.py:242-243` records its measured cost: **7 ms marginal per config
in a BATCH**, ~16.7 s alone. Reverts are mutually independent, so they batch.

I first wrote here that compile_s "runs to tens of seconds" and that this would be a compile-bound
search. That was wrong on both counts: compile_s over 1821 measured trials is median 0.42 s, p90
0.45 s, max 1.53 s. The tens-of-seconds and 111 GiB figures in this project belong to one agent's
pathological hand-written PTX, not to ordinary trials -- citing them here priced the proposal
three orders of magnitude too high.
"""
from __future__ import annotations

import json
import os
import sys


def _read(path: str) -> list[dict]:
    ev = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return ev


def _params(tr: dict) -> dict:
    p = tr.get("params") or {}
    v = p.get("values")
    return dict(v) if isinstance(v, dict) else {}


def _key(d: dict) -> str:
    return json.dumps(d, sort_keys=True)


def report(events: list[dict], label: str) -> dict:
    by_space: dict[str, list[dict]] = {}
    for e in events:
        if e.get("type") != "TRIAL_DONE":
            continue
        tr = (e.get("payload") or {}).get("trial") or {}
        by_space.setdefault(str(tr.get("space_id") or "?"), []).append(tr)

    out = {"cases": 0, "d_values": [], "free_reverts": 0, "total_reverts": 0,
           "worst_case_evals": 0}
    print("=" * 78)
    print(label)
    for sid, trials in sorted(by_space.items()):
        done = [t for t in trials if t.get("status") == "complete"]
        refused = [t for t in trials
                   if str(t.get("failure_kind") or "") == "infeasible_shared_memory"]
        if not (done and refused):
            continue
        # theta*: the best complete trial in this space.
        best, best_ms = None, None
        for t in done:
            lat = t.get("latency_ms") or {}
            m = lat.get("median")
            if not isinstance(m, (int, float)):
                m = lat.get("mean")
            if isinstance(m, (int, float)) and (best_ms is None or m < best_ms):
                best_ms, best = m, t
        if best is None:
            continue
        bp = _params(best)
        # Every config whose shared_bytes the record already knows -- a free lookup.
        known: dict[str, object] = {}
        for t in trials:
            sb = (t.get("profile") or {}).get("shared_bytes")
            if sb is not None:
                known[_key(_params(t))] = sb
        print("  space %s   theta* has %d measured config(s) with shared_bytes"
              % (sid, len(known)))
        for t in refused:
            rp = _params(t)
            differing = sorted(k for k in rp if k in bp and rp[k] != bp[k])
            d = len(differing)
            if not d:
                continue
            out["cases"] += 1
            out["d_values"].append(d)
            worst = d * (d + 1) // 2
            out["worst_case_evals"] += worst
            # How many of the FIRST-LEVEL reverts are already measured? Each is the refused config
            # with exactly one knob put back to its theta* value.
            free = 0
            for k in differing:
                cand = dict(rp)
                cand[k] = bp[k]
                out["total_reverts"] += 1
                if _key(cand) in known:
                    free += 1
            out["free_reverts"] += free
            print("      d=%-3d worst-case %-4d evals   first-level reverts already measured: "
                  "%d/%d   knobs: %s"
                  % (d, worst, free, d, ",".join(differing[:6]) + ("..." if d > 6 else "")))
    return out


if __name__ == "__main__":
    paths = [a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl")
             for a in sys.argv[1:] if not a.startswith("-")]
    tot = {"cases": 0, "d_values": [], "free_reverts": 0, "total_reverts": 0,
           "worst_case_evals": 0}
    for path in paths:
        if not os.path.exists(path):
            continue
        f = report(_read(path), "/".join(path.split(os.sep)[-3:-1]))
        tot["cases"] += f["cases"]
        tot["d_values"].extend(f["d_values"])
        for k in ("free_reverts", "total_reverts", "worst_case_evals"):
            tot[k] += f[k]
    print()
    print("=" * 78)
    if not tot["cases"]:
        print("NO refused config had a measured theta* to back toward. Nothing to price -- this is")
        print("not a finding that the search would be cheap.")
        raise SystemExit(0)
    ds = sorted(tot["d_values"])
    med = ds[len(ds) // 2]
    print("PRICING MULTI-KNOB ABLATION over %d refused config(s)" % tot["cases"])
    print("  knobs differing from theta*: min %d  median %d  max %d" % (ds[0], med, ds[-1]))
    print("  greedy worst case, summed:   %d evaluations" % tot["worst_case_evals"])
    print("  first-level reverts already measured in the record: %d / %d (%.0f%%)"
          % (tot["free_reverts"], tot["total_reverts"],
             100.0 * tot["free_reverts"] / tot["total_reverts"] if tot["total_reverts"] else 0.0))
    print()
    print("WHAT IT COSTS, with the record's own measured prices:")
    # correctness.py:170-191 measures the screen precisely: 48 configs in ONE worker process took
    # 11.02 s total, a marginal 7 ms each, against a median 16.7 s for a probe in its own process
    # -- essentially all of that being process start plus torch/CUDA/KernelBench import. So the
    # per-config price is 7 ms but there is an ~11 s FIXED cost per batch, and the batch COUNT is
    # what a multi-level backward pass drives up: each level is a new batch, because the next
    # level's points depend on this level's answers. Pricing only the marginal 7 ms understates it.
    _FIXED_S, _MARGINAL_S, _ALONE_S = 11.0, 0.007, 16.7
    _ds = sorted(tot["d_values"])
    _med_d = _ds[len(_ds) // 2]
    for n_eval, n_batches, tag in (
            (tot["total_reverts"], 1, "first level only, ONE batch"),
            (tot["worst_case_evals"], _med_d, "greedy, one batch per level (all walls together)"),
            (tot["worst_case_evals"], tot["cases"], "greedy, a batch per refused config")):
        secs = n_batches * _FIXED_S + n_eval * _MARGINAL_S
        print("  %-46s %6d evals / %4d batch(es) = %7.1f s (%.2f h)"
              % (tag, n_eval, n_batches, secs, secs / 3600.0))
    print("  %-46s %6d evals / one process each = %.1f h"
          % ("for contrast, unbatched", tot["worst_case_evals"],
             tot["worst_case_evals"] * _ALONE_S / 3600.0))
    print()
    print("  => THE BATCH COUNT IS THE COST, not the evaluation count. At 7 ms each the")
    print("     evaluations are nearly free; the ~11 s process start is not. So the affordable")
    print("     design submits each LEVEL as one batch across ALL walls at once, and the")
    print("     unaffordable one opens a process per wall. A test must pin which one it is:")
    print("     a per-wall fallback costs 100x and still returns the right answer.")
    print()
    print("  For scale, from the same measurement: the shared-affecting subgrid of a real space")
    print("  reaches 600,000 points = 70 min at 7 ms, so EXHAUSTIVE screening is out of reach.")
    print("  A directed backward pass is the point -- it visits d, then d-1, ..., not the grid.")
    print()
    print("TWO PRICES THIS DOES NOT INCLUDE, both of which must be measured before committing:")
    print("  * the screen CACHES per materialized source, so novel points populate the cache and")
    print("    a cached FAILURE is deliberately not stored (correctness.py:240-244); a batch that")
    print("    times out therefore costs a re-probe, not a permanent blind spot.")
    print("  * the wall clock is always the binding budget (8/8 finished runs ended on it), so")
    print("    even a minute per wall competes with trials. Price it per RUN, not per wall.")
    if tot["total_reverts"] and tot["free_reverts"] / tot["total_reverts"] < 0.25:
        print()
        print("=> MOSTLY NOVEL (%.0f%% already measured). These points are not in the record, so"
              % (100.0 * tot["free_reverts"] / tot["total_reverts"]))
        print("   they need screening -- but screening is the 7 ms/16.7 s operation above, NOT a")
        print("   full compile-and-time. Do not price this against compile_s.")
