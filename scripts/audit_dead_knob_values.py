#!/usr/bin/env python3
"""How much trial budget goes to knob VALUES that never once succeed?

Motivation. `correctness_mismatch` is reported to Optuna as FAIL, deliberately: such a failure
can be non-deterministic or a candidate defect rather than a property of the point, so teaching
the sampler to avoid that region could be teaching it noise (tpe.py's own comment). FAIL trials
are excluded from the TPE model entirely, so the sampler never learns.

That reasoning holds for a scattered failure. It does NOT obviously hold for a categorical knob
value that fails EVERY time it is drawn -- there the determinism is visible in the data, and
each further draw is a trial spent re-learning nothing.

This measures the size of that effect rather than assuming it. Counted per (candidate, knob,
value), never pooled across candidates: a value fatal for one candidate may be fine for another,
and pooling would manufacture a pattern.

The denominator is DISTINCT TRIALS, not a sum over knobs. A single trial can draw two hopeless
values at once (COMPUTE_DTYPE=bf16 together with BC_CACHE_DTYPE=bf16), so summing per-knob waste
double-counts and can exceed 100% -- which is how I caught this script's first version.
"""
import collections
import json
import pathlib
import sys

MIN_EVIDENCE = 3   # failures before "this value never works" is a claim the data supports


def load(run: pathlib.Path):
    """(candidate, values, ok, index) per trial, in run order."""
    out = []
    p = run / "events.jsonl"
    if not p.exists():
        return out
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        if e.get("type") != "TRIAL_DONE":
            continue
        pay = e.get("payload") or {}
        t = pay.get("trial") or {}
        out.append((pay.get("candidate_id") or t.get("candidate_id") or "?",
                    (t.get("params") or {}).get("values") or {},
                    t.get("status") == "complete"))
    return out


grand_wasted = grand_total = 0
for arg in sys.argv[1:]:
    run = pathlib.Path(arg)
    trials = load(run)
    if not trials:
        continue

    # Pass 1: outcome sequence per (candidate, knob, value).
    seq = collections.defaultdict(list)
    for cand, vals, ok in trials:
        for k, v in vals.items():
            if isinstance(v, str):
                seq[(cand, k, v)].append(ok)

    hopeless = {key for key, states in seq.items()
                if not any(states) and len(states) > MIN_EVIDENCE}

    # Pass 2: a trial is wasted if it drew a hopeless value AFTER that value had already
    # failed MIN_EVIDENCE times. Counted once per trial, however many hopeless knobs it drew.
    seen_failures = collections.Counter()
    wasted_trials = 0
    for cand, vals, ok in trials:
        was_wasted = False
        for k, v in vals.items():
            if not isinstance(v, str):
                continue
            key = (cand, k, v)
            if key in hopeless and seen_failures[key] >= MIN_EVIDENCE:
                was_wasted = True
            if not ok:
                seen_failures[key] += 1
        if was_wasted:
            wasted_trials += 1

    if hopeless:
        print(f"=== {run.name}   ({len(trials)} trials)")
        for (cand, k, v) in sorted(hopeless, key=str):
            states = seq[(cand, k, v)]
            print(f"    {cand[:14]:<14} {k}={v:<14} drawn {len(states):>3}  completed 0")
        print(f"    -> {wasted_trials} trial(s) drew an already-hopeless value "
              f"({100*wasted_trials/len(trials):.0f}% of this run's trials)")
    grand_wasted += wasted_trials
    grand_total += len(trials)

print()
print(f"trials that drew an already-hopeless value: {grand_wasted}")
print(f"total trials examined                     : {grand_total}")
if grand_total:
    print(f"share                                     : "
          f"{100*grand_wasted/grand_total:.1f}%")
