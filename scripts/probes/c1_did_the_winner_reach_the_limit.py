"""C1's first half, measured: did the winners actually approach the hardware limit, and did alignment
get them there FASTER?

C1 (new framing) claims a framework that "aligns the kernel's resource dimensions with the hardware's,
so the kernel REACHES the hardware's physical limit, and reaches it faster by letting the agent spend
idle resources". Two testable halves, and the run log carries the material for both:

  REACHED?  `DIMENSION_STATE.binding` is the per-dimension binding read (each dimension declares its
            own polarity -- occupancy binds when LOW, shared_bytes when HIGH -- so `n_binding` is the
            count of dimensions actually at their limit, not a count of large numbers). A winner with
            several binding dimensions is at a wall; one with none is not limited by the hardware at
            all, and for that one C1's story does not apply.

  FASTER?   the arms differ in whether the agent SEES the aligned per-dimension view (`mode: vector`)
            or one bottleneck label (`mode: label`). If alignment speeds up the approach to the limit,
            the vector arms should reach a given quality earlier in the run, not merely end better.
            Measured here as the normalised arrival time of the winner (0 = run start, 1 = run end)
            plus the best-so-far trace, because "faster" is a statement about the trajectory and the
            final number cannot carry it.

`unreachable_ceilings` is printed alongside because a ceiling the harness could not measure must not
be silently read as "not reached" -- that is the difference between a kernel that failed to reach a
limit and one whose limit we never knew.
"""
import json
import os
import sys
from collections import Counter


def load(rd):
    out = []
    with open(os.path.join(rd, "events.jsonl"), encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    return out


for spec in sys.argv[1:]:
    label, rd = spec.split("=", 1)
    evs = load(rd)
    t0, t1 = evs[0]["ts"], evs[-1]["ts"]
    fin = [e for e in evs if e.get("type") == "RUN_FINISHED"]
    best = (((fin[-1].get("payload") or {}).get("summary") or {}).get("best") or {}) if fin else {}
    winner = best.get("candidate_id")

    print(f"=== {label}")
    print(f"  winner {winner}  {best.get('final_reeval_ms')} ms  "
          f"same-precision {(best.get('honest_verdict') or {}).get('same_precision_speedup')}x")

    # REACHED? -- the winner's own binding read, and the run-wide distribution for context.
    per_cand = {}
    for e in evs:
        if e.get("type") != "DIMENSION_STATE":
            continue
        p = e.get("payload") or {}
        cid = p.get("candidate_id")
        if cid:
            per_cand[cid] = p
    counts = Counter(int(p.get("n_binding") or 0) for p in per_cand.values())
    print(f"  DIMENSION_STATE for {len(per_cand)} candidates; n_binding distribution "
          f"{dict(sorted(counts.items()))}")
    w = per_cand.get(winner)
    if w:
        binding = w.get("binding")
        names = (list(binding) if isinstance(binding, dict)
                 else binding if isinstance(binding, list) else [])
        print(f"  WINNER binding dims ({w.get('n_binding')}): {names}")
        unreach = w.get("unreachable_ceilings") or []
        print(f"  WINNER unreachable ceilings: {unreach or 'none'}"
              + ("   <-- these are NOT 'unreached', they are unmeasured" if unreach else ""))
    else:
        print(f"  *** no DIMENSION_STATE for the winner -- cannot say whether it hit a limit")

    # FASTER? -- when did the winner arrive, and how did best-so-far progress?
    arrival = None
    for e in evs:
        p = e.get("payload") or {}
        c = (p.get("candidate") or {})
        if c.get("candidate_id") == winner and e.get("type") == "CANDIDATE_REGISTERED":
            arrival = (e["ts"] - t0) / (t1 - t0)
            break
    print(f"  winner registered at normalised time "
          f"{f'{arrival:.2f}' if arrival is not None else '?'} of the run "
          f"(uniform arrival would be 0.50)")

    # Best-so-far trace over completed trials, to compare trajectories rather than endpoints.
    best_so_far, trace = None, []
    for e in evs:
        if e.get("type") != "TRIAL_DONE":
            continue
        t = (e.get("payload") or {}).get("trial") or {}
        if t.get("status") != "complete":
            continue
        lat = t.get("latency_ms") or {}
        med = lat.get("median")
        ms = med if isinstance(med, (int, float)) and med > 0 else lat.get("mean")
        if not isinstance(ms, (int, float)):
            continue
        if best_so_far is None or ms < best_so_far:
            best_so_far = ms
            trace.append(((e["ts"] - t0) / (t1 - t0), ms))
    print(f"  best-so-far improved {len(trace)} times; at 25/50/75% of the run: " + ", ".join(
        f"{q:.0%}={next((ms for f, ms in reversed(trace) if f <= q), float('nan')):.3f}ms"
        for q in (0.25, 0.50, 0.75)))
    print()
