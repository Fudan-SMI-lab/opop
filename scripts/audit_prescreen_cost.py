#!/usr/bin/env python3
"""What does F5's prescreen cost per candidate, and what does it buy?

Motivation: on run-l3-21 a prescreen worker ran 15 minutes on a 40-variant batch, long enough
to trip a stall alarm. The design measurement was 16.7 s process start plus ~7 ms marginal per
config (11.02 s for 48 configs) on a simple pipelined matmul -- but that was one kernel, and a
real L3 candidate can carry several kernels per variant, each compiled separately.

So: measure the cost against the benefit, per candidate. The benefit is `infeasible` (trials the
screen prevented, at ~18.6 s each) plus the post-materialize screen's own refusals; the cost is
wall clock the run cannot spend on trials.

Reads only events.jsonl.
"""
import json
import pathlib
import sys


def cid(e):
    """candidate_id wherever this event type puts it."""
    p = e.get("payload") or {}
    for key in ("candidate_id",):
        if p.get(key):
            return p[key]
    for sub in ("candidate", "space", "trial"):
        d = p.get(sub)
        if isinstance(d, dict) and d.get("candidate_id"):
            return d["candidate_id"]
    return None


run = pathlib.Path(sys.argv[1])
evs = []
for ln in (run / "events.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        evs.append(json.loads(ln))
    except Exception:  # noqa: BLE001
        continue

# Anchor: the event immediately preceding SPACE_PRESCREENED for that candidate is the best
# available "screen started" marker, since _prescreen_space runs at the top of _tune.
prev_ts = {}
rows = []
last_ts_any = None
for e in evs:
    t = e.get("type")
    c = cid(e)
    if t == "SPACE_PRESCREENED":
        p = e.get("payload") or {}
        started = prev_ts.get(c, last_ts_any)
        dur = (e["ts"] - started) if started else None
        rows.append((c, dur, p.get("configs_probed"), p.get("infeasible")))
    if c:
        prev_ts[c] = e["ts"]
    last_ts_any = e["ts"]

screened_infeasible = sum(1 for e in evs if e.get("type") == "CONFIG_SCREENED_INFEASIBLE")
shmem_trials = 0
shmem_by_screen = 0
for e in evs:
    if e.get("type") != "TRIAL_DONE":
        continue
    t = (e["payload"].get("trial") or {})
    if t.get("failure_kind") != "infeasible_shared_memory":
        continue
    shmem_trials += 1
    if "compile-only screen" in str(t.get("failure_detail") or ""):
        shmem_by_screen += 1

print(f"{'candidate':<18}{'screen_s':>10}{'probed':>8}{'infeasible':>11}")
print("-" * 48)
tot = 0.0
tot_inf = 0
for c, d, n, inf in rows:
    ds = f"{d:.1f}" if d else "?"
    print(f"{(c or '?')[:18]:<18}{ds:>10}{n or 0:>8}{inf if inf is not None else -1:>11}")
    tot += d or 0.0
    tot_inf += inf or 0

print("-" * 48)
print(f"prescreens               : {len(rows)}")
print(f"wall clock in prescreen  : {tot/60:.1f} min")
print(f"configs the SAMPLER avoided (guard, from the batch cache): {tot_inf}")
print(f"CONFIG_SCREENED_INFEASIBLE (post-materialize refusals)   : {screened_infeasible}")
print(f"trials ending in infeasible_shared_memory                : {shmem_trials}"
      f"  (of which refused by the compile screen: {shmem_by_screen})")
print()
# What the screen actually SAVES is a launch that would have raised, not a whole trial:
# `CONFIG_SCREENED_INFEASIBLE` still consumes a trial slot and a worker round-trip. Only the
# guard-level avoidance (tot_inf) removes a point from the sampler's consideration entirely.
#
# And the two counters are NOT additive. Every one of this run's infeasible_shared_memory trials
# carries "compile-only screen" in its detail, i.e. the post-materialize refusals ARE those
# trials -- counting both double-counts the same events. Measured: 33 and 33.
avoided = tot_inf
print(f"approx trial time avoided : {avoided*18.6/60:.1f} min "
      f"({avoided} guard-level avoidances x 18.6 s)")
if tot > 0:
    print(f"net                       : {(avoided*18.6 - tot)/60:+.1f} min")
    if avoided * 18.6 < tot:
        print("  -> the screen costs more wall clock than it saves on this task")
print()
print("Note: a post-materialize refusal is NOT a saved trial -- it still spent a trial slot and")
print("a worker round-trip; it only avoided a launch that would have raised. Only the")
print("guard-level count removes a point from the sampler entirely.")
