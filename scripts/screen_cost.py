#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""How much wall clock do compile screens cost, and does any of it come back as a verdict?

Prompted by box 3 sitting 15 min idle with the GPU at 0% while `ptxas` chewed 99.5% of one core on
152,185 lines of PTX (7.5 MB). Not the recorded 111 GiB OOM case -- RSS was flat at 680 MiB on a box
with 979 GB free -- so it is compute-bound, bounded by `build_timeout_s`, and by design can never
reject a candidate. The question is therefore not "is it broken" but "what does it cost".

Measures, per candidate:
  * gaps between consecutive events longer than a threshold, which is where a blocking screen shows up
  * how many CONFIG_SCREENED_INFEASIBLE the screen actually produced (its only payoff)
so a 20-minute screen that saves 40 cheap trials is a win and one that saves nothing is pure loss.

    python screen_cost.py <run_dir> [gap_threshold_s]
"""
from __future__ import annotations

import io
import json
import os
import sys

run = sys.argv[1]
THRESH = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0

evs = []
with io.open(os.path.join(run, "events.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                evs.append(json.loads(line))
            except ValueError:
                pass
print("run: %s   events: %d" % (os.path.basename(run), len(evs)))

# Agent calls are the OTHER reason for a long gap, and they dwarf a compile screen. Excluding their
# spans is what makes the remaining gaps attributable to GPU-side work -- without this the biggest
# "screen cost" would just be a rewriter call, which is the recorded shape of blaming the wrong thing.
agent_spans = []
open_ = {}
for e in evs:
    ty, p = e.get("type"), (e.get("payload") or {})
    k = p.get("call_id") or p.get("module")
    if ty == "AGENT_CALL_STARTED":
        open_[k] = e.get("ts") or 0.0
    elif ty in ("AGENT_CALL_FINISHED", "AGENT_CALL_FAILED") and k in open_:
        agent_spans.append((open_.pop(k), e.get("ts") or 0.0))


def in_agent(t0: float, t1: float) -> bool:
    return any(s <= t0 and t1 <= en for s, en in agent_spans)


gaps = []
for a, b in zip(evs, evs[1:]):
    t0, t1 = a.get("ts") or 0.0, b.get("ts") or 0.0
    d = t1 - t0
    if d >= THRESH and not in_agent(t0, t1):
        gaps.append((d, a.get("seq"), a.get("type"), b.get("type")))
gaps.sort(reverse=True)

print("\nnon-agent gaps >= %.0f s: %d, totalling %.2f h" % (
    THRESH, len(gaps), sum(g[0] for g in gaps) / 3600.0))
for d, seq, ta, tb in gaps[:12]:
    print("  %7.1f s  after seq %-5s %-28s -> %s" % (d, seq, ta, tb))

n_screened = sum(1 for e in evs if e.get("type") == "CONFIG_SCREENED_INFEASIBLE")
n_trials = sum(1 for e in evs if e.get("type") == "TRIAL_DONE")
print("\nCONFIG_SCREENED_INFEASIBLE: %d   (the screen's only payoff -- each is a trial not run)"
      % n_screened)
print("TRIAL_DONE: %d" % n_trials)
if gaps and n_screened:
    print("=> %.1f s of blocking per config actually screened out" % (
        sum(g[0] for g in gaps) / n_screened))
elif gaps:
    print("=> %.2f h of blocking and NOTHING screened out: pure loss" % (
        sum(g[0] for g in gaps) / 3600.0))

# A screen that times out is the expensive case, and the harness records it. Named separately from a
# screen that answers, because only the timeout costs the full build_timeout_s.
kinds = {}
for e in evs:
    if e.get("type") in ("SCREEN_FAILED", "COMPILE_PROBE_FAILED", "WORKER_TIMEOUT",
                         "PRESCREEN_FAILED"):
        kinds[e.get("type")] = kinds.get(e.get("type"), 0) + 1
print("screen/worker failure events: %s" % (kinds or "none recorded under those names"))

# THE EVENT LOG UNDERSTATES THIS, so count the jobs directory too. A worker job that never finished
# leaves its `*.json` spec with no matching `*.out.json`, and a batch that dies produces neither an
# output file nor a `SPACE_PRESCREENED` event -- so on box 3 the log showed 4 prescreens while disk
# showed 5, the missing one being a third timeout. Measured there: 3 of 5 batch prescreens timed out
# (60%) against 2 of 117 per-trial screens (1.7%), which is the batching hypothesis confirmed --
# batching 40 configs multiplies the chance that SOME member is the one whose ptxas runs 20 minutes.
jobs = os.path.join(run, "jobs")
if os.path.isdir(jobs):
    specs = {}
    for f in sorted(os.listdir(jobs)):
        if not f.endswith(".json") or f.endswith(".out.json"):
            continue
        kind = "prescreen" if "prescreen" in f else (
            "compile-screen" if "compile-screen" in f else "other")
        done = os.path.exists(os.path.join(jobs, f[:-len(".json")] + ".out.json"))
        tot, nod = specs.get(kind, (0, 0))
        specs[kind] = (tot + 1, nod + (0 if done else 1))
    print("\njobs on disk (a spec with no .out.json never completed):")
    for kind in ("prescreen", "compile-screen", "other"):
        if kind not in specs:
            continue
        tot, nod = specs[kind]
        print("  %-16s %3d jobs, %2d with no output  (%.1f%%)%s"
              % (kind, tot, nod, 100.0 * nod / tot,
                 "   <- batching multiplies the chance of hitting a slow config"
                 if kind == "prescreen" and nod else ""))
    if "prescreen" in specs and n_screened == 0 and specs["prescreen"][1]:
        print("  => every prescreen that timed out screened NOTHING: the batch's answers die with it")

