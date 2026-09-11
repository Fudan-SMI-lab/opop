#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Does the expectation ledger ever reach a prompt, or does family rotation guarantee it never does?

S2d(c) is the claim that reconciled predictions improve the NEXT rewrite. The ledger is keyed per
family (`self.ledger.get(family_id, [])`), so it reaches a prompt only when the SAME family gets a
SECOND rewrite round. Measured on both live arms: round 1 went to one family and round 2 to a
different one, so each rewriter call so far has correctly received an empty ledger -- correct
behaviour, and also a structural threat to the treatment ever being administered.

This counts, per run: how many rewrite rounds each family received. `max_families_active` and the
convergence order decide the rotation, so whether a family is ever revisited is a property of the
control loop, not of S2d.

If no family in the finished corpus ever got two rounds, S2d(c) cannot be tested by a 12 h run at all
and that is a finding about the experiment design, not about the ledger.

    python ledger_reach.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import io
import json
import os
import sys


def rounds_per_family(run: str):
    evs = []
    p = os.path.join(run, "events.jsonl")
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    evs.append(json.loads(line))
                except ValueError:
                    pass
    # A rewrite ROUND is one rewriter call's worth of REWRITE_PRODUCED for a family. Counting the
    # events themselves would double every round (two candidates per round, 9/9 measured), so group
    # by (family, timestamp of the producing call).
    rounds = {}
    for e in evs:
        if e.get("type") != "REWRITE_PRODUCED":
            continue
        pl = e.get("payload") or {}
        rounds.setdefault(pl.get("family_id"), set()).add(round(e.get("ts") or 0.0, 0))
    recon = {}
    for e in evs:
        if e.get("type") == "EXPECTATIONS_RECONCILED":
            fid = (e.get("payload") or {}).get("family_id")
            recon[fid] = recon.get(fid, 0) + 1
    return rounds, recon, len(evs)


print("%-42s %-16s %7s %7s %s" % ("run", "family", "rounds", "recon", "ledger could reach a prompt?"))
print("-" * 104)
any_second = False
for run in sys.argv[1:]:
    got = rounds_per_family(run)
    if not got:
        print("%-42s (no events.jsonl)" % os.path.basename(run)[:42])
        continue
    rounds, recon, n = got
    if not rounds:
        print("%-42s (no rewrite round yet, %d events)" % (os.path.basename(run)[:42], n))
        continue
    for fid in sorted(rounds, key=lambda f: -len(rounds[f])):
        k = len(rounds[fid])
        # The ledger reaches a prompt only if a round FOLLOWED a reconciliation for the same family,
        # i.e. the family has at least 2 rounds AND at least 1 reconciled entry.
        reach = "YES" if (k >= 2 and recon.get(fid, 0) >= 1) else "no"
        if reach == "YES":
            any_second = True
        print("%-42s %-16s %7d %7d %s" % (os.path.basename(run)[:42], fid, k,
                                          recon.get(fid, 0), reach))
print("-" * 104)
print("any family anywhere received a SECOND rewrite round after a reconciliation: %s"
      % ("YES" if any_second else "NO -- S2d(c) has never been administered"))
