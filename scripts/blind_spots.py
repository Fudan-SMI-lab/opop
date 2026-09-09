"""Quantify the three blind spots the user named, from our own run history.

The user's framing: v2's "explore parameters -> hit a wall -> trigger a rewrite" loop IS a
form of aligning an operator's resource usage with the hardware's resource supply, but it is
blind in three ways. This measures each blindness on real runs so the v3 design is grounded
in numbers rather than in my description of the numbers.

BLIND SPOT 1: no notion of conversion efficiency -- was the cost of a rewrite worth what it
              bought? Measured as: for each rewrite, what did latency do AND what did the
              resource footprint (shared, regs, occupancy) do?
BLIND SPOT 2: the agent only learns WHICH wall it hit (often not even that). Measured as: how
              many distinct bottleneck verdicts did the agent ever see, and how often was the
              verdict the same uninformative label?
BLIND SPOT 3: the agent cannot know what its change will DO to resource usage. Measured as:
              the spread of resource footprints between a parent and its rewrite -- i.e. how
              large the unpredicted swing actually is.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = [Path(a) for a in sys.argv[1:]]

for run in runs:
    ev = run / "events.jsonl"
    if not ev.exists():
        continue

    parents = {}            # cand -> [parent ids]
    origin = {}
    best_of = {}            # cand -> best median ms
    prof_of = {}            # cand -> profile at its best trial
    verdicts = []           # (cand, kind, evidence)
    per_cand_trials = defaultdict(int)

    with open(ev, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            t = e.get("type")
            p = e.get("payload") or {}
            if t == "CANDIDATE_REGISTERED":
                c = p.get("candidate") or p
                cid = c.get("candidate_id")
                parents[cid] = c.get("parent_ids") or []
                origin[cid] = c.get("origin")
            elif t == "TRIAL_DONE":
                tr = p.get("trial") or p
                cid = tr.get("candidate_id")
                per_cand_trials[cid] += 1
                if (tr.get("status") or "").lower() != "complete":
                    continue
                m = (tr.get("latency_ms") or {}).get("median")
                if not isinstance(m, (int, float)):
                    continue
                if cid not in best_of or m < best_of[cid]:
                    best_of[cid] = m
                    prof_of[cid] = tr.get("profile") or {}
            elif t == "BOTTLENECK_CLASSIFIED":
                verdicts.append((p.get("candidate_id"), p.get("kind"), p.get("evidence") or {}))

    print("=" * 100)
    print("RUN %s" % run.name)
    print("=" * 100)

    # --- BLIND SPOT 2: what did the agent actually learn about the wall?
    kinds = defaultdict(int)
    for _c, k, _e in verdicts:
        kinds[k] += 1
    tot = sum(kinds.values())
    print("\n[BLIND SPOT 2] bottleneck verdicts the agent ever saw: %d reports, %d distinct labels"
          % (tot, len(kinds)))
    for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print("    %-22s %3d  (%4.1f%%)" % (k, n, 100 * n / tot if tot else 0))
    # how many dimensions were near their limit AT THE SAME TIME?
    multi = 0
    detail = []
    for c, k, evd in verdicts:
        near = []
        dram = evd.get("pct_of_dram_peak")
        comp = evd.get("pct_of_compute_peak")
        occ = evd.get("occupancy")
        if isinstance(dram, (int, float)) and dram >= 80:
            near.append("dram %.0f%%" % dram)
        if isinstance(comp, (int, float)) and comp >= 80:
            near.append("compute %.0f%%" % comp)
        if isinstance(occ, (int, float)) and occ <= 0.30:
            near.append("occupancy %.0f%%" % (occ * 100))
        if evd.get("n_spills"):
            near.append("spills %s" % evd.get("n_spills"))
        if len(near) >= 2:
            multi += 1
            detail.append((c, k, near))
    print("    reports where >=2 dimensions were simultaneously near a limit: %d of %d (%.0f%%)"
          % (multi, tot, 100 * multi / tot if tot else 0))
    for c, k, near in detail[:6]:
        print("        %s labelled '%s' while: %s" % (c, k, " + ".join(near)))

    # --- BLIND SPOT 1 + 3: what did each rewrite cost and buy?
    print("\n[BLIND SPOT 1+3] rewrites: what changed in latency AND in resource footprint")
    hdr = ("%-18s %-18s %9s %9s %8s | %7s %7s %7s %7s"
           % ("child(rewrite)", "parent", "parent ms", "child ms", "delta", "d_shared",
              "d_regs", "d_spill", "d_warps"))
    print("  " + hdr)
    n_rew = 0
    swings = []
    for cid, par in parents.items():
        if origin.get(cid) != "rewrite" or not par:
            continue
        pid = par[0]
        if cid not in best_of or pid not in best_of:
            continue
        n_rew += 1
        pc, cc = prof_of.get(pid) or {}, prof_of.get(cid) or {}

        def d(field):
            a, b = pc.get(field), cc.get(field)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return b - a
            return None

        ds, dr, dsp, dw = d("shared_bytes"), d("n_regs"), d("n_spills"), d("num_warps")
        pct = 100 * (best_of[cid] - best_of[pid]) / best_of[pid]
        swings.append((abs(ds) if ds is not None else None,
                       abs(dr) if dr is not None else None, pct))
        print("  %-18s %-18s %9.4f %9.4f %+7.1f%% | %7s %7s %7s %7s"
              % (cid, pid, best_of[pid], best_of[cid], pct,
                 ds if ds is not None else "?", dr if dr is not None else "?",
                 dsp if dsp is not None else "?", dw if dw is not None else "?"))
    if not n_rew:
        print("  (no rewrite with both parent and child measured)")
    else:
        sh = [s for s, _r, _p in swings if s is not None]
        rg = [r for _s, r, _p in swings if r is not None]
        if sh:
            print("\n  unpredicted shared-memory swing across rewrites: median %d B, max %d B"
                  % (sorted(sh)[len(sh) // 2], max(sh)))
        if rg:
            print("  unpredicted register swing across rewrites:      median %d, max %d"
                  % (sorted(rg)[len(rg) // 2], max(rg)))
        print("  --> this is the magnitude the agent currently cannot anticipate at all.")
    print()
