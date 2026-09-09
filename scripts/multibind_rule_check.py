"""Gap A-1: when several dimensions bind at once, which one should the agent attack?

This is the only gap on the register marked "cannot proceed without it", and it opened when
we rejected scalarisation -- a single composite number answered it for free, and nothing
replaced the answer.

It cannot be settled by argument, but part of it CAN be settled from data we already have,
at zero GPU cost. Three questions, in the order that matters:

  Q1 How often does it actually happen? If multi-binding is rare, the gap is not urgent.
     Measured as: per candidate, how many dimensions sit within X% of their ceiling at the
     winning trial.

  Q2 When it happens, does the historical record show which choice was right? For every
     rewrite that followed a multi-binding diagnosis, we know which dimension the rewrite
     actually moved (from the resource delta) and whether latency improved. That is a
     labelled dataset of decisions and outcomes -- already on disk, never read.

  Q3 Is there a rule that beats picking arbitrarily? Three candidate rules are scored
     against Q2's labels:
       R1 highest utilisation first    (the scalarisation-flavoured rule)
       R2 furthest-from-lower-bound first  (the "most remaining room" rule)
       R3 whatever the agent chose     (the current de-facto rule)
     plus the two controls that make the comparison honest:
       C1 always pick the same dimension  (constant baseline)
       C2 pick uniformly at random        (chance baseline)

The point of C1/C2 is that a rule scoring 60% means nothing until we know what constant and
chance score. If R1/R2/R3 do not beat both controls, the honest finding is that we have no
basis for ranking -- which is itself the answer to A-1, and a publishable one, rather than a
rule invented to fill the hole.

Zero GPU. Reads events.jsonl only.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", type=Path)
ap.add_argument("--near", type=float, default=0.85,
                help="a dimension counts as binding at this fraction of its ceiling")
ap.add_argument("--sweep", action="store_true",
                help="report Q1 across several thresholds instead of one")
ap.add_argument("--min-gain", type=float, default=5.0,
                help="a rewrite counts as having HELPED only above this %% improvement. "
                     "Default 5%%: per-trial std reaches 16%% of the mean and near-tie bands "
                     "span 5-9%%, so a -0.2%% rewrite is not evidence of anything (D-3).")
args = ap.parse_args()
NEAR = args.near
runs = args.runs

# Dimensions we can read per trial, the field carrying the value, its scale, and -- the part
# that is easy to get wrong -- WHICH DIRECTION MEANS "BINDING".
#
# For bandwidth, compute, shared capacity and register pressure, HIGH means "against the wall".
# For occupancy, LOW means "against the wall": an occupancy of 17% is a severe constraint, and
# an occupancy of 100% is slack. My first version applied the high-is-binding test to all five
# and reported 0 multi-binding candidates -- while the very case that motivated A-1 (L3:48 at
# DRAM 93.7% AND occupancy 17%) was sitting in the input. A polarity error does not fail
# loudly; it produces a clean, wrong table. Hence `high_is_binding` is explicit per dimension.
DIMS = {
    "dram_bandwidth":    dict(field="pct_of_dram_peak",    scale=100.0, high_is_binding=True),
    "compute":           dict(field="pct_of_compute_peak", scale=100.0, high_is_binding=True),
    "occupancy":         dict(field="occupancy",           scale=1.0,   high_is_binding=False),
    "shared_capacity":   dict(field=None, scale=None,      high_is_binding=True),
    "register_pressure": dict(field=None, scale=None,      high_is_binding=True),
}
SHARED_LIMIT = 101376
REG_LIMIT = 255


def pressures(profile, required=("pct_of_dram_peak", "pct_of_compute_peak")):
    """Per-dimension PRESSURE in [0,1], where 1 always means 'against the wall'.

    Normalising every dimension so that high = constrained is what makes them comparable at
    all; it is not a composite score (no weighting, no summing, no argmax across dimensions
    for a headline). It only lets one threshold be applied to all of them.
    """
    raw = {}
    for name, spec in DIMS.items():
        if spec["field"] is None:
            continue
        v = profile.get(spec["field"])
        raw[name] = (v / spec["scale"]) if isinstance(v, (int, float)) else None
    sb = profile.get("shared_bytes")
    raw["shared_capacity"] = (sb / SHARED_LIMIT) if isinstance(sb, (int, float)) else None
    nr = profile.get("n_regs")
    raw["register_pressure"] = (nr / REG_LIMIT) if isinstance(nr, (int, float)) else None

    out = {}
    for name, v in raw.items():
        if v is None:
            out[name] = None
        elif DIMS[name]["high_is_binding"]:
            out[name] = min(v, 1.0) if v <= 1.5 else 1.0   # clamp; >100% is documented
        else:
            out[name] = 1.0 - v      # low occupancy -> high pressure
    missing = [f for f in required if profile.get(f) is None]
    return out, raw, missing


def load_classified(run):
    """candidate_id -> the classifier's own evidence dict.

    This is where the per-dimension utilisations actually live. My first version read them from
    TRIAL_DONE.profile, which carries only n_regs / n_spills / shared_bytes / occupancy --
    pct_of_dram_peak and pct_of_compute_peak are NOT there. Three of five columns came out
    empty, no dimension could co-bind with another, and the script reported 0 multi-binding
    candidates while cand-6498ec62 -- DRAM 94.6% AND occupancy 16.7%, the exact case that
    motivated A-1 -- sat in the input file. Absent fields do not fail; they read as slack.
    """
    ev = run / "events.jsonl"
    out = {}
    if not ev.exists():
        return out
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
            if t not in ("BOTTLENECK_CLASSIFIED", "BOTTLENECK_REPORTED"):
                continue
            p = e.get("payload") or {}
            cid = p.get("candidate_id")
            if not cid:
                continue
            if t == "BOTTLENECK_CLASSIFIED":
                # current shape: the classifier's own measured evidence
                out[cid] = dict(p.get("evidence") or {}, _kind=p.get("kind"),
                                _src="BOTTLENECK_CLASSIFIED")
            elif cid not in out:
                # Older runs (before BOTTLENECK_CLASSIFIED existed) carry only the ANALYST's
                # report: summary / parameter_limits / hypotheses / suggested_action. There is
                # no pct_of_dram_peak, no occupancy, no measured utilisation of any kind --
                # only agent prose. Those runs cannot answer Q1, and pretending otherwise would
                # read every dimension as slack. Recorded as present-but-unmeasurable.
                r = p.get("report") or {}
                out.setdefault(cid, dict(_kind=None, _src="BOTTLENECK_REPORTED(prose only)",
                                         _prose_keys=sorted(r.keys())))
    return out


def load(run):
    """Winning trial per candidate, plus every trial's (params, profile, latency)."""
    ev = run / "events.jsonl"
    if not ev.exists():
        return None
    best = {}
    trials = defaultdict(list)
    with open(ev, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            if (tr.get("status") or "").lower() != "complete":
                continue
            m = (tr.get("latency_ms") or {}).get("median")
            if not isinstance(m, (int, float)):
                continue
            cid = tr.get("candidate_id")
            pr = tr.get("profile") or {}
            trials[cid].append((m, pr))
            if cid not in best or m < best[cid][0]:
                best[cid] = (m, pr)
    return best, trials


def parent_of(run):
    """candidate_id -> parent candidate_id, for rewrite lineage."""
    ev = run / "events.jsonl"
    par = {}
    with open(ev, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "CANDIDATE_REGISTERED":
                continue
            c = (e.get("payload") or {}).get("candidate") or {}
            pids = c.get("parent_ids") or []
            if pids:
                par[c.get("candidate_id")] = pids[0]
    return par


# ---------------- Q1: how often do several dimensions bind at once? ----------------
print("=" * 96)
print("Q1  HOW OFTEN DO SEVERAL DIMENSIONS BIND AT ONCE?")
print("    pressure is normalised so 1.0 = against the wall in EVERY dimension")
print("    (occupancy is inverted: 17%% occupancy is pressure 0.83, not 0.17)")
print("=" * 96)

allruns = {}
for run in runs:
    got = load(run)
    if not got:
        continue
    best, trials = got
    allruns[run] = (best, trials, parent_of(run), load_classified(run))

# A run whose classifier evidence is missing cannot answer Q1 at all -- say so rather than
# silently reporting every dimension as slack.
n_missing = 0
for run, (best, trials, par, cls) in allruns.items():
    have = sum(1 for c in cls.values() if c.get("pct_of_dram_peak") is not None)
    srcs = Counter(c.get("_src") for c in cls.values())
    print("  %-34s candidates=%2d  diagnosed=%2d  with measured dram%%=%2d   %s"
          % (run.name[:34], len(best), len(cls), have,
             " ".join("%s x%d" % (k.split("(")[0], v) for k, v in srcs.items())))
    if have == 0 and cls:
        n_missing += 1
if n_missing:
    print()
    print("  *** %d run(s) predate BOTTLENECK_CLASSIFIED and carry only the analyst's prose"
          % n_missing)
    print("      report -- no pct_of_dram_peak, no occupancy, no measured utilisation at all.")
    print("      Those runs are EXCLUDED from Q1/Q3 rather than scored as all-slack. This")
    print("      halves the usable corpus and is itself a finding: the measured per-dimension")
    print("      record only exists from the run where BOTTLENECK_CLASSIFIED was added.")
print()

THRESHOLDS = [0.70, 0.80, 0.85, 0.90] if args.sweep else [NEAR]
for th in THRESHOLDS:
    hist = Counter()
    combos = Counter()
    for run, (best, trials, par, cls) in allruns.items():
        for cid in sorted(cls):
            if cls[cid].get("pct_of_dram_peak") is None:
                continue          # prose-only: unmeasurable, not slack
            pres, raw, miss = pressures(cls[cid])
            b = tuple(sorted(d for d, v in pres.items() if v is not None and v >= th))
            hist[len(b)] += 1
            if len(b) >= 2:
                combos[b] += 1
    tot = sum(hist.values())
    if not tot:
        continue
    multi = sum(v for k, v in hist.items() if k >= 2)
    print("  threshold %.2f:  %s   =>  multi-binding %d/%d (%.1f%%)"
          % (th, "  ".join("%dd:%d" % (k, hist[k]) for k in sorted(hist)),
             multi, tot, 100 * multi / tot))
    for combo, n in combos.most_common(4):
        print("        %2d x  %s" % (n, " + ".join(combo)))

print()
print("  per-candidate detail at threshold %.2f (pressure, high = constrained):" % NEAR)
print("  %-24s %9s %9s %9s %9s %9s  %s"
      % ("candidate", "dram", "compute", "occ", "shared", "regs", "binding"))
for run, (best, trials, par, cls) in allruns.items():
    print("  -- %s" % run.name)
    for cid in sorted(cls, key=lambda c: cls[c].get("gpu_ms") or 9e9):
        if cls[cid].get("pct_of_dram_peak") is None:
            continue
        pres, raw, miss = pressures(cls[cid])
        b = [d for d, v in pres.items() if v is not None and v >= NEAR]
        def f(d):
            v = pres.get(d)
            return "%.2f" % v if v is not None else "-"
        print("  %-24s %9s %9s %9s %9s %9s  %s"
              % (cid[:24], f("dram_bandwidth"), f("compute"), f("occupancy"),
                 f("shared_capacity"), f("register_pressure"),
                 ("%d: %s" % (len(b), ",".join(b))) if b else "0"))

# ---------------- Q2: labelled decisions from rewrite lineage ----------------
print()
print("=" * 96)
print("Q2  WHAT DID THE REWRITE ACTUALLY MOVE, AND DID IT HELP?")
print("    For each parent->child rewrite: which dimension changed most, and did latency drop?")
print("=" * 96)
print("%-24s %-24s %10s %10s  %-18s %s"
      % ("parent", "child", "parent ms", "child ms", "dim moved most", "helped?"))

labelled = []
for run, (best, trials, par, cls) in allruns.items():
    for cid, pid in sorted(par.items()):
        if cid not in best or pid not in best:
            continue
        pm, pp = best[pid]
        cm, cp = best[cid]
        if pid not in cls or cid not in cls:
            continue
        if cls[pid].get("pct_of_dram_peak") is None or cls[cid].get("pct_of_dram_peak") is None:
            continue
        up, _, _ = pressures(cls[pid])
        uc, _, _ = pressures(cls[cid])
        deltas = {d: (uc[d] - up[d]) for d in up
                  if up.get(d) is not None and uc.get(d) is not None}
        if not deltas:
            continue
        moved = max(deltas, key=lambda d: abs(deltas[d]))
        helped = cm < pm
        binding_parent = [d for d, v in up.items() if v is not None and v >= NEAR]
        labelled.append(dict(run=run.name, parent=pid, child=cid, pm=pm, cm=cm,
                             moved=moved, helped=helped,
                             binding_parent=binding_parent,
                             deltas=deltas, u_parent=up,
                             ev_parent=cls[pid], ev_child=cls[cid]))
        print("%-24s %-24s %10.4f %10.4f  %-18s %s"
              % (pid[:24], cid[:24], pm, cm, moved,
                 "yes %+.1f%%" % (100 * (cm - pm) / pm) if helped
                 else "no  %+.1f%%" % (100 * (cm - pm) / pm)))

multi_lab = [l for l in labelled if len(l["binding_parent"]) >= 2]
print()
print("    labelled rewrites: %d;  of those with >=2 binding dims in the parent: %d"
      % (len(labelled), len(multi_lab)))
if len(multi_lab) < 5:
    print("    *** TOO FEW TO SCORE ANY RULE. This is the finding, not a reason to invent one.")
    print("        A rule fitted on %d cases cannot be distinguished from chance." % len(multi_lab))

# ---------------- Q3: do any rules beat constant and chance? ----------------
print()
print("=" * 96)
print("Q3  DOES ANY RANKING RULE BEAT THE CONTROLS?")
print("    scored on the %d multi-binding rewrites: did the rule pick the dimension that the"
      % len(multi_lab))
print("    successful rewrite actually moved?  (only rewrites that HELPED can be scored)")
print("=" * 96)

# D-3: an improvement inside the noise floor is not an improvement. Per-trial std reaches 16%
# of the mean and near-tie bands span 5-9%, so scoring rules against -0.2% "successes" would be
# fitting to noise. --min-gain makes the threshold explicit and sweepable.
scoreable = [l for l in multi_lab if l["helped"]
             and 100 * (l["pm"] - l["cm"]) / l["pm"] >= args.min_gain]
inside_noise = [l for l in multi_lab if l["helped"] and l not in scoreable]
print("    multi-binding AND helped: %d;  of those, gain >= %.1f%%: %d  (%d inside the noise "
      "floor, excluded)" % (len(multi_lab), args.min_gain, len(scoreable), len(inside_noise)))
if inside_noise:
    print("      excluded gains: %s"
          % ", ".join("%.1f%%" % (100 * (l["pm"] - l["cm"]) / l["pm"]) for l in inside_noise))

if scoreable:
    # Sanity-check the LABEL before scoring any rule against it. Uniform random over a 2-3 way
    # choice should land 33-50%; far below that means the label is often outside the choice set,
    # i.e. the successful rewrite moved a dimension that was NOT binding. That is a finding
    # about the rewrites, not a verdict on the rules -- and scoring rules against an
    # out-of-set label punishes every rule equally while looking like evidence.
    inset = sum(1 for l in scoreable if l["moved"] in l["binding_parent"])
    print("    of those, the most-moved dimension WAS one of the binding ones: %d/%d (%.1f%%)"
          % (inset, len(scoreable), 100 * inset / len(scoreable)))
    if inset < len(scoreable):
        print("    *** %d successful rewrites moved a NON-binding dimension the most."
              % (len(scoreable) - inset))
        print("        No rule that picks from the binding set can match those, so every rate")
        print("        below is bounded by %.1f%% and must be read against that, not 100%%."
              % (100 * inset / len(scoreable)))
        for d, n in Counter(l["moved"] for l in scoreable
                            if l["moved"] not in l["binding_parent"]).most_common():
            print("          moved-most-but-not-binding: %-20s x%d" % (d, n))
    print()

    # --- robustness: "moved most" is an argmax ACROSS dimensions, and Delta-pressure has no
    # common unit between them (the same A-3 dimensional problem). A dimension whose pressure
    # naturally swings more will win that argmax regardless of relevance. So the fragile
    # question ("which one moved most") is re-asked in a form that never compares dimensions
    # to each other: did the rewrite RELIEVE any dimension that was binding?
    RELIEF = 0.05
    relieved_binding = relieved_none = 0
    for l in scoreable:
        drops = [v for v in l["deltas"].values() if v is not None and v <= -RELIEF]
        if any(l["deltas"].get(d) is not None and l["deltas"][d] <= -RELIEF
               for d in l["binding_parent"]):
            relieved_binding += 1
        elif not drops:
            relieved_none += 1
    print("    ROBUST FORM (no cross-dimension comparison): of the %d rewrites with a gain "
          ">= %.1f%%," % (len(scoreable), args.min_gain))
    print("      %2d relieved at least one BINDING dimension by >= %.2f pressure"
          % (relieved_binding, RELIEF))
    print("      %2d relieved nothing by that much (got faster without relieving any dimension)"
          % relieved_none)
    print("      %2d relieved only NON-binding dimensions"
          % (len(scoreable) - relieved_binding - relieved_none))
    if relieved_binding < len(scoreable) / 2:
        print("    => Most successful rewrites did NOT work by relieving a binding dimension.")
        print("       That undercuts A-1's premise ('which binding dimension to attack') more")
        print("       than it answers it -- and it holds without any cross-dimension argmax,")
        print("       so it does not rest on the fragile ranking above.")
    print()

    # --- so what DID the "relieved nothing" rewrites do? If a kernel gets materially faster
    # while every resource pressure stays put, it did not win by easing a wall -- it won by
    # doing LESS WORK (fewer bytes, fewer flops, fewer launches) or by using a cheaper
    # instruction mix. Those are visible in the classifier evidence, so the question is
    # answerable rather than speculative, and the answer decides what the finding means:
    # a resource-pressure framing that misses the dominant mechanism is a framing problem,
    # not a ranking problem.
    quiet = [l for l in scoreable
             if not any(l["deltas"].get(d) is not None and l["deltas"][d] <= -RELIEF
                        for d in l["binding_parent"])
             and not any(v <= -RELIEF for v in l["deltas"].values() if v is not None)]
    if quiet:
        print("    WHAT DID THE %d 'RELIEVED NOTHING' REWRITES CHANGE INSTEAD?" % len(quiet))
        print("      pressure deltas (+ = moved TOWARD the wall), then throughput deltas")
        print("      %-20s %6s %7s %7s %7s %7s %7s %9s"
              % ("child", "gain%", "dP_dram", "dP_comp", "dP_occ", "dP_reg", "d_AI", "d_TFLOPs"))
        toward = 0
        for l in quiet:
            pe, ce = l["ev_parent"], l["ev_child"]
            dp = l["deltas"]

            def p(dim):
                v = dp.get(dim)
                return ("%+7.2f" % v) if v is not None else "      -"

            def d(f):
                a, b = pe.get(f), ce.get(f)
                return ("%+7.2f" % (b - a)) if isinstance(a, (int, float)) \
                    and isinstance(b, (int, float)) else "      -"
            rises = [dim for dim, v in dp.items() if v is not None and v >= RELIEF]
            if rises:
                toward += 1
            print("      %-20s %5.1f%% %s %s %s %s %s %s   %s"
                  % (l["child"][:20], 100 * (l["pm"] - l["cm"]) / l["pm"],
                     p("dram_bandwidth"), p("compute"), p("occupancy"),
                     p("register_pressure"), d("arithmetic_intensity"), d("achieved_tflops"),
                     "toward: " + ",".join(rises) if rises else "flat"))
        print()
        print("      %d of %d moved at least one dimension TOWARD its wall by >= %.2f."
              % (toward, len(quiet), RELIEF))
        if toward >= len(quiet) / 2:
            print("      => These rewrites did not RELIEVE a constraint, they FILLED a slack")
            print("         resource: throughput rose on a dimension that had room. That is")
            print("         the user's 'aligning the operator's resource profile with the")
            print("         hardware's' -- and it is the OPPOSITE of what A-1 assumes the")
            print("         agent should be told to do. A-1 asks which wall to push away;")
            print("         the data says the winning move is usually which gap to fill.")
    print()

    def r1(l):   # highest pressure = tightest wall
        c = {d: l["u_parent"][d] for d in l["binding_parent"]}
        return max(c, key=lambda d: c[d])

    def r2(l):   # lowest pressure among the binding set = most remaining room
        c = {d: l["u_parent"][d] for d in l["binding_parent"]}
        return min(c, key=lambda d: c[d])

    rng = random.Random(0)

    def c2(l):   # chance
        return rng.choice(l["binding_parent"])

    rules = {"R1 tightest wall first": r1,
             "R2 most remaining room": r2}
    # C1: every constant choice, so we report the BEST constant -- the honest bar
    dims_seen = sorted({d for l in scoreable for d in l["binding_parent"]})
    for d in dims_seen:
        rules["C1 always " + d] = (lambda dd: (lambda l: dd if dd in l["binding_parent"]
                                               else l["binding_parent"][0]))(d)
    rules["C2 uniform random"] = c2

    print()
    print("    %-34s %8s %8s" % ("rule", "correct", "rate"))
    results = {}
    for name, fn in rules.items():
        hit = sum(1 for l in scoreable if fn(l) == l["moved"])
        results[name] = hit
        print("    %-34s %8d %7.1f%%" % (name, hit, 100 * hit / len(scoreable)))

    best_ctrl = max((v for k, v in results.items() if k.startswith("C")), default=0)
    best_rule = max((v for k, v in results.items() if k.startswith("R")), default=0)
    print()
    print("    best control %d/%d (%.1f%%);  best rule %d/%d (%.1f%%)"
          % (best_ctrl, len(scoreable), 100 * best_ctrl / len(scoreable),
             best_rule, len(scoreable), 100 * best_rule / len(scoreable)))
    if best_rule <= best_ctrl:
        print("    => NO RULE BEATS THE CONTROLS. The honest answer to A-1 on this evidence is")
        print("       'we have no basis for ranking' -- report that, do not invent a rule.")
    else:
        print("    => a rule leads by %d cases. With n=%d that is %s."
              % (best_rule - best_ctrl, len(scoreable),
                 "not enough to claim anything" if len(scoreable) < 20
                 else "worth a prospective test"))
else:
    print("    nothing scoreable. A-1 cannot be settled from the historical record.")

print()
print("=" * 96)
print("WHAT THIS CAN AND CANNOT ANSWER")
print("=" * 96)
print("  CAN:  how often multiple dimensions bind; whether the record contains enough")
print("        decisions to score a rule at all; whether any rule beats constant and chance.")
print("  CANNOT: which dimension SHOULD have been chosen -- we only see the one that was.")
print("        A rewrite that moved dimension X and helped does not prove X was the best")
print("        choice; the counterfactual was never run. So a high score here is suggestive,")
print("        and a low score is decisive.")
