"""P2': is the CONDITIONED slope a better out-of-sample predictor than the MARGINAL one?

THE PRIMARY SCIENTIFIC ENDPOINT of the v4.1 experiments, and the one that answers the
novelty threat: the baseline analyst already produces "ceiling + slope + rewrite" shaped
reasoning by itself (memory: the-baseline-analyst-already-produces-the-c2-shape), so C2
cannot claim a new capability. It can claim the number is QUANTITATIVELY BETTER, and this
is the direct measurement of that.

WHAT MAKES IT A REAL TEST. Every C4 block measures four fresh latencies under FROZEN
partners, in two role-keyed pairs:

    discovery pair  (F_d, N_d) -> g_d, the PREDICTION
    validation pair (F_v, N_v) -> y,   the HOLDOUT

g_d and y are the same quantity measured twice on disjoint measurements, so |g_d - y| is
an honest out-of-sample error. Nothing about the prediction was fitted to the validation
pair; the roles were frozen by the geometry before either pair ran, and the four points
execute in RANDOMIZED order so arrival position carries no role information.

THE COMPARISON. The legacy predictor is `stats.py`'s `latency_by_value`: a per-choice
MEDIAN over whatever trials TPE happened to run, unconditioned on the partners. Reading it
from the FINISHED run would compare two different information sets, so it is recomputed
from the trials available at each C4's own cutoff, and both sides predict the SAME y.

THREE DEFECTS THIS READER HAD, all found by external review on 2026-09-15, all real, all
measured before being fixed (docs/audit-p2prime-reader-defects.md). They are documented
here because each one has a "obvious" wrong version that a future edit could reintroduce:

  1. SIGN AND DENOMINATOR. `scanner.py` computes g = 1 - N/F, positive when the near-wall
     end is FASTER. This reader returned (n_med - f_med)/n_med -- opposite sign AND a
     different denominator -- under a docstring claiming the same convention. The two
     disagreed in sign on 12 of 12 live contrasts. Percentages must never be naively
     negated: the exact algebra for a reversed ratio is -g/(1-g), and the fix here is to
     compute `1 - n/f` directly instead.
  2. POOL SCOPE. Every completed trial in the RUN went into one list, bucketed by axis name
     with no candidate filter, so one candidate's `BK=32` was averaged with another
     implementation's `BK=32`. On the later contrasts 90-98% of the "marginal" came from
     other candidates. The legacy analyst's actual scope is `crun.trials`, which
     accumulates across a candidate's space expansions and is never cleared
     (orchestrator.py:1333, :2461) and is passed to `stats_analyzer.analyze` at :1033,
     :1504 and :1959 -- so the faithful scope is THIS CANDIDATE, ALL ITS SPACES. Not "this
     space" (which would understate the legacy baseline) and not "the whole run".
  3. HOLDOUT LEAK. `TRIAL_DONE` is journalled at orchestrator.py:1324, BEFORE the scan step
     at :1335 folds the result in and emits `SCAN_BLOCK_DONE`. A prefix cut at
     SCAN_BLOCK_DONE therefore already contains the C4's own four trials -- the validation
     pair included, which is what y is computed from. Cutting on event order is not cutting
     on information. Every contrast in both pilots leaked all four of its own points.

So the marginal is reported under FOUR definitions rather than one. Publishing a single
"corrected" number would hide which change moved it, and the two runs move in opposite
directions (l3:43 50.9x -> 20.3x while l3:21 25.8x -> 104.8x), so only the direction and
the range survive as a claim -- never a point estimate without its definition and its n.

Three ways the marginal can decline to answer, kept separate because they are different
findings: the axis has fewer than two measured values at the cutoff (no slope exists),
the F/N values themselves were never sampled (the marginal cannot address this contrast),
or a bucket is a single trial (a "median" of one). Counting those as errors would let the
conditioned side win by default. Dropping the leaked points MAKES MORE contrasts decline
(12 -> 7 and 11 -> 9), which is itself the finding that those marginals existed only
because of the leak -- reported as a missing denominator, never as a conditioned win.

Usage:
    python scripts/probes/v41_p2prime.py <run_dir> [<run_dir> ...]
    python scripts/probes/v41_p2prime.py --definition old <run_dir>    # reproduce a
                                                                       # historical table

Runs are read separately and reported per run AND pooled, never silently merged: box1 and
box4 have different CPUs and the two tasks are different venues.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kernel_optimizer.store.read import read_events  # noqa: E402

# The four marginal definitions, weakest to strongest. `noleak` is the reportable one;
# the others exist so a reader can see which defect moved which number, and so the
# historical tables remain reproducible instead of being quietly overwritten.
DEFINITIONS = ("old", "signfix", "scoped", "noleak")
DEFAULT_DEFINITION = "noleak"


def _typed(v: object) -> str:
    """Kept for the wall-payload fallback: `ConditionedWall.payload()` already reprs its
    endpoints, so both sides of a comparison go through `repr`."""
    return repr(v)


class Contrast:
    """One C4 block's prediction and holdout, plus the marginal recomputed at its cutoff."""

    def __init__(self, payload: dict, seq: int) -> None:
        self.seq = seq
        self.candidate_id = payload.get("candidate_id")
        self.scan_id = payload.get("scan_id")
        self.axis = payload.get("axis")
        self.g_d = payload.get("g_d")
        self.y = payload.get("y")
        self.full = bool(payload.get("full"))
        self.direction = payload.get("direction")
        self.f_lat = payload.get("f_lat") or []
        self.n_lat = payload.get("n_lat") or []
        self.tau = payload.get("tau")
        self.rho_f = payload.get("rho_f")
        self.rho_n = payload.get("rho_n")
        # The frozen endpoints, as `repr` strings. Present from the commit that added
        # axis_f_value/axis_n_value to SCAN_BLOCK_DONE; runs journalled before it have
        # neither, and this reads as "marginal not computable" rather than as a result.
        self.f_repr = payload.get("axis_f_value")
        self.n_repr = payload.get("axis_n_value")
        # One slot per definition, so nothing is overwritten and the table can show all four.
        self.g_m: dict[str, float | None] = {d: None for d in DEFINITIONS}
        self.g_m_skip: dict[str, str | None] = {d: None for d in DEFINITIONS}
        self.n_pool: dict[str, int] = {}
        self.n_own_leaked = 0


def _marginal_slope(trials: list[dict], axis: str, f_repr: str, n_repr: str,
                    *, legacy_sign: bool = False) -> tuple[float | None, str | None]:
    """The LEGACY predictor, recomputed from the trials available at a cutoff.

    `stats.py` builds `latency_by_value` as a median per choice over completed trials --
    unconditioned on the partners, which is the defect v4.1 exists to fix. Reproduced here
    rather than imported because `TuningStatsAnalyzer` needs a live ParameterSpace and the
    point is to read exactly the same arithmetic on a historical prefix.

    Endpoints arrive as `repr` strings from the event (a value can be bool/int/str and JSON
    collapses True/1), so trial values are compared through `repr` too.

    Returns (slope, None) or (None, reason). The slope is `1 - n_med/f_med`: the SAME
    convention as `scanner.py`'s g_d = 1 - N/F, so positive means the near-wall (N) end is
    FASTER. `legacy_sign=True` reproduces the retracted `(n-f)/n` form -- kept only so a
    historical table can be regenerated, never for a new claim.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        params = (t.get("params") or {}).get("values") or {}
        if axis not in params:
            continue
        lat = t.get("latency_ms") or {}
        ms = lat.get("median")
        if ms is None:
            ms = lat.get("mean")           # keys are `median`/`mean`, no `_ms` suffix
        if ms is None or ms <= 0:
            continue
        buckets[repr(params[axis])].append(float(ms))

    if len(buckets) < 2:
        return None, "no_slope_at_cutoff"
    if f_repr not in buckets or n_repr not in buckets:
        return None, "values_unsampled_at_cutoff"
    if len(buckets[f_repr]) < 2 or len(buckets[n_repr]) < 2:
        # A "median" of one trial is a single noisy sample wearing a robust name.
        return None, "thin_bucket_at_cutoff"
    f_med, n_med = median(buckets[f_repr]), median(buckets[n_repr])
    if f_med <= 0 or n_med <= 0:
        return None, "nonpositive_median"
    if legacy_sign:
        return (n_med - f_med) / n_med, None
    return 1.0 - n_med / f_med, None


def analyse(run_dir: Path) -> dict:
    contrasts: list[Contrast] = []
    # Three pools, because the three defects need three different histories.
    run_pool: list[dict] = []                          # defect 2's wrong scope, for `old`
    cand_pool: dict[str, list[dict]] = defaultdict(list)   # the legacy scope
    prefix_at: dict[str, dict[str, list[dict]]] = {}
    # Which trial_ids belong to which scan block, so a C4's own points can be excluded by
    # IDENTITY rather than by guessing from arrival order (roles are randomized).
    own_trials: dict[str, set[str]] = defaultdict(set)
    # Fallback for runs journalled before SCAN_BLOCK_DONE carried the endpoints: the wall
    # a C4 was built from names them, and UW_PROBE_BATCH.walls[] reprs them already. Keyed
    # by (space_id, axis) because that is all both events share.
    wall_endpoints: dict[tuple, tuple[str, str]] = {}

    events = read_events(run_dir)
    # Pass 1: scan-point -> trial_id map. Needed before the main pass because a block's
    # SCAN_POINT_DONE events can be interleaved with the trials of other blocks.
    for ev in events:
        p = ev.get("payload") or {}
        if ev.get("type") == "SCAN_POINT_DONE" and p.get("trial_id"):
            own_trials[str(p.get("scan_id"))].add(str(p["trial_id"]))

    for seq, ev in enumerate(events):
        t = ev.get("type")
        p = ev.get("payload", {})
        if t == "TRIAL_DONE":
            rec = p.get("trial") or p
            if rec.get("status") == "complete":
                run_pool.append(rec)
                cand_pool[str(rec.get("candidate_id"))].append(rec)
        elif t == "UW_PROBE_BATCH":
            for w in p.get("walls", []):
                if w.get("f_value") is not None and w.get("n_value") is not None:
                    wall_endpoints[(p.get("space_id"), w.get("axis"))] = (
                        w["f_value"], w["n_value"])
        elif t == "SCAN_BLOCK_ADMITTED" and p.get("kind") == "C4":
            # space_id lives here, not on SCAN_BLOCK_DONE, so the wall fallback needs it.
            wall_endpoints.setdefault(("scan:" + str(p.get("scan_id")), p.get("axis")),
                                      wall_endpoints.get(
                                          (p.get("space_id"), p.get("axis")), (None, None)))
        elif t == "SCAN_BLOCK_DONE" and p.get("kind") == "C4":
            c = Contrast(p, seq)
            if c.f_repr is None or c.n_repr is None:
                fb = wall_endpoints.get(("scan:" + str(c.scan_id), c.axis), (None, None))
                c.f_repr, c.n_repr = fb
            cid = str(c.candidate_id)
            own = own_trials.get(str(c.scan_id), set())
            cand_hist = list(cand_pool[cid])
            # The leak-free prefix: this candidate's history MINUS this contrast's own four
            # points. Cutting on event order alone leaves them in (orchestrator writes
            # TRIAL_DONE before SCAN_BLOCK_DONE), which puts y's own measurements into the
            # competitor's training data.
            noleak = [x for x in cand_hist if str(x.get("trial_id")) not in own]
            c.n_own_leaked = len(cand_hist) - len(noleak)
            prefix_at[c.scan_id] = {
                "old": list(run_pool), "signfix": list(run_pool),
                "scoped": cand_hist, "noleak": noleak,
            }
            contrasts.append(c)

    for c in contrasts:
        for d in DEFINITIONS:
            if not (c.axis and c.f_repr and c.n_repr):
                # Not a measurement outcome: the journal simply does not name the endpoints.
                c.g_m_skip[d] = "endpoints_not_journalled"
                continue
            pool = prefix_at[c.scan_id][d]
            c.n_pool[d] = len(pool)
            c.g_m[d], c.g_m_skip[d] = _marginal_slope(
                pool, c.axis, c.f_repr, c.n_repr, legacy_sign=(d == "old"))

    return {"run": run_dir.name, "contrasts": contrasts, "n_trials": len(run_pool)}


def _summary(full: list[Contrast], definition: str) -> dict:
    """Median errors, sign agreement and denominators for one definition."""
    both = [c for c in full if c.g_m[definition] is not None]
    if not both:
        return {"n": 0}
    err_c = sorted(abs(c.g_d - c.y) for c in both)
    err_m = sorted(abs(c.g_m[definition] - c.y) for c in both)
    med = lambda a: a[len(a) // 2] if len(a) % 2 else (a[len(a) // 2 - 1] + a[len(a) // 2]) / 2
    return {
        "n": len(both),
        "med_c": med(err_c), "med_m": med(err_m),
        "ratio": (med(err_m) / med(err_c)) if med(err_c) else None,
        "cond_closer": sum(1 for c in both if abs(c.g_d - c.y) < abs(c.g_m[definition] - c.y)),
        "sign_m": sum(1 for c in both if (c.g_m[definition] > 0) == (c.y > 0)),
    }


def _report(res: dict, definition: str) -> dict:
    cs: list[Contrast] = res["contrasts"]
    full = [c for c in cs if c.full and c.g_d is not None and c.y is not None]
    print(f"\n=== {res['run']} ===")
    print(f"completed trials: {res['n_trials']}   C4 blocks: {len(cs)}   "
          f"fresh-complete: {len(full)}")
    if not full:
        print("no complete contrast: P2' is unreadable in this run (not a negative result)")
        return {"n": 0}

    leaked = [c.n_own_leaked for c in full]
    if any(leaked):
        print(f"own scan trials excluded from the marginal history: "
              f"min {min(leaked)} max {max(leaked)} (they include this contrast's own "
              f"validation pair -- see the module docstring, defect 3)")

    print(f"\n  reportable definition: {definition}   "
          f"(all four shown; `old` is the RETRACTED form)")
    print("\n  contrast          axis            g_d        y      |err_c|   "
          "g_m        |err_m|   winner    pool")
    for c in full:
        err_c = abs(c.g_d - c.y)
        gm = c.g_m[definition]
        if gm is None:
            print(f"  {c.scan_id[:16]:16s} {str(c.axis)[:14]:14s} "
                  f"{c.g_d:8.4f} {c.y:8.4f} {err_c:8.4f}   "
                  f"{'-':9s} {'-':8s}  ({c.g_m_skip[definition]})")
            continue
        err_m = abs(gm - c.y)
        win = "conditioned" if err_c < err_m else ("marginal" if err_m < err_c else "tie")
        print(f"  {c.scan_id[:16]:16s} {str(c.axis)[:14]:14s} "
              f"{c.g_d:8.4f} {c.y:8.4f} {err_c:8.4f}   "
              f"{gm:9.4f} {err_m:8.4f}  {win:11s} {c.n_pool.get(definition, 0)}")

    # Sign agreement is the weaker but more robust reading: it survives when the
    # magnitudes are inside the noise floor.
    sign_c = sum(1 for c in full if (c.g_d > 0) == (c.y > 0))
    print(f"\n  sign agreement (conditioned vs holdout): {sign_c}/{len(full)}")

    print("\n  -- every definition, so no single number hides which defect moved it --")
    print("  %-9s %4s  %-11s %-11s %-8s %-9s %s"
          % ("defn", "n", "med|err_m|", "med|err_c|", "ratio", "cond wins", "marginal sign"))
    out = {}
    for d in DEFINITIONS:
        s = _summary(full, d)
        out[d] = s
        if not s["n"]:
            skips = defaultdict(int)
            for c in full:
                skips[c.g_m_skip[d]] += 1
            print("  %-9s %4d  never computable: %s"
                  % (d, 0, ", ".join(f"{k}={v}" for k, v in sorted(skips.items()))))
            continue
        star = " <== reportable" if d == definition else ("  (RETRACTED)" if d == "old" else "")
        print("  %-9s %4d  %-11.4f %-11.4f %-8s %d/%-7d %d/%d%s"
              % (d, s["n"], s["med_m"], s["med_c"],
                 ("%.1fx" % s["ratio"]) if s["ratio"] else "-",
                 s["cond_closer"], s["n"], s["sign_m"], s["n"], star))

    # Declines, by reason, for the reportable definition: a missing marginal is a missing
    # denominator, never a conditioned win.
    skips = defaultdict(int)
    for c in full:
        if c.g_m[definition] is None:
            skips[c.g_m_skip[definition]] += 1
    if skips:
        print(f"\n  marginal DECLINED under `{definition}` "
              f"({sum(skips.values())}/{len(full)}), by reason:")
        for k, v in sorted(skips.items()):
            print(f"    {k}: {v}")
        print("  => report these as a MISSING DENOMINATOR, not as a conditioned win.")

    # Directions actually exercised: the inward branch was unreachable before v4.1.
    dirs = defaultdict(int)
    for c in full:
        dirs[c.direction or "unresolved"] += 1
    print(f"  mint directions: {dict(dirs)}")
    return out.get(definition, {"n": 0})


def main() -> int:
    args = [a for a in sys.argv[1:]]
    definition = DEFAULT_DEFINITION
    if "--definition" in args:
        i = args.index("--definition")
        definition = args[i + 1]
        del args[i:i + 2]
        if definition not in DEFINITIONS:
            print(f"--definition must be one of {DEFINITIONS}")
            return 2
    if not args:
        print(__doc__)
        return 2
    tot_both = tot_wins = 0
    for arg in args:
        s = _report(analyse(Path(arg)), definition)
        tot_both += s.get("n", 0)
        tot_wins += s.get("cond_closer", 0)
    if len(args) > 1:
        print(f"\n=== pooled under `{definition}` "
              f"(layers kept separate above; pool only for a headline) ===")
        if tot_both:
            print(f"conditioned closer on {tot_wins}/{tot_both} comparable contrasts")
            print("NB: contrasts from the same run share a candidate and a venue -- this is "
                  "not a count of independent search repeats.")
        else:
            print("no contrast had a computable marginal competitor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
