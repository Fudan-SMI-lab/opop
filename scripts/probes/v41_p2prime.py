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

THE COMPARISON, and why the cutoff matters more than anything else here. The legacy
predictor is `stats.py`'s `latency_by_value`: a per-choice MEDIAN over whatever trials TPE
happened to run. Reading it from the FINISHED run and comparing against g_d would compare
two different information sets -- the marginal would have the whole run's trials, the
conditioned only its four points -- and would flatter whichever side got more data. So the
marginal is recomputed from the trials that existed AT THE C4's OWN CUTOFF (event sequence
< the C4's SCAN_BLOCK_DONE), on the same axis, and both predict the SAME holdout y.

Three ways the marginal can decline to answer, kept separate because they are different
findings: the axis has fewer than two measured values at the cutoff (no slope exists),
the F/N values themselves were never sampled (the marginal cannot address this contrast),
or a bucket is a single trial (a "median" of one). Counting those as errors would let the
conditioned side win by default.

Usage:
    python scripts/probes/v41_p2prime.py <run_dir> [<run_dir> ...]

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
        self.g_m: float | None = None
        self.g_m_skip: str | None = None


def _marginal_slope(trials: list[dict], axis: str, f_repr: str, n_repr: str
                    ) -> tuple[float | None, str | None]:
    """The LEGACY predictor, recomputed from the trials available at a cutoff.

    `stats.py` builds `latency_by_value` as a median per choice over completed trials --
    unconditioned on the partners, which is the defect v4.1 exists to fix. Reproduced here
    rather than imported because `TuningStatsAnalyzer` needs a live ParameterSpace and the
    point is to read exactly the same arithmetic on a historical prefix.

    Endpoints arrive as `repr` strings from the event (a value can be bool/int/str and JSON
    collapses True/1), so trial values are compared through `repr` too.

    Returns (slope, None) or (None, reason). Same sign convention as the conditioned g:
    positive means the N (near-wall) side is SLOWER than the F side.
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
    if f_med <= 0:
        return None, "nonpositive_median"
    return (n_med - f_med) / n_med, None


def analyse(run_dir: Path) -> dict:
    contrasts: list[Contrast] = []
    trials: list[dict] = []
    prefix_at: dict[str, list[dict]] = {}
    # Fallback for runs journalled before SCAN_BLOCK_DONE carried the endpoints: the wall
    # a C4 was built from names them, and UW_PROBE_BATCH.walls[] reprs them already. Keyed
    # by (space_id, axis) because that is all both events share.
    wall_endpoints: dict[tuple, tuple[str, str]] = {}

    for seq, ev in enumerate(read_events(run_dir)):
        t = ev.get("type")
        p = ev.get("payload", {})
        if t == "TRIAL_DONE":
            rec = p.get("trial") or p
            if rec.get("status") == "complete":
                trials.append(rec)
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
            prefix_at[c.scan_id] = list(trials)      # the cutoff: trials BEFORE this close
            contrasts.append(c)

    for c in contrasts:
        if c.axis and c.f_repr and c.n_repr:
            c.g_m, c.g_m_skip = _marginal_slope(prefix_at[c.scan_id], c.axis,
                                                c.f_repr, c.n_repr)
        else:
            # Not a measurement outcome: the journal simply does not name the endpoints.
            c.g_m_skip = "endpoints_not_journalled"

    return {"run": run_dir.name, "contrasts": contrasts, "n_trials": len(trials)}


def _report(res: dict) -> tuple[int, int]:
    cs: list[Contrast] = res["contrasts"]
    full = [c for c in cs if c.full and c.g_d is not None and c.y is not None]
    print(f"\n=== {res['run']} ===")
    print(f"completed trials: {res['n_trials']}   C4 blocks: {len(cs)}   "
          f"fresh-complete: {len(full)}")
    if not full:
        print("no complete contrast: P2' is unreadable in this run (not a negative result)")
        return 0, 0

    print("\n  contrast          axis            g_d        y      |err_c|   "
          "g_m        |err_m|   winner")
    both = 0
    cond_wins = 0
    for c in full:
        err_c = abs(c.g_d - c.y)
        if c.g_m is None:
            print(f"  {c.scan_id[:16]:16s} {str(c.axis)[:14]:14s} "
                  f"{c.g_d:8.4f} {c.y:8.4f} {err_c:8.4f}   "
                  f"{'-':9s} {'-':8s}  ({c.g_m_skip})")
            continue
        err_m = abs(c.g_m - c.y)
        both += 1
        win = "conditioned" if err_c < err_m else ("marginal" if err_m < err_c else "tie")
        if err_c < err_m:
            cond_wins += 1
        print(f"  {c.scan_id[:16]:16s} {str(c.axis)[:14]:14s} "
              f"{c.g_d:8.4f} {c.y:8.4f} {err_c:8.4f}   "
              f"{c.g_m:9.4f} {err_m:8.4f}  {win}")

    # Sign agreement is the weaker but more robust reading: it survives when the
    # magnitudes are inside the noise floor.
    sign_c = sum(1 for c in full if (c.g_d > 0) == (c.y > 0))
    print(f"\n  sign agreement (conditioned vs holdout): {sign_c}/{len(full)}")
    sign_m = [c for c in full if c.g_m is not None]
    if sign_m:
        agree_m = sum(1 for c in sign_m if (c.g_m > 0) == (c.y > 0))
        print(f"  sign agreement (marginal vs holdout):    {agree_m}/{len(sign_m)}")
    if both:
        med_c = median([abs(c.g_d - c.y) for c in full if c.g_m is not None])
        med_m = median([abs(c.g_m - c.y) for c in full if c.g_m is not None])
        print(f"  median |error|: conditioned {med_c:.4f}  vs  marginal {med_m:.4f}"
              f"   (n={both} pairs)")
        print(f"  conditioned closer on {cond_wins}/{both}")
    else:
        print("  the marginal could not be computed for ANY contrast at its own cutoff:")
        skips = defaultdict(int)
        for c in full:
            skips[c.g_m_skip] += 1
        for k, v in sorted(skips.items()):
            print(f"    {k}: {v}")
        print("  => the conditioned side has no competitor here. Report as "
              "'marginal undefined at cutoff', NOT as a conditioned win.")

    # Directions actually exercised: the inward branch was unreachable before v4.1.
    dirs = defaultdict(int)
    for c in full:
        dirs[c.direction or "unresolved"] += 1
    print(f"  mint directions: {dict(dirs)}")
    return both, cond_wins


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tot_both = tot_wins = 0
    for arg in sys.argv[1:]:
        b, w = _report(analyse(Path(arg)))
        tot_both += b
        tot_wins += w
    if len(sys.argv) > 2:
        print(f"\n=== pooled (layers kept separate above; pool only for a headline) ===")
        if tot_both:
            print(f"conditioned closer on {tot_wins}/{tot_both} comparable contrasts")
        else:
            print("no contrast had a computable marginal competitor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
