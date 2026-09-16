"""Offline P2': compare role-keyed conditioned and marginal predictions of the same y.

old/signfix/scoped are RETRACTED diagnostic stages, not robustness arms: old retains
(N-F)/N and run-wide pooling; signfix uses 1-N/F; scoped restricts to the candidate.
All three retain the block's own scan trials at SCAN_BLOCK_DONE for reproducibility.
noleak (default) removes ALL own scan trial IDs, discovery included, at block close.
It compares paid local discovery with a candidate's historical prefix across all spaces;
it is a reportable historical comparison, not a universally matched-information test.
discovery_cutoff instead includes candidate history through the last discovery TRIAL_DONE
(inclusive), including discovery data for the marginal. It requires four distinct role
identities and both validation completions strictly later; otherwise it declines.
Completion order is journal evidence, not proof of physical execution start times.

Pools retain record multiplicity. Reused flags are coverage metadata; an unmarked record
is NOT independently verified fresh. Missing marginals are missing denominators, not
conditioned wins. Changes in n establish eligibility, not complete holdout dependence.
Neither these definitions nor synthetic reader tests establish general conditioned wins.
Runs are reported separately; pooled contrasts are not independent search repeats.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kernel_optimizer.store.read import read_events  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.probes.v41_p2prime_history import (  # noqa: E402
    DEFINITION_INFO, Coverage, Event, Payload, PoolMetadata, Summary, Trial, collect_prefixes,
)

DEFINITIONS = ("old", "signfix", "scoped", "noleak", "discovery_cutoff")
DEFAULT_DEFINITION = "noleak"


def _typed(v: object) -> str:
    """Kept for the wall-payload fallback: `ConditionedWall.payload()` already reprs its
    endpoints, so both sides of a comparison go through `repr`."""
    return repr(v)


class Contrast:
    """One C4 block's prediction and holdout, plus the marginal recomputed at its cutoff."""

    def __init__(self, payload: Payload, seq: int) -> None:
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
        # Separate slots retain the historical analyses without overwriting them.
        self.g_m: dict[str, float | None] = {d: None for d in DEFINITIONS}
        self.g_m_skip: dict[str, str | None] = {d: None for d in DEFINITIONS}
        self.n_pool: dict[str, int] = {}
        self.n_own_leaked = 0
        self.pool_metadata: dict[str, PoolMetadata] = {}


class Analysis(TypedDict):
    run: str
    contrasts: list[Contrast]
    n_trials: int
    coverage: Coverage
    definitions: dict[str, Summary]


def _marginal_slope(trials: Sequence[Trial], axis: str, f_repr: str, n_repr: str,
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


def analyse(run_dir: Path) -> Analysis:
    contrasts: list[Contrast] = []
    events = [Event(type=ev.get("type", ""), payload=ev.get("payload") or {})
              for ev in read_events(run_dir)]
    prefixes, coverage = collect_prefixes(events)
    for prefix in prefixes:
        c = Contrast(prefix["payload"], prefix["seq"])
        c.n_own_leaked = prefix["n_own_leaked"]
        c.pool_metadata = prefix["metadata"]
        for d in DEFINITIONS:
            pool = [r.trial for r in prefix["pools"][d]]
            c.n_pool[d] = len(pool)
            if c.pool_metadata[d]["cutoff_reason"] is not None:
                c.g_m_skip[d] = "information_cutoff_unavailable"
                continue
            if not (c.axis and c.f_repr and c.n_repr):
                # Not a measurement outcome: the journal simply does not name the endpoints.
                c.g_m_skip[d] = "endpoints_not_journalled"
                continue
            c.g_m[d], c.g_m_skip[d] = _marginal_slope(
                pool, c.axis, c.f_repr, c.n_repr, legacy_sign=(d == "old"))
        contrasts.append(c)
    full = [c for c in contrasts if c.full and c.g_d is not None and c.y is not None]
    return {"run": run_dir.name, "contrasts": contrasts, "n_trials": coverage["raw_records"],
            "coverage": coverage, "definitions": {d: _summary(full, d) for d in DEFINITIONS}}


def _summary(full: list[Contrast], definition: str) -> Summary:
    """Median errors, sign agreement and denominators for one definition."""
    both = [(g, y, m) for c in full for g, y, m in [(c.g_d, c.y, c.g_m[definition])]
            if g is not None and y is not None and m is not None]
    missing: dict[str, int] = defaultdict(int)
    for c in full:
        if c.g_m[definition] is None:
            missing[c.g_m_skip[definition] or "unspecified"] += 1
    metadata: Summary = {**DEFINITION_INFO[definition], "eligible_contrasts": len(full),
                "missing_reasons": dict(missing), "ratio": None,
                "cutoff_inclusive": True, "deduplicated": False, "n": 0,
                "med_c": None, "med_m": None, "cond_closer": 0, "sign_m": 0}
    if not both:
        return {**metadata, "n": 0}
    err_c = [abs(g - y) for g, y, m in both]
    err_m = [abs(m - y) for g, y, m in both]
    return {
        **metadata,
        "n": len(both),
        "med_c": median(err_c), "med_m": median(err_m),
        "ratio": (median(err_m) / median(err_c)) if median(err_c) else None,
        "cond_closer": sum(1 for g, y, m in both if abs(g - y) < abs(m - y)),
        "sign_m": sum(1 for g, y, m in both if (m > 0) == (y > 0)),
    }


def _report(res: Analysis, definition: str) -> Summary:
    cs: list[Contrast] = res["contrasts"]
    full = [c for c in cs if c.full and c.g_d is not None and c.y is not None]
    print(f"\n=== {res['run']} ===")
    print(f"completed trials: {res['n_trials']}   C4 blocks: {len(cs)}   "
          f"full with prediction/holdout: {len(full)}")
    print(f"completed-record coverage (unmarked is not verified fresh): {res['coverage']}")
    if not full:
        print("no complete contrast: P2' is unreadable in this run (not a negative result)")
        return res["definitions"][definition]

    leaked = [c.n_own_leaked for c in full]
    if any(leaked):
        print(f"own scan trials excluded from the marginal history: "
              f"min {min(leaked)} max {max(leaked)} (they include this contrast's own "
               f"validation pair and discovery pair under noleak)")

    print(f"\n  selected definition: {definition} ({DEFINITION_INFO[definition]['status']})")
    print("  noleak: paid local discovery vs historical prefix; not matched information.")
    print("  old/signfix/scoped: retracted diagnostics, not robustness arms.")
    print("\n  contrast          axis            g_d        y      |err_c|   "
          "g_m        |err_m|   winner    pool")
    for c, g, y in [(c, c.g_d, c.y) for c in full if c.g_d is not None and c.y is not None]:
        err_c = abs(g - y)
        gm = c.g_m[definition]
        if gm is None:
            print(f"  {str(c.scan_id)[:16]:16s} {str(c.axis)[:14]:14s} "
                  f"{g:8.4f} {y:8.4f} {err_c:8.4f}   "
                  f"{'-':9s} {'-':8s}  ({c.g_m_skip[definition]})")
            continue
        err_m = abs(gm - y)
        win = "conditioned" if err_c < err_m else ("marginal" if err_m < err_c else "tie")
        print(f"  {str(c.scan_id)[:16]:16s} {str(c.axis)[:14]:14s} "
              f"{g:8.4f} {y:8.4f} {err_c:8.4f}   "
              f"{gm:9.4f} {err_m:8.4f}  {win:11s} {c.n_pool.get(definition, 0)}")

    # Sign agreement is the weaker but more robust reading: it survives when the
    # magnitudes are inside the noise floor.
    sign_c = sum(1 for c in full if c.g_d is not None and c.y is not None
                 and (c.g_d > 0) == (c.y > 0))
    print(f"\n  sign agreement (conditioned vs holdout): {sign_c}/{len(full)}")

    print("\n  -- every definition, so no single number hides which defect moved it --")
    print("  %-9s %4s  %-11s %-11s %-8s %-9s %s"
          % ("defn", "n", "med|err_m|", "med|err_c|", "ratio", "cond wins", "marginal sign"))
    out = {}
    for d in DEFINITIONS:
        s = res["definitions"][d]
        out[d] = s
        if not s["n"]:
            skips = s["missing_reasons"]
            print("  %-9s %4d  never computable: %s"
                  % (d, 0, ", ".join(f"{k}={v}" for k, v in sorted(skips.items()))))
            continue
        star = " <== selected" if d == definition else ""
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
    return out[definition]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", choices=DEFINITIONS, default=DEFAULT_DEFINITION)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    definition = args.definition
    tot_both = tot_wins = 0
    for run_dir in args.run_dirs:
        s = _report(analyse(run_dir), definition)
        tot_both += s.get("n", 0)
        tot_wins += s.get("cond_closer", 0)
    if len(args.run_dirs) > 1:
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
