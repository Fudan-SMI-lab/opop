"""Why do the two S7 arms differ in trial COUNT, and is the cause the mechanism?

THE QUESTION THIS ANSWERS. At 51 min the treatment arm had 8 trials and the control 35. A 4.4x rate
gap between two arms that differ in one config key is the shape a confound takes, and the pair exists
to remove exactly that ambiguity -- so the gap has to be attributed before either arm's latency can be
read as a result.

THE MECHANISM IS RULED OUT FIRST, and cheaply: `SLOPE_GUIDE_STEP` count is 0 in both arms. S7 has not
fired once, so whatever is slowing the treatment arm is not S7. That leaves two candidates, and they
have opposite consequences:

  * a TAIL -- a few pathological configs TPE happened to draw in one arm. Does not compound: the arms
    are tuning DIFFERENT candidates (different generator seeds), so their config draws are not paired
    and a tail is expected variance between two independent searches.
  * a shifted BODY -- every trial slower. Would compound over 12 h into unequal search, which is the
    confound `check_arm_search_parity.py` exists to name.

The discriminator is the MEDIAN of the per-trial cost, robust to a bounded number of outliers by
construction. `job_wall_s` is read from the trial record rather than from inter-event gaps, because
`max_shared_jobs: 2` overlaps one trial's compile with the previous trial's timing and a gap therefore
measures "time until the next TRIAL_DONE landed" -- which produced medians of 0.0 s and a ratio of
192988x the first time this project tried it.

Run on box4:  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/s7_divergence.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")


def _latest(arm: str) -> Path | None:
    ds = sorted(BASE.glob(f"{arm}/run-l3-43-*"))
    return ds[-1] if ds else None


def _read(run: Path) -> list[dict]:
    out = []
    with (run / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _pl(e: dict) -> dict:
    return e.get("payload") or e


def _num(x) -> float | None:
    return float(x) if isinstance(x, (int, float)) else None


def arm(name: str) -> dict | None:
    run = _latest(name)
    if run is None:
        return None
    ev = _read(run)
    span = ev[-1]["ts"] - ev[0]["ts"] if len(ev) > 1 else 0.0

    walls: list[float] = []
    compiles: list[float] = []
    rows: list[tuple[float, str, dict]] = []
    kinds: dict[str, int] = {}
    agent_s = 0.0
    open_calls: dict[str, float] = {}
    agent_each: list[tuple[float, str]] = []

    for e in ev:
        t = e.get("type")
        p = _pl(e)
        if t == "TRIAL_DONE":
            tr = p.get("trial") or p
            fk = tr.get("failure_kind") or "ok"
            kinds[fk] = kinds.get(fk, 0) + 1
            w = _num(tr.get("job_wall_s"))
            prof = tr.get("profile") or {}
            c = _num(prof.get("compile_s")) if isinstance(prof, dict) else None
            if w is not None:
                walls.append(w)
                rows.append((w, fk, (tr.get("params") or {}).get("values") or {}))
            if c is not None:
                compiles.append(c)
        elif t == "AGENT_CALL_STARTED":
            open_calls[str(p.get("call_id") or e.get("seq"))] = e["ts"]
        elif t in ("AGENT_CALL_FINISHED", "AGENT_CALL_FAILED"):
            k = str(p.get("call_id") or "")
            st = open_calls.pop(k, None)
            if st is None and open_calls:
                st = open_calls.pop(min(open_calls, key=lambda x: open_calls[x]))
            if st is not None:
                agent_s += e["ts"] - st
                agent_each.append((e["ts"] - st, str(p.get("module") or p.get("agent") or "?")))

    return {
        "run": run.name,
        "span_h": span / 3600.0,
        "n": len(walls),
        "kinds": kinds,
        "wall_median": statistics.median(walls) if walls else 0.0,
        "wall_sum_h": sum(walls) / 3600.0,
        "wall_max": max(walls) if walls else 0.0,
        "compile_median": statistics.median(compiles) if compiles else 0.0,
        "agent_h": agent_s / 3600.0,
        "agent_frac": agent_s / span if span else 0.0,
        "agent_each": sorted(agent_each, reverse=True)[:4],
        "worst": sorted(rows, key=lambda r: r[0], reverse=True)[:3],
        "steps": sum(1 for e in ev if e.get("type") == "SLOPE_GUIDE_STEP"),
        # In-flight trials are the missing time: a trial that has not landed contributes to the wall
        # clock and to no count. Without this the sums do not reconcile and the gap looks unexplained.
        "unaccounted_h": (span - agent_s - sum(walls)) / 3600.0,
    }


def main() -> int:
    a, b = arm("s7-control"), arm("s7-treatment")
    if a is None or b is None:
        print("missing arm(s)")
        return 2

    for lbl, r in (("CONTROL", a), ("TREATMENT", b)):
        print("%-10s %s  span %.2f h" % (lbl, r["run"], r["span_h"]))
        print("%-10s trials %d   S7 steps %d   kinds %s" % ("", r["n"], r["steps"], r["kinds"]))
        print("%-10s job_wall_s: median %.1f  max %.0f  sum %.2f h" % (
            "", r["wall_median"], r["wall_max"], r["wall_sum_h"]))
        print("%-10s compile_s median %.1f   agent %.2f h (%.0f%% of span)   unaccounted %.2f h" % (
            "", r["compile_median"], r["agent_h"], 100 * r["agent_frac"], r["unaccounted_h"]))
        for s, mod in r["agent_each"][:3]:
            print("%-10s   agent call %6.0f s  %s" % ("", s, mod))
        for w, fk, vals in r["worst"]:
            keep = {k: vals.get(k) for k in ("BLOCK_M", "BLOCK_N", "NUM_WARPS", "NUM_STAGES",
                                             "COMPUTE_DTYPE", "DOT_MODE") if k in vals}
            print("%-10s   slowest %6.0f s  %-26s %s" % ("", w, fk, keep))
        print()

    print("=" * 78)
    if a["steps"] or b["steps"]:
        print("  * S7 HAS FIRED -- the mechanism is no longer excluded as a cause; re-read below")
    else:
        print("  * S7 steps 0/0: the mechanism has not fired, so it cannot be causing the gap")

    ma, mb = a["wall_median"], b["wall_median"]
    if ma > 0 and mb > 0:
        body = max(ma, mb) / min(ma, mb)
        verdict = "TAIL (does not compound)" if body <= 1.15 else "SHIFTED BODY (compounds)"
        print("  * per-trial median %.1f vs %.1f s => %.2fx  %s" % (ma, mb, body, verdict))
    print("  * agent share %.0f%% vs %.0f%% of the wall clock -- agent latency is not GPU search and "
          "is the other place the gap can live" % (100 * a["agent_frac"], 100 * b["agent_frac"]))
    print("  * arms tune DIFFERENT candidates, so config draws are unpaired: a tail is expected "
          "variance, not a confound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
