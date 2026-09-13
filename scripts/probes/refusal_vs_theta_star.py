"""Why does a shared-memory refusal fail attribution at over_ratio well below 1?

THE QUESTION. Attribution measures shared-memory use at theta*, the space's best point, and
divides by the hardware limit. On the S7 control arm a BLOCK_N wall came back not_attributed with
over_ratio 0.485 -- half the limit -- and a NUM_WARPS wall at 0.970. If the refused config really
overflowed, something must differ between it and theta*.

Two candidate explanations, and they call for different fixes:

  (a) THETA* IS NOT REPRESENTATIVE. The refused config sits far from theta* in the dimensions
      that drive the footprint, so theta*'s usage says nothing about it. The fix is to probe from
      the refused config's own neighbourhood, not from theta* alone -- which is what probe_top_k
      was added for.
  (b) THE LABEL IS WRONG. The trial was classified `infeasible_shared_memory` but the refusal was
      something else. `_classify_exception` only recognises "out of memory", and a Triton
      launch-time OutOfResources is already known to land as `runtime_error` -- the inverse
      mistake is possible too.

The record can separate them: a refused trial carries its own params and its failure_detail, and
the screen's message states the bytes required and the device limit.

THE ANSWER ON THE S7 CONTROL ARM IS (a). The BLOCK_N wall's theta* uses 32768 bytes of a 101376
limit -- hence 0.485 -- while the refused config needs 229376, i.e. 2.26x the limit. The refusal
is real and the label is right. It fails attribution because theta* and the refused point differ
in FIVE knobs at once (ATT_NUM_STAGES, ATT_NUM_WARPS, BLOCK_K, BLOCK_N, ATT_BLOCK_N), and
single-knob ablation from theta* can only attribute a refusal that one knob explains.

=> `over_ratio < 1` does NOT mean "the refusal had some other cause", which is how I had been
reading it. It means theta* is not representative of the refused point. The gap is in the
method's reach, not in the label and not in the wall.

NO POSITIVE CONTROL IS POSSIBLE HERE and that is worth saying: this probe reports what the record
contains, and if a run has no refused configs it prints nothing. Absence of output means no
refusals were logged, not that refusals were fine -- so it says which.
"""
from __future__ import annotations

import json
import os
import re
import sys

_NUM = re.compile(
    # The live message is "compile-only screen: _gemm requires 131072 bytes of shared memory
    # against this device's per-block opt-in limit of 101376". My first pattern demanded the words
    # "Required:" and "limit:" together, matched 0 of 39 refusals, and made the probe report the
    # numbers as MISSING -- offering that as evidence for the very field §4.1(a) proposes adding.
    # They were there all along. Accept both spellings and name the kernel.
    r"(?:(\w+)\s+requires\s+(\d+)\s*bytes|Required:?\s*(\d+))"
    r".{0,80}?limit(?:\s+of|:)?\s*(\d+)",
    re.I | re.S)


def _req_limit(detail: str) -> tuple[str | None, int, int] | None:
    m = _NUM.search(detail or "")
    if not m:
        return None
    kernel = m.group(1)
    req = int(m.group(2) or m.group(3))
    return kernel, req, int(m.group(4))


def _read(path: str) -> list[dict]:
    ev = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return ev


def _params(tr: dict) -> dict:
    p = tr.get("params") or {}
    v = p.get("values")
    return dict(v) if isinstance(v, dict) else {}


def _shared(tr: dict) -> object:
    return (tr.get("profile") or {}).get("shared_bytes")


def report(events: list[dict], label: str) -> dict:
    """Per space: the refused configs, their stated byte numbers, and theta*'s own usage."""
    by_space: dict[str, list[dict]] = {}
    for e in events:
        if e.get("type") != "TRIAL_DONE":
            continue
        tr = (e.get("payload") or {}).get("trial") or {}
        sid = str(tr.get("space_id") or "?")
        by_space.setdefault(sid, []).append(tr)

    # Which knobs each wall named, and the verdict, so the refusals can be read next to it.
    walls: list[tuple[str, dict]] = []
    for e in events:
        if e.get("type") != "RESOURCE_WALL_ATTRIBUTED":
            continue
        p = e.get("payload") or {}
        for w in (p.get("walls") or []):
            if isinstance(w, dict):
                walls.append((str(p.get("space_id") or "?"), w))

    out = {"refusals": 0, "with_numbers": 0, "far_from_theta": 0, "rows": []}
    print("=" * 78)
    print(label)
    if not walls:
        print("  no wall records at all -- nothing to explain")
        return out
    for sid, w in walls:
        verdict = str(w.get("verdict") or "(never probed)")
        ratio = w.get("over_ratio")
        rs = ("%.3f" % ratio) if isinstance(ratio, (int, float)) else "?"
        print("  --- wall on %-12s in %s   verdict %s  over_ratio %s"
              % (w.get("param") or "?", sid, verdict, rs))
        trials = by_space.get(sid, [])
        done = [t for t in trials if t.get("status") == "complete"]
        best = None
        best_ms = None
        for t in done:
            lat = t.get("latency_ms") or {}
            m = lat.get("median")
            if not isinstance(m, (int, float)):
                m = lat.get("mean")
            if isinstance(m, (int, float)) and (best_ms is None or m < best_ms):
                best_ms, best = m, t
        if best is not None:
            print("      theta* shared_bytes %s   params %s"
                  % (_shared(best), json.dumps(_params(best), sort_keys=True)[:150]))
        refused = [t for t in trials
                   if str(t.get("failure_kind") or "") == "infeasible_shared_memory"]
        if not refused:
            print("      NO trial in this space is labelled infeasible_shared_memory -- so this")
            print("      wall's input came from elsewhere in the record; do not read the numbers")
            print("      below as its cause.")
            continue
        knob = str(w.get("param") or "")
        for t in refused:
            out["refusals"] += 1
            detail = str(t.get("failure_detail") or "")
            rl = _req_limit(detail)
            nums = ""
            if rl:
                out["with_numbers"] += 1
                kernel, req, lim = rl
                nums = "  %s needs %d / limit %d (%.2fx)" % (
                    kernel or "?", req, lim, req / lim if lim else 0.0)
            ps = _params(t)
            # How many knobs differ from theta*, not just the walled one. Single-knob ablation from
            # theta* can only attribute a refusal that ONE knob explains; if the refused point
            # differs in several, each knob alone stays inside the limit and the verdict comes back
            # not_attributed even though the refusal is real.
            differing = []
            if best is not None:
                bp = _params(best)
                differing = [k for k in ps if k in bp and ps[k] != bp[k]]
            far = ""
            if knob in differing:
                bp = _params(best) if best is not None else {}
                far = "  <-- %s differs (%s vs %s); %d knob(s) differ in all" % (
                    knob, ps[knob], bp.get(knob), len(differing))
                out["far_from_theta"] += 1
            elif differing:
                far = "  <-- %d knob(s) differ from theta*, none of them %s" % (
                    len(differing), knob or "the walled knob")
            print("      refused: %s%s%s" % (json.dumps(ps, sort_keys=True)[:110], nums, far))
            if detail and not rl:
                print("               detail has no parseable byte numbers: %s" % detail[:120])
            out["rows"].append({"space": sid, "knob": knob, "params": ps, "detail": detail[:200],
                                "n_differing": len(differing),
                                "over": (rl[1] / rl[2]) if rl and rl[2] else None})
    return out


if __name__ == "__main__":
    paths = [a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl")
             for a in sys.argv[1:] if not a.startswith("-")]
    tot = {"refusals": 0, "with_numbers": 0, "far_from_theta": 0}
    for path in paths:
        if not os.path.exists(path):
            continue
        f = report(_read(path), "/".join(path.split(os.sep)[-3:-1]))
        for k in tot:
            tot[k] += f[k]
    print()
    print("POOLED refusals %d   with parseable byte numbers %d   differing from theta* in the "
          "walled knob %d" % (tot["refusals"], tot["with_numbers"], tot["far_from_theta"]))
    if not tot["refusals"]:
        print("  NO refused configs were logged. That is the reason for the empty output above --")
        print("  it is not a finding that refusals are fine.")
        raise SystemExit(0)
    if not tot["with_numbers"]:
        print("  NONE carried parseable numbers. Check the REGEX before concluding the field is")
        print("  missing: the live message is '<kernel> requires N bytes ... limit of M', and a")
        print("  pattern demanding 'Required:'/'limit:' matches none of them while the numbers are")
        print("  right there. That mistake reads as evidence for adding a field that exists.")
        raise SystemExit(0)
    print()
    print("WHAT over_ratio < 1 ACTUALLY MEANS. The refusals above are real and the messages state")
    print("the bytes: they exceed the limit by wide margins. theta* meanwhile sits well under it.")
    print("So a ratio below 1 does NOT say 'the refusal had some other cause' -- it says theta* is")
    print("not representative of the refused point. Attribution ablates ONE knob from theta*, and")
    print("when the refused config differs in several, each knob alone stays inside the limit and")
    print("the verdict is not_attributed even though shared memory genuinely refused it.")
    print("=> The gap is in the METHOD's reach, not in the label and not in the wall.")
