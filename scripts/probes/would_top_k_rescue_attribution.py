"""Would probe_top_k: 3 actually rescue attribution? Check the 2nd/3rd fastest points offline.

WHY THIS IS NOT RHETORICAL. The step-4 all-on arm's whole design rests on one claim: raising the
origin count from 1 to 3 turns unattributable walls into attributable ones. That claim is testable
from a finished arm's own record, with no GPU time -- and if it is false, the all-on arm will report
zero attributed walls exactly like the K=1 arms did, and the pair will measure nothing.

THE MECHANISM, and why K=3 might change nothing. Attribution reverts ONE knob at a time starting
from an origin, and asks whether the wall's refused value then fits under the limit. It fails when
the origin is too far from the refused point. Every origin K picks is a config that RAN, so all of
them are under the limit -- the question is whether the 2nd and 3rd fastest are CLOSER to the refused
point (in knobs that differ) than the 1st is. If they sit in the same corner of the space, K=3 buys
nothing but probe time.

WHAT IS MEASURED, per wall:
  * how many knobs separate the refused config from each of the 3 fastest measured configs
  * whether any of the three is close enough for a single-knob revert to be possible at all (d == 1)
  * the spread of d across the three, because a spread of 0 means the origins are interchangeable

WHAT THIS CANNOT SAY. It does not run the screen, so it cannot tell you the refused config would FIT
after a revert -- only whether a single-knob revert is geometrically reachable. A d == 1 pair can
still fail the screen. So a positive reading here is necessary, not sufficient, and the probe says so
rather than claiming K=3 will work.

POSITIVE CONTROL. `--selftest` builds a record where the 2nd fastest is one knob from the refusal
while the 1st is five away -- K=3 must be reported as rescuing it -- and one where all three origins
sit at the same distance, where K=3 must be reported as buying nothing.
"""
from __future__ import annotations

import json
import os
import sys


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


def _params(d: dict) -> dict:
    p = d.get("params") or {}
    v = p.get("values")
    if isinstance(v, dict):
        return dict(v)
    return dict(p) if isinstance(p, dict) else {}


def _ms(trial: dict) -> float | None:
    """median else mean -- the keys carry NO `_ms` suffix."""
    lat = trial.get("latency_ms") or {}
    for k in ("median", "mean"):
        v = lat.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def collect(events: list[dict]) -> dict:
    """space_id -> {"fast": [(ms, params) sorted], "refused": [params, ...]}"""
    out: dict[str, dict] = {}
    for e in events:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "TRIAL_DONE":
            tr = p.get("trial") or {}
            sid = str(tr.get("space_id") or "?")
            d = out.setdefault(sid, {"fast": [], "refused": []})
            if tr.get("status") == "complete":
                m = _ms(tr)
                pr = _params(tr)
                if m is not None and pr:
                    d["fast"].append((m, pr))
            elif str(tr.get("failure_kind") or "") == "infeasible_shared_memory":
                pr = _params(tr)
                if pr:
                    d["refused"].append(pr)
        elif t == "CONFIG_SCREENED_INFEASIBLE":
            # The compile-only screen refuses configs too, and its payload is the other source of
            # refused points. `candidate_id` is its grouping key, not `space_id`.
            sid = str(p.get("candidate_id") or "?")
            pr = _params(p)
            if pr:
                out.setdefault(sid, {"fast": [], "refused": []})["refused"].append(pr)
    for d in out.values():
        d["fast"].sort(key=lambda kv: kv[0])
    return out


def distance(a: dict, b: dict) -> int | None:
    """How many knobs differ. None when the two do not describe the same knob set."""
    if set(a) != set(b) or not a:
        return None
    return sum(1 for k in a if a[k] != b[k])


def analyse(spaces: dict, k: int = 3) -> dict:
    res = {"walls": 0, "reachable_at_1": 0, "reachable_within_k": 0,
           "rescued_by_k": 0, "no_spread": 0, "d_by_rank": {}, "rows": []}
    for sid, d in sorted(spaces.items()):
        fast, refused = d["fast"], d["refused"]
        if not fast or not refused:
            continue
        origins = fast[:k]
        for rp in refused:
            ds = []
            for rank, (_ms_, op) in enumerate(origins):
                dd = distance(rp, op)
                ds.append(dd)
                if dd is not None:
                    res["d_by_rank"].setdefault(rank, []).append(dd)
            known = [x for x in ds if x is not None]
            if not known:
                continue
            res["walls"] += 1
            first = ds[0] if ds and ds[0] is not None else None
            at1 = first == 1
            within = any(x == 1 for x in known)
            if at1:
                res["reachable_at_1"] += 1
            if within:
                res["reachable_within_k"] += 1
            if within and not at1:
                res["rescued_by_k"] += 1
            if len(set(known)) == 1:
                res["no_spread"] += 1
            res["rows"].append((sid, ds))
    return res


def report(res: dict, k: int) -> None:
    print("=" * 78)
    if not res["walls"]:
        print("NO refused config shares a knob set with any measured config, so the question is")
        print("UNANSWERED here rather than answered negatively.")
        return
    n = res["walls"]
    print("CAN A SINGLE-KNOB REVERT EVEN REACH THE REFUSED POINT?  %d refused config(s)" % n)
    print("  from the FASTEST point only (this is K=1, today's default):")
    print("      one knob away: %d / %d (%.0f%%)"
          % (res["reachable_at_1"], n, 100.0 * res["reachable_at_1"] / n))
    print("  from any of the %d fastest (this is what K=%d buys):" % (k, k))
    print("      one knob away: %d / %d (%.0f%%)"
          % (res["reachable_within_k"], n, 100.0 * res["reachable_within_k"] / n))
    print("      NEWLY reachable, i.e. rescued by K>1: %d" % res["rescued_by_k"])
    print("  refusals where all %d origins sit at the SAME distance: %d / %d" % (k, res["no_spread"], n))
    for rank in sorted(res["d_by_rank"]):
        ds = sorted(res["d_by_rank"][rank])
        print("      origin rank %d: d min %d  median %d  max %d"
              % (rank + 1, ds[0], ds[len(ds) // 2], ds[-1]))
    print()
    if res["rescued_by_k"]:
        print("=> K=%d WOULD CHANGE SOMETHING: %d refusal(s) become geometrically reachable that were"
              % (k, res["rescued_by_k"]))
        print("   not from the fastest point alone.")
    elif res["reachable_within_k"]:
        print("=> K=%d CHANGES NOTHING HERE: every reachable refusal was already reachable from the"
              % k)
        print("   fastest point. The extra origins cost probe time and buy no new attribution.")
    else:
        print("=> NO refusal is one knob from ANY of the %d fastest points. Raising K cannot help;" % k)
        print("   the gap is the DISTANCE, and only walking back from the refused point closes it.")
    print()
    print("NECESSARY, NOT SUFFICIENT. This measures geometric reachability only -- it does not run")
    print("the screen, so a d==1 pair can still fail to fit under the limit. Read a positive result")
    print("as 'K=%d is worth setting', never as 'attribution will now succeed'." % k)


def _selftest() -> int:
    def _t(space, params, ms):
        return {"type": "TRIAL_DONE", "payload": {"trial": {
            "space_id": space, "status": "complete", "params": {"values": params},
            "latency_ms": {"median": ms}}}}

    def _ref(space, params):
        return {"type": "TRIAL_DONE", "payload": {"trial": {
            "space_id": space, "status": "fail", "failure_kind": "infeasible_shared_memory",
            "params": {"values": params}}}}

    ok = True
    # The 2nd fastest is ONE knob from the refusal; the fastest is two away. K=3 must rescue it.
    evs = [_t("s", {"A": 1, "B": 1}, 1.0),       # fastest, d=2 from refusal
           _t("s", {"A": 9, "B": 1}, 2.0),       # 2nd,     d=1 from refusal
           _ref("s", {"A": 9, "B": 9})]
    r = analyse(collect(evs), k=3)
    if r["walls"] != 1:
        print("FAIL: the refusal was not counted (%s)" % r)
        ok = False
    elif r["reachable_at_1"] != 0:
        print("FAIL: the fastest point is 2 knobs away, must not count as reachable (%s)" % r)
        ok = False
    elif r["reachable_within_k"] != 1 or r["rescued_by_k"] != 1:
        print("FAIL: K=3 did not rescue a refusal one knob from the 2nd fastest (%s)" % r)
        ok = False
    # All origins at the same distance: K must be reported as buying nothing.
    evs = [_t("s", {"A": 1, "B": 1}, 1.0), _t("s", {"A": 1, "B": 2}, 2.0),
           _t("s", {"A": 1, "B": 3}, 3.0), _ref("s", {"A": 9, "B": 9})]
    r = analyse(collect(evs), k=3)
    if r["rescued_by_k"]:
        print("FAIL: K was credited with a rescue when no origin is one knob away (%s)" % r)
        ok = False
    elif r["reachable_within_k"]:
        print("FAIL: a 2-knob gap was called reachable (%s)" % r)
        ok = False
    # A screen refusal on the candidate key must also be collected.
    evs = [_t("c", {"A": 1}, 1.0),
           {"type": "CONFIG_SCREENED_INFEASIBLE", "payload": {
               "candidate_id": "c", "kernel": "_gemm", "limit": 101376, "max_shared": 131072,
               "params": {"A": 9}}}]
    r = analyse(collect(evs), k=3)
    if r["walls"] != 1 or r["reachable_at_1"] != 1:
        print("FAIL: a compile-screen refusal one knob away was missed (%s)" % r)
        ok = False
    # Mismatched knob sets must be skipped, not counted as distance 0.
    if distance({"A": 1}, {"A": 1, "B": 2}) is not None:
        print("FAIL: differing knob sets returned a distance")
        ok = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    k = 3
    for a in sys.argv[1:]:
        if a.startswith("--k="):
            k = int(a.split("=", 1)[1])
    paths = [a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl")
             for a in sys.argv[1:] if not a.startswith("-")]
    merged: dict[str, dict] = {}
    for path in paths:
        if not os.path.exists(path):
            print("missing: %s" % path)
            continue
        for sid, d in collect(_read(path)).items():
            key = "%s::%s" % (path.split(os.sep)[-3], sid)
            merged[key] = d
    if not merged:
        print("no readable record")
        raise SystemExit(2)
    report(analyse(merged, k=k), k)
