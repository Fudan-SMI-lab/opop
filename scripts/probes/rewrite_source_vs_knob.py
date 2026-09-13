"""Did the SOURCE change move the resource dims, or did the KNOBS?

WHY THIS EXISTS. `rewrite_vs_promise.py` prints each side's resource profile at its OWN best
point. That is the right thing for "what did we end up with", but it is the WRONG thing for
"was the hypothesis's stated mechanism real": parent and child sit at different tile sizes, and
shared_bytes / n_regs / occupancy move with the tile whatever the source says. Reading a
direction off that comparison attributes a knob effect to a source edit.

The only reading that separates them is a point where BOTH sides were measured at the SAME
params. Then the knobs are held and the difference is the source. This probe looks for such a
point and REFUSES to substitute the own-best comparison when there is none -- a rewrite whose
child re-parameterized to a different knob set has no fixed-knob reading at all, and saying so
is the answer, not a gap to paper over.

POSITIVE CONTROL. `--selftest` builds two arms: one where a matched point exists (must print a
fixed-knob delta) and one where the child's knob names differ (must print NO COMPARABLE POINT
and no delta). Without the second, "found no matched point" and "found one and it agreed" would
print the same reassuring nothing.

Also prints measured vs TOTAL trials per space. `rewrite_vs_promise.py` says "closed at 35
trials" counting completes only, while check_arm_search_parity.py's per-space budget counts
every trial. Two denominators, one word: 35-of-40 reads as a budget shortfall that never
happened.
"""
from __future__ import annotations

import json
import os
import sys

_DIMS = ("n_regs", "n_spills", "shared_bytes", "num_warps", "occupancy", "occ_limiter")


def _robust_ms(lat: object) -> float | None:
    """LatencyStats.robust_ms is a @property and never serialized: reproduce median-else-mean.

    THE KEYS ARE `median` AND `mean`, not `median_ms`/`mean_ms`. Writing the `_ms` suffix from
    memory makes this return None for EVERY trial, which does not look like a bug: the caller
    then reports "no closed sibling to compare against", a sentence about the run rather than
    about the reader. That is exactly how it failed the first time it was pointed at the live
    pair, while its own selftest passed on fixtures that used the invented names.
    """
    if not isinstance(lat, dict):
        return None
    for k in ("median", "mean"):
        v = lat.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _dims_of(tr: dict) -> dict:
    prof = tr.get("profile") or {}
    out = {}
    for k in _DIMS:
        if k in prof:
            out[k] = prof[k]
    occ = prof.get("occupancy")
    if isinstance(occ, dict):  # occupancy is NESTED; a flat read fakes "unmeasured"
        if "occupancy" in occ:
            out["occupancy"] = occ["occupancy"]
        if "limiter" in occ:
            out["occ_limiter"] = occ["limiter"]
    return out


def _params_of(tr: dict) -> dict:
    p = tr.get("params") or {}
    v = p.get("values")
    return dict(v) if isinstance(v, dict) else {}


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


def report(events: list[dict], label: str) -> dict:
    """Returns the machine-readable findings so the selftest can assert on them."""
    fam_of: dict[str, str] = {}
    rewrites: list[dict] = []
    all_trials: dict[str, list[dict]] = {}
    done_trials: dict[str, list[dict]] = {}
    closed: set[str] = set()
    for e in events:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "CANDIDATE_REGISTERED":
            c = p.get("candidate") or p
            if c.get("candidate_id"):
                fam_of[str(c["candidate_id"])] = str(c.get("family_id") or "?")
        elif t == "REWRITE_PRODUCED":
            rewrites.append(e)
        elif t == "TRIAL_DONE":
            tr = p.get("trial") or {}
            cid = str(tr.get("candidate_id"))
            all_trials.setdefault(cid, []).append(tr)
            if tr.get("status") == "complete":
                done_trials.setdefault(cid, []).append(tr)
        elif t == "TUNING_DONE" and p.get("candidate_id"):
            closed.add(str(p["candidate_id"]))

    def _best(cid: str) -> dict | None:
        rows = [(m, tr) for tr in done_trials.get(cid, [])
                if (m := _robust_ms(tr.get("latency_ms"))) is not None]
        return min(rows, key=lambda r: r[0])[1] if rows else None

    print("=" * 78)
    print(label)
    findings = {"matched": 0, "no_common_knobs": 0, "no_matched_point": 0, "skipped_open": 0}
    for e in rewrites:
        p = e.get("payload") or {}
        child = str(p.get("candidate_id") or "?")
        fid = str(p.get("family_id") or fam_of.get(child, "?"))
        print("  --- %s (%s) in %s" % (child, p.get("hypothesis_id") or "?", fid))
        if child not in closed:
            print("      space still OPEN -- no verdict (winners arrive late by construction)")
            findings["skipped_open"] += 1
            continue
        n_done, n_all = len(done_trials.get(child, [])), len(all_trials.get(child, []))
        # Say WHY the two numbers differ only when they do -- an unconditional "the rest failed"
        # clause on an all-complete space is a claim about trials that do not exist.
        print("      %d measured of %d trials spent%s"
              % (n_done, n_all,
                 " (%d failed; the BUDGET was not short)" % (n_all - n_done)
                 if n_all > n_done else ""))
        sibs = [c for c, f in fam_of.items() if f == fid and c != child and c in closed]
        pbest, parent = None, None
        for c in sibs:
            tr = _best(c)
            m = _robust_ms(tr.get("latency_ms")) if tr else None
            if m is not None and (pbest is None or m < pbest):
                pbest, parent = m, c
        cb_tr = _best(child)
        if parent is None or cb_tr is None:
            print("      no closed sibling to compare against")
            continue
        p_tr = _best(parent)
        pk, ck = set(_params_of(p_tr)), set(_params_of(cb_tr))
        common = pk & ck
        if not common or common != pk or common != ck:
            print("      knob sets DIFFER (parent-only %s, child-only %s)"
                  % (sorted(pk - ck) or "-", sorted(ck - pk) or "-"))
        if not common:
            print("      NO COMPARABLE POINT: the child re-parameterized to a disjoint knob set,")
            print("      so no fixed-knob reading exists. The own-best comparison in")
            print("      rewrite_vs_promise.py mixes source and knob effects and cannot be")
            print("      substituted here.")
            findings["no_common_knobs"] += 1
            continue
        target = {k: _params_of(p_tr)[k] for k in common}
        hits = [tr for tr in done_trials.get(child, [])
                if {k: _params_of(tr).get(k) for k in common} == target]
        if not hits:
            print("      NO MATCHED POINT: the child never measured the parent's best params")
            print("      %s" % json.dumps(target, sort_keys=True)[:200])
            print("      => the direction of any resource change is NOT separable from the knob")
            print("      change. Do not read a mechanism off the own-best profiles.")
            findings["no_matched_point"] += 1
            continue
        m = min(hits, key=lambda tr: _robust_ms(tr.get("latency_ms")) or 9e9)
        pd, cd = _dims_of(p_tr), _dims_of(m)
        print("      FIXED-KNOB reading at the parent's best params (%d common knob(s)):"
              % len(common))
        print("        latency       parent %-12s child %s"
              % ("%.4f ms" % (_robust_ms(p_tr.get("latency_ms")) or 0),
                 "%.4f ms" % (_robust_ms(m.get("latency_ms")) or 0)))
        for k in sorted(set(pd) | set(cd)):
            print("        %-13s parent %-12s child %s"
                  % (k, pd.get(k, "-"), cd.get(k, "-")))
        print("      This holds the knobs, so the difference IS the source change.")
        findings["matched"] += 1
    return findings


def _selftest() -> int:
    def _tr(cid, params, ms, prof, status="complete"):
        # `median`, NOT `median_ms` -- copied field-for-field from a live TRIAL_DONE record.
        # A fixture that invents the key name lets the reader's own typo pass (see
        # a-fixture-invented-to-match-the-reader-proves-nothing).
        return {"type": "TRIAL_DONE", "payload": {"trial": {
            "candidate_id": cid, "trial_id": cid + json.dumps(params, sort_keys=True),
            "status": status, "params": {"values": params},
            "latency_ms": {"median": ms, "mean": ms, "min": ms, "max": ms,
                           "std": 0.0, "n_samples": 100, "samples": []},
            "profile": prof}}}

    prof_p = {"n_regs": 200, "shared_bytes": 16384, "occupancy": {"occupancy": 0.25,
                                                                 "limiter": "registers"}}
    prof_c = {"n_regs": 160, "shared_bytes": 8192, "occupancy": {"occupancy": 0.42,
                                                                 "limiter": "registers"}}
    base = [
        {"type": "CANDIDATE_REGISTERED", "payload": {"candidate":
            {"candidate_id": "par", "family_id": "f1"}}},
        {"type": "CANDIDATE_REGISTERED", "payload": {"candidate":
            {"candidate_id": "kid", "family_id": "f1"}}},
        {"type": "REWRITE_PRODUCED", "payload":
            {"candidate_id": "kid", "family_id": "f1", "hypothesis_id": "H1"}},
        _tr("par", {"BLOCK_M": 64}, 3.0, prof_p),
        {"type": "TUNING_DONE", "payload": {"candidate_id": "par"}},
        {"type": "TUNING_DONE", "payload": {"candidate_id": "kid"}},
    ]
    ok = True
    # 1. matched point present -> must produce a fixed-knob delta
    f = report(base + [_tr("kid", {"BLOCK_M": 64}, 2.7, prof_c),
                       _tr("kid", {"BLOCK_M": 32}, 2.9, prof_c)], "SELFTEST matched")
    if f["matched"] != 1:
        print("FAIL: a matched point was not read as one (%s)" % f)
        ok = False
    # 2. disjoint knob names -> must refuse, and must NOT count as matched
    f = report(base + [_tr("kid", {"TILE": 64}, 2.7, prof_c)], "SELFTEST disjoint knobs")
    if f["no_common_knobs"] != 1 or f["matched"]:
        print("FAIL: a disjoint knob set was not refused (%s)" % f)
        ok = False
    # 3. same knobs, child never measured the parent's value -> must refuse
    f = report(base + [_tr("kid", {"BLOCK_M": 32}, 2.7, prof_c)], "SELFTEST no matched value")
    if f["no_matched_point"] != 1 or f["matched"]:
        print("FAIL: an unmatched value was not refused (%s)" % f)
        ok = False
    # 4. failed trials must not shrink the reported budget
    f = report(base + [_tr("kid", {"BLOCK_M": 64}, 2.7, prof_c),
                       _tr("kid", {"BLOCK_M": 16}, 0.0, {}, status="fail")],
               "SELFTEST failed trial counted in spend")
    if f["matched"] != 1:
        print("FAIL: a failed trial broke the matched reading (%s)" % f)
        ok = False
    # 5. the median-else-mean fallback must actually fall back
    if _robust_ms({"mean": 2.5, "min": 2.4}) != 2.5:
        print("FAIL: mean-only record did not fall back")
        ok = False
    if _robust_ms({"median": 3.0, "mean": 9.0}) != 3.0:
        print("FAIL: median was not preferred over mean")
        ok = False
    # 6. the WRONG key names must read as absent. If someone later "helpfully" accepts both
    # spellings, this probe would go back to silently reading None off real records whose
    # schema had changed -- an accepted typo is a permanent tolerance for the empty answer.
    if _robust_ms({"median_ms": 3.0, "mean_ms": 3.0}) is not None:
        print("FAIL: the _ms spelling was accepted, so a schema change reads as a latency")
        ok = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    base = "/root/autodl-tmp/opop-workspace/opop-glm/runs-v3"
    run = "run-l3-43-20260913-202332"
    for arm in ("s7-treatment", "s7-control"):
        path = os.path.join(base, arm, run, "events.jsonl")
        if os.path.exists(path):
            report(_read(path), "%s  %s" % (arm, run))
