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


def report(events: list[dict], label: str, quiet: bool = False) -> dict:
    """Returns the machine-readable findings so the selftest can assert on them."""
    def _say(*a: object) -> None:
        if not quiet:
            print(*a)

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

    _say("=" * 78)
    _say(label)
    findings = {"matched": 0, "no_common_knobs": 0, "no_matched_point": 0, "skipped_open": 0,
                "no_sibling": 0, "identity": 0, "moved": 0, "moved_dims": {}, "cases": []}
    for e in rewrites:
        p = e.get("payload") or {}
        child = str(p.get("candidate_id") or "?")
        fid = str(p.get("family_id") or fam_of.get(child, "?"))
        _say("  --- %s (%s) in %s" % (child, p.get("hypothesis_id") or "?", fid))
        if child not in closed:
            _say("      space still OPEN -- no verdict (winners arrive late by construction)")
            findings["skipped_open"] += 1
            continue
        n_done, n_all = len(done_trials.get(child, [])), len(all_trials.get(child, []))
        # Say WHY the two numbers differ only when they do -- an unconditional "the rest failed"
        # clause on an all-complete space is a claim about trials that do not exist.
        _say("      %d measured of %d trials spent%s"
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
            _say("      no closed sibling to compare against")
            findings["no_sibling"] += 1
            continue
        p_tr = _best(parent)
        pk, ck = set(_params_of(p_tr)), set(_params_of(cb_tr))
        common = pk & ck
        if not common or common != pk or common != ck:
            _say("      knob sets DIFFER (parent-only %s, child-only %s)"
                 % (sorted(pk - ck) or "-", sorted(ck - pk) or "-"))
        if not common:
            _say("      NO COMPARABLE POINT: the child re-parameterized to a disjoint knob set,")
            _say("      so no fixed-knob reading exists. The own-best comparison in")
            _say("      rewrite_vs_promise.py mixes source and knob effects and cannot be")
            _say("      substituted here.")
            findings["no_common_knobs"] += 1
            continue
        target = {k: _params_of(p_tr)[k] for k in common}
        hits = [tr for tr in done_trials.get(child, [])
                if {k: _params_of(tr).get(k) for k in common} == target]
        if not hits:
            _say("      NO MATCHED POINT: the child never measured the parent's best params")
            _say("      %s" % json.dumps(target, sort_keys=True)[:200])
            _say("      => the direction of any resource change is NOT separable from the knob")
            _say("      change. Do not read a mechanism off the own-best profiles.")
            findings["no_matched_point"] += 1
            continue
        m = min(hits, key=lambda tr: _robust_ms(tr.get("latency_ms")) or 9e9)
        pd, cd = _dims_of(p_tr), _dims_of(m)
        pms, cms = _robust_ms(p_tr.get("latency_ms")) or 0.0, _robust_ms(m.get("latency_ms")) or 0.0
        _say("      FIXED-KNOB reading at the parent's best params (%d common knob(s)):"
             % len(common))
        _say("        latency       parent %-12s child %-12s (%+.2f%%)"
             % ("%.4f ms" % pms, "%.4f ms" % cms,
                100 * (pms - cms) / pms if pms else 0.0))
        # Which dims MOVED is the quantity §4.4 needs across runs. A dim present on only one
        # side counts as moved -- "the child stopped reporting it" is not "unchanged".
        moved = []
        for k in sorted(set(pd) | set(cd)):
            same = k in pd and k in cd and pd[k] == cd[k]
            if not same:
                moved.append(k)
            _say("        %-13s parent %-12s child %-12s %s"
                 % (k, pd.get(k, "-"), cd.get(k, "-"), "" if same else "<-- MOVED"))
        findings["matched"] += 1
        findings["cases"].append({"run": label, "child": child,
                                  "hypothesis_id": p.get("hypothesis_id"),
                                  "moved": moved, "n_dims": len(set(pd) | set(cd)),
                                  "latency_pct": (100 * (pms - cms) / pms) if pms else 0.0})
        if moved:
            findings["moved"] += 1
            for k in moved:
                findings["moved_dims"][k] = findings["moved_dims"].get(k, 0) + 1
            _say("      %d of %d dim(s) MOVED at fixed knobs => the source edit did something"
                 % (len(moved), len(set(pd) | set(cd))))
        else:
            findings["identity"] += 1
            _say("      IDENTITY on all %d dim(s) at fixed knobs => on the dimensions we"
                 % len(set(pd) | set(cd)))
            _say("      measure, this source edit was a NO-OP. Any improvement credited to it")
            _say("      came from the tuner moving to a different point in the new space.")
    return findings
    return findings


def _pool(paths: list[str]) -> None:
    """Cross-run tally. The DENOMINATOR is the decidable set, not the rewrite count."""
    tot = {"matched": 0, "no_common_knobs": 0, "no_matched_point": 0, "skipped_open": 0,
           "no_sibling": 0, "identity": 0, "moved": 0}
    dims: dict[str, int] = {}
    cases: list[dict] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        label = "/".join(path.split(os.sep)[-3:-1])
        f = report(_read(path), label)
        for k in tot:
            tot[k] += f.get(k, 0)
        for k, v in f["moved_dims"].items():
            dims[k] = dims.get(k, 0) + v
        cases.extend(f["cases"])

    print()
    print("=" * 78)
    print("POOLED OVER %d RUN(S)" % len(paths))
    undecidable = tot["no_common_knobs"] + tot["no_matched_point"] + tot["skipped_open"] \
        + tot["no_sibling"]
    print("  rewrites seen            %d" % (tot["matched"] + undecidable))
    print("  DECIDABLE (matched pt)   %d   <-- the only honest denominator" % tot["matched"])
    print("  undecidable              %d  (open %d, no sibling %d, disjoint knobs %d, "
          "value never measured %d)"
          % (undecidable, tot["skipped_open"], tot["no_sibling"],
             tot["no_common_knobs"], tot["no_matched_point"]))
    if not tot["matched"]:
        print("  NO RATE: nothing was decidable. This is not '0% no-ops'.")
        return
    print()
    print("  source edit MOVED >=1 measured dim   %d / %d  (%.0f%%)"
          % (tot["moved"], tot["matched"], 100.0 * tot["moved"] / tot["matched"]))
    print("  source edit was IDENTITY on all      %d / %d  (%.0f%%)"
          % (tot["identity"], tot["matched"], 100.0 * tot["identity"] / tot["matched"]))
    if dims:
        print("  which dims move, when any does:")
        for k, v in sorted(dims.items(), key=lambda kv: -kv[1]):
            print("      %-14s %d" % (k, v))
    print()
    print("  per case (latency delta is AT FIXED KNOBS, so it is the source edit's own effect;")
    print("  +-2-4%% is the re-eval noise floor, so treat anything inside it as no signal):")
    for c in sorted(cases, key=lambda c: -len(c["moved"])):
        print("      %-14s %-3s %+6.2f%%  %d/%d dims moved%s"
              % (c["child"][:14], c["hypothesis_id"] or "?", c["latency_pct"],
                 len(c["moved"]), c["n_dims"],
                 "  " + ",".join(c["moved"]) if c["moved"] else ""))
    print()
    print("READ IT WITH THESE LIMITS.")
    print("  * The dims are the ones we MEASURE (regs, spills, shared, warps, occupancy,")
    print("    limiter). An edit that changes instruction mix or memory access ORDER with no")
    print("    footprint change is IDENTITY here and is NOT thereby a no-op in general --")
    print("    the honest claim is 'a no-op on the dimensions C1 aligns'.")
    print("  * The matched point is the parent's best. A source edit could move a dim")
    print("    everywhere else and not there; that is the price of holding the knobs.")
    print("  * `no sibling`/`disjoint knobs` are not failures of the rewrite, they are")
    print("    silence. Rolling them into either rate would manufacture one.")


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
    # 7. identity vs moved must actually be distinguished, in BOTH directions. Without this a
    # classifier stuck on either answer looks like a finding: "every edit is a no-op" reads as
    # a result, and "none is" reads as reassurance.
    f = report(base + [_tr("kid", {"BLOCK_M": 64}, 2.7, prof_p)], "SELFTEST identity", quiet=True)
    if f["identity"] != 1 or f["moved"]:
        print("FAIL: an identical profile was not classified as identity (%s)" % f)
        ok = False
    f = report(base + [_tr("kid", {"BLOCK_M": 64}, 2.7, prof_c)], "SELFTEST moved", quiet=True)
    if f["moved"] != 1 or f["identity"]:
        print("FAIL: a changed profile was not classified as moved (%s)" % f)
        ok = False
    elif sorted(f["moved_dims"]) != ["n_regs", "occupancy", "shared_bytes"]:
        print("FAIL: the moved dims were misnamed (%s)" % f["moved_dims"])
        ok = False
    # 8. a dim the child stopped reporting counts as MOVED, not as unchanged -- otherwise a
    # profile that lost a field would read as evidence of stability.
    f = report(base + [_tr("kid", {"BLOCK_M": 64}, 2.7, {"n_regs": 200})],
               "SELFTEST dropped dim", quiet=True)
    if f["moved"] != 1 or "shared_bytes" not in f["moved_dims"]:
        print("FAIL: a dropped dim was read as unchanged (%s)" % f)
        ok = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        _pool([a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl") for a in args])
        raise SystemExit(0)
    base = "/root/autodl-tmp/opop-workspace/opop-glm/runs-v3"
    run = "run-l3-43-20260913-202332"
    for arm in ("s7-treatment", "s7-control"):
        path = os.path.join(base, arm, run, "events.jsonl")
        if os.path.exists(path):
            report(_read(path), "%s  %s" % (arm, run))
