"""Is the DECIDABLE set biased? Some closed rewrites never measured the parent's best params.

WHY THIS MATTERS. rewrite_source_vs_knob.py pools the decidable rewrites and reports a share of
them as source no-ops. That share is only about rewrites in general if the undecidable ones are
missing for reasons unrelated to what the edit did. Two ways it could be biased:

  (a) The parent's best point is REFUSED in the child's space -- e.g. the edit raised the
      shared-memory footprint so the parent's tile no longer fits. Those are exactly the
      rewrites that changed a resource dim the most, so dropping them would understate
      "moved" and inflate the no-op rate.
  (b) TPE simply never sampled that point in its trials. That is a sampler coincidence and
      carries no information about the edit.

(a) and (b) look identical in the pooled tally and have opposite implications, so this probe
separates them: for each undecidable rewrite it asks whether the parent's best params were
ATTEMPTED in the child's space and, if so, with what failure_kind.

Counts are deliberately not quoted in this docstring: they moved once already when the parent
bound was corrected (24% no-ops over 17 decidable became 11% over 28), and a number frozen in a
comment outlives the run it came from. Read them from the output.

POSITIVE CONTROL. --selftest builds a child that attempted the point and failed on shared
memory (must be reported as refused) and one that never attempted it (must be reported as
unsampled). Without both, "every one is a coincidence" and "the probe cannot see attempts"
print the same reassuring answer.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewrite_source_vs_knob import _params_of, _read, _robust_ms  # noqa: E402


def audit(events: list[dict], label: str, quiet: bool = False) -> dict:
    def _say(*a: object) -> None:
        if not quiet:
            print(*a)

    fam_of: dict[str, str] = {}
    rewrites: list[dict] = []
    all_trials: dict[str, list[dict]] = {}
    done: dict[str, list[dict]] = {}
    closed: set[str] = set()
    closed_at: dict[str, int] = {}
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
                done.setdefault(cid, []).append(tr)
        elif t == "TUNING_DONE" and p.get("candidate_id"):
            cid = str(p["candidate_id"])
            closed.add(cid)
            # Same seq bound as rewrite_source_vs_knob.py: the parent must have closed BEFORE the
            # rewrite was produced. Without it this probe's denominator diverges from the pooled
            # tally's (it read 16 undecidable against the tally's 7) because a later sibling
            # supplies a different "parent best" and therefore a different point to match.
            closed_at.setdefault(cid, int(e.get("seq") or 0))

    def _best(cid: str) -> dict | None:
        rows = [(m, tr) for tr in done.get(cid, [])
                if (m := _robust_ms(tr.get("latency_ms"))) is not None]
        return min(rows, key=lambda r: r[0])[1] if rows else None

    out = {"refused": 0, "unsampled": 0, "kinds": {}, "rows": []}
    for e in rewrites:
        p = e.get("payload") or {}
        child = str(p.get("candidate_id") or "?")
        if child not in closed:
            continue
        fid = str(p.get("family_id") or fam_of.get(child, "?"))
        sibs = [c for c, f in fam_of.items()
                if f == fid and c != child and c in closed
                and closed_at.get(c, 1 << 62) < int(e.get("seq") or 0)]
        pbest, parent = None, None
        for c in sibs:
            tr = _best(c)
            m = _robust_ms(tr.get("latency_ms")) if tr else None
            if m is not None and (pbest is None or m < pbest):
                pbest, parent = m, c
        p_tr, cb_tr = _best(parent or ""), _best(child)
        if p_tr is None or cb_tr is None:
            continue
        common = set(_params_of(p_tr)) & set(_params_of(cb_tr))
        if not common:
            continue
        target = {k: _params_of(p_tr)[k] for k in common}
        # Was it ATTEMPTED? Search ALL trials, not just complete ones -- the whole point is to
        # find the failures. A completed hit means the rewrite was decidable and is not ours.
        att = [tr for tr in all_trials.get(child, [])
               if {k: _params_of(tr).get(k) for k in common} == target]
        if any(tr.get("status") == "complete" for tr in att):
            continue
        if att:
            kinds = sorted({str(tr.get("failure_kind") or "(none)") for tr in att})
            detail = ""
            for tr in att:
                d = str(tr.get("failure_detail") or "")
                if d:
                    detail = d[:110]
                    break
            out["refused"] += 1
            for k in kinds:
                out["kinds"][k] = out["kinds"].get(k, 0) + 1
            out["rows"].append({"run": label, "child": child, "why": "refused",
                                "kinds": kinds, "detail": detail})
            _say("  %-16s %-4s ATTEMPTED and FAILED: %s" % (child[:16],
                 p.get("hypothesis_id") or "?", ",".join(kinds)))
            if detail:
                _say("                        %s" % detail)
        else:
            out["unsampled"] += 1
            out["rows"].append({"run": label, "child": child, "why": "unsampled",
                                "kinds": [], "detail": ""})
            _say("  %-16s %-4s NEVER ATTEMPTED in %d trials (sampler coincidence)"
                 % (child[:16], p.get("hypothesis_id") or "?", len(all_trials.get(child, []))))
    return out


def _selftest() -> int:
    def _tr(cid, params, status, kind=None, detail=None, ms=3.0):
        tr = {"candidate_id": cid, "status": status, "params": {"values": params},
              "latency_ms": {"median": ms, "mean": ms}, "profile": {"n_regs": 200}}
        if kind:
            tr["failure_kind"] = kind
        if detail:
            tr["failure_detail"] = detail
        return {"type": "TRIAL_DONE", "payload": {"trial": tr}}

    base = [
        {"type": "CANDIDATE_REGISTERED", "payload": {"candidate":
            {"candidate_id": "par", "family_id": "f1"}}},
        {"type": "CANDIDATE_REGISTERED", "payload": {"candidate":
            {"candidate_id": "kid", "family_id": "f1"}}},
        {"seq": 5, "type": "REWRITE_PRODUCED", "payload":
            {"candidate_id": "kid", "family_id": "f1", "hypothesis_id": "H1"}},
        _tr("par", {"BLOCK_M": 64}, "complete", ms=3.0),
        {"seq": 4, "type": "TUNING_DONE", "payload": {"candidate_id": "par"}},
        {"seq": 9, "type": "TUNING_DONE", "payload": {"candidate_id": "kid"}},
    ]
    ok = True
    f = audit(base + [_tr("kid", {"BLOCK_M": 64}, "fail", "infeasible_shared_memory",
                          "Required 131072, limit 101376"),
                      _tr("kid", {"BLOCK_M": 32}, "complete", ms=2.5)],
              "st", quiet=True)
    if f["refused"] != 1 or f["unsampled"]:
        print("FAIL: a refused point was not reported as refused (%s)" % f)
        ok = False
    elif f["kinds"] != {"infeasible_shared_memory": 1}:
        print("FAIL: the failure kind was lost (%s)" % f["kinds"])
        ok = False
    f = audit(base + [_tr("kid", {"BLOCK_M": 32}, "complete", ms=2.5)], "st", quiet=True)
    if f["unsampled"] != 1 or f["refused"]:
        print("FAIL: an unsampled point was not reported as unsampled (%s)" % f)
        ok = False
    # A decidable rewrite must be OUT of this audit entirely -- otherwise the two probes would
    # double-count the same rewrite in incompatible ways.
    f = audit(base + [_tr("kid", {"BLOCK_M": 64}, "complete", ms=2.5)], "st", quiet=True)
    if f["refused"] or f["unsampled"]:
        print("FAIL: a decidable rewrite leaked into the undecidable audit (%s)" % f)
        ok = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    paths = [a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl")
             for a in sys.argv[1:] if not a.startswith("-")]
    tot = {"refused": 0, "unsampled": 0}
    kinds: dict[str, int] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        print("=" * 78)
        label = "/".join(path.split(os.sep)[-3:-1])
        print(label)
        f = audit(_read(path), label)
        tot["refused"] += f["refused"]
        tot["unsampled"] += f["unsampled"]
        for k, v in f["kinds"].items():
            kinds[k] = kinds.get(k, 0) + v
    n = tot["refused"] + tot["unsampled"]
    print()
    print("=" * 78)
    print("WHY THE UNDECIDABLE ONES ARE UNDECIDABLE (%d closed rewrites)" % n)
    if not n:
        print("  none -- every closed rewrite had a matched point")
        raise SystemExit(0)
    print("  REFUSED   %d / %d (%.0f%%) -- the child's space could not run the parent's best"
          % (tot["refused"], n, 100.0 * tot["refused"] / n))
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print("      %-32s %d" % (k, v))
    print("  UNSAMPLED %d / %d (%.0f%%) -- TPE never picked that point"
          % (tot["unsampled"], n, 100.0 * tot["unsampled"] / n))
    print()
    if tot["refused"]:
        print("  A REFUSED point is NOT missing at random: it means the edit changed the")
        print("  footprint enough that the parent's tile no longer runs. Those are among the")
        print("  rewrites that moved a resource dim MOST, so excluding them BIASES the pooled")
        print("  no-op rate UPWARD. Report the rate with this share stated.")
    else:
        print("  Every one is a sampler coincidence, which carries no information about the")
        print("  edit => the decidable set is unbiased on this axis. Note the axis: this rules")
        print("  out refusal, not every route to a biased sample.")
