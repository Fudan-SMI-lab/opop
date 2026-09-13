"""CAN single-knob ablation bridge the gap at all? Measure one-knob deltas from the record.

WHY. §4.5 diagnoses attribution's failures as a reach problem: the refused point differs from
theta* in several knobs at once, so no single revert brings it under the limit. That diagnosis
predicts something checkable, for free: single-knob changes should rarely move shared_bytes by
the factor needed. If instead one knob routinely moves it 2x, the diagnosis is wrong and the
failures need another explanation.

WHAT IS MEASURED, and it is measurement rather than a model. The record holds shared memory for
two kinds of config: measured trials (profile.shared_bytes) and refused ones
(CONFIG_SCREENED_INFEASIBLE.max_shared). Take every PAIR of those that differs in EXACTLY ONE
knob and report the ratio. That is a local difference, which is the only thing this project has
been able to measure -- three attempts at a separable global resource map were disconfirmed
(`resource-map-is-not-separable`), so nothing here fits or extrapolates a surface.

THE PAIRING MUST BE PER KERNEL. The screen reports max_shared for a NAMED kernel (_flash_attn,
_gemm, _gemm_bias all appear in one run), while a trial's profile.shared_bytes is the figure the
whole trial was judged on. Pairing across those would compare two different buffers and
manufacture ratios, so the kernel name is part of the grouping key and "(trial)" is its own key.

POSITIVE CONTROL. `--selftest` builds a space where one knob doubles shared_bytes (must be found,
with ratio 2.0) and one where the only pairs differ in two knobs (must find nothing, and say so
rather than printing an empty distribution as if it were a result).
"""
from __future__ import annotations

import json
import os
import statistics
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


def collect(events: list[dict]) -> dict:
    """(group, kernel) -> list of (params, shared_bytes, was_refused)."""
    obs: dict[tuple[str, str], list[tuple[dict, int, bool]]] = {}
    for e in events:
        t = e.get("type")
        p = e.get("payload") or {}
        if t == "TRIAL_DONE":
            tr = p.get("trial") or {}
            sb = (tr.get("profile") or {}).get("shared_bytes")
            if isinstance(sb, int):
                obs.setdefault((str(tr.get("space_id") or "?"), "(trial)"), []).append(
                    (_params(tr), sb, False))
        elif t == "CONFIG_SCREENED_INFEASIBLE":
            ms = p.get("max_shared")
            if isinstance(ms, int):
                obs.setdefault((str(p.get("candidate_id") or "?"), str(p.get("kernel") or "?")),
                               []).append((_params(p), ms, True))
    return obs


def report(obs: dict, label: str, limit: int = 101376, quiet: bool = False) -> dict:
    out = {"pairs": 0, "by_knob": {}, "ratios": [], "bridging": 0, "groups": 0,
           "mixed_groups": 0, "n_over": 0, "n_under": 0}
    if not quiet:
        print("=" * 78)
        print(label)
    for (group, kernel), rows in sorted(obs.items()):
        if len(rows) < 2:
            continue
        out["groups"] += 1
        # Can a crossing even be SEEN in this group? Trials (which ran, so they are under the
        # limit) and screen refusals (which are over it) land in DIFFERENT groups by construction,
        # because their shared-memory figures describe different buffers. So a group holding both
        # sides is the precondition for observing a crossing at all -- without it, "0 bridging
        # pairs" is forced by the grouping and says nothing about knobs.
        n_over = sum(1 for _, s, _ in rows if s > limit)
        out["n_over"] += n_over
        out["n_under"] += len(rows) - n_over
        if n_over and n_over < len(rows):
            out["mixed_groups"] += 1
        for i in range(len(rows)):
            pi, si, _ = rows[i]
            for j in range(i + 1, len(rows)):
                pj, sj, _ = rows[j]
                if set(pi) != set(pj) or not pi:
                    continue
                diff = [k for k in pi if pi[k] != pj[k]]
                if len(diff) != 1:
                    continue
                hi, lo = max(si, sj), min(si, sj)
                if lo <= 0:
                    continue
                ratio = hi / lo
                out["pairs"] += 1
                out["ratios"].append(ratio)
                st = out["by_knob"].setdefault(diff[0], {"n": 0, "max_ratio": 1.0})
                st["n"] += 1
                st["max_ratio"] = max(st["max_ratio"], ratio)
                if (si > limit) != (sj > limit):
                    out["bridging"] += 1
    return out


def _selftest() -> int:
    def _trial(space, params, sb):
        return {"type": "TRIAL_DONE", "payload": {"trial": {
            "space_id": space, "status": "complete", "params": {"values": params},
            "profile": {"shared_bytes": sb}}}}

    ok = True
    f = report(collect([_trial("sp", {"BLOCK_M": 64, "BLOCK_N": 64}, 32768),
                        _trial("sp", {"BLOCK_M": 128, "BLOCK_N": 64}, 65536)]),
               "st", quiet=True)
    if f["pairs"] != 1 or abs(f["ratios"][0] - 2.0) > 1e-9:
        print("FAIL: a one-knob doubling was not measured (%s)" % f)
        ok = False
    elif f["by_knob"].get("BLOCK_M", {}).get("max_ratio") != 2.0:
        print("FAIL: the ratio was not attributed to BLOCK_M (%s)" % f["by_knob"])
        ok = False
    f = report(collect([_trial("sp", {"BLOCK_M": 64, "BLOCK_N": 64}, 32768),
                        _trial("sp", {"BLOCK_M": 128, "BLOCK_N": 128}, 131072)]),
               "st", quiet=True)
    if f["pairs"]:
        print("FAIL: a two-knob difference was counted as one knob (%s)" % f)
        ok = False
    f = report(collect([_trial("sp", {"BLOCK_M": 64}, 65536),
                        _trial("sp", {"BLOCK_M": 128}, 131072)]), "st", quiet=True)
    if f["bridging"] != 1:
        print("FAIL: a limit crossing was not counted (%s)" % f)
        ok = False
    elif f["mixed_groups"] != 1:
        print("FAIL: a group holding both sides was not recognised as mixed (%s)" % f)
        ok = False
    # And the negative: a group entirely under the limit must NOT be called mixed, or the
    # "crossings are not observable" branch could never fire and the fix would be inert.
    f = report(collect([_trial("sp", {"BLOCK_M": 64}, 32768),
                        _trial("sp", {"BLOCK_M": 128}, 65536)]), "st", quiet=True)
    if f["mixed_groups"] or f["bridging"]:
        print("FAIL: an all-under group was treated as mixed (%s)" % f)
        ok = False
    elif f["n_under"] != 2 or f["n_over"]:
        print("FAIL: the over/under tally is wrong (%s)" % f)
        ok = False
    # A trial and a named-kernel screen observation must NOT pair: different buffers.
    f = report(collect([_trial("sp", {"BLOCK_M": 64}, 65536),
                        {"type": "CONFIG_SCREENED_INFEASIBLE", "payload": {
                            "candidate_id": "sp", "kernel": "_gemm", "max_shared": 131072,
                            "limit": 101376, "params": {"BLOCK_M": 128}}}]), "st", quiet=True)
    if f["pairs"]:
        print("FAIL: a trial was paired with a named-kernel screen (%s)" % f)
        ok = False
    # Two screen observations on the SAME kernel must pair -- otherwise the key is too strict and
    # the probe would report "no pairs" on a record full of them.
    f = report(collect([
        {"type": "CONFIG_SCREENED_INFEASIBLE", "payload": {
            "candidate_id": "c", "kernel": "_gemm", "max_shared": 131072, "limit": 101376,
            "params": {"BLOCK_M": 128}}},
        {"type": "CONFIG_SCREENED_INFEASIBLE", "payload": {
            "candidate_id": "c", "kernel": "_gemm", "max_shared": 262144, "limit": 101376,
            "params": {"BLOCK_M": 256}}}]), "st", quiet=True)
    if f["pairs"] != 1:
        print("FAIL: two same-kernel screens did not pair (%s)" % f)
        ok = False
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    paths = [a if a.endswith(".jsonl") else os.path.join(a, "events.jsonl")
             for a in sys.argv[1:] if not a.startswith("-")]
    tot = {"pairs": 0, "by_knob": {}, "ratios": [], "bridging": 0, "groups": 0,
           "mixed_groups": 0, "n_over": 0, "n_under": 0}
    for path in paths:
        if not os.path.exists(path):
            continue
        f = report(collect(_read(path)), "/".join(path.split(os.sep)[-3:-1]))
        for k in ("pairs", "bridging", "groups", "mixed_groups", "n_over", "n_under"):
            tot[k] += f[k]
        tot["ratios"].extend(f["ratios"])
        for k, st in f["by_knob"].items():
            d = tot["by_knob"].setdefault(k, {"n": 0, "max_ratio": 1.0})
            d["n"] += st["n"]
            d["max_ratio"] = max(d["max_ratio"], st["max_ratio"])
    print()
    print("=" * 78)
    if not tot["pairs"]:
        print("NO PAIR in the record differs in exactly one knob (%d group(s) examined)."
              % tot["groups"])
        print("So this question is UNANSWERED, not answered in the negative: TPE does not sample")
        print("one-knob-apart points, which is itself why single-knob ablation has to synthesise")
        print("them (only 1%% were already measured, per price_multiknob_ablation.py).")
        raise SystemExit(0)
    rs = sorted(tot["ratios"])
    n = len(rs)
    print("SINGLE-KNOB REACH over %d one-knob pair(s) in %d (group, kernel) group(s)"
          % (n, tot["groups"]))
    print("  shared_bytes ratio: median %.2fx  p90 %.2fx  max %.2fx   (1.00 = knob moves nothing)"
          % (statistics.median(rs), rs[int(0.9 * (n - 1))], rs[-1]))
    # Report the crossing count ONLY where a crossing could have been observed. Printing
    # "0 / 603 (0%)" when no group holds both sides of the limit states a structural impossibility
    # as if it were a measurement -- and I read it that way myself before checking.
    if tot["mixed_groups"]:
        print("  pairs where ONE knob crosses the %d limit: %d / %d (%.0f%%)  [%d group(s) hold "
              "both sides]" % (101376, tot["bridging"], n, 100.0 * tot["bridging"] / n,
                               tot["mixed_groups"]))
    else:
        print("  CROSSINGS ARE NOT OBSERVABLE IN THIS RECORD: %d observation(s) sit above the"
              % tot["n_over"])
        print("  %d limit and %d below, but NO (group, kernel) holds both sides, so no pair within"
              % (101376, tot["n_under"]))
        print("  a group can cross it. The reason is structural: a trial's shared_bytes comes from")
        print("  a config that RAN (hence under the limit) and a screen refusal's max_shared from")
        print("  one that did not (hence over), and those are different groups because they")
        print("  describe different buffers. So the crossing count is withheld rather than")
        print("  printed as 0 -- it is unmeasurable here, not measured to be zero.")
    print("  per knob (n, largest single-knob ratio seen):")
    for k, st in sorted(tot["by_knob"].items(), key=lambda kv: -kv[1]["max_ratio"])[:12]:
        print("      %-16s n=%-4d max %.2fx" % (k, st["n"], st["max_ratio"]))
    print()
    print("HOW THIS BEARS ON §4.5. The refused configs there exceed the limit by 1.13x-2.26x, and")
    print("a single knob can only attribute such a refusal if reverting it alone removes that")
    print("much. On MAGNITUDE a single knob can: p90 is %.2fx and the max %.2fx, so the needed"
          % (rs[int(0.9 * (n - 1))], rs[-1]))
    print("factors are within one knob's range. The median %.2fx is not, so it is not typical."
          % statistics.median(rs))
    print()
    print("=> MAGNITUDE IS NOT THE OBSTACLE; POSITION IS. Attribution starts from theta*, and")
    print("   theta* is a config that RAN, so it sits under the limit -- as does every measured")
    print("   point near it. Enlarging the set of origins on that side (§4.5 fix 1) keeps every")
    print("   origin under the limit. Only walking BACK from the refused point (§4.5 fix 2)")
    print("   traverses the limit at all. That orders the two fixes, and this probe cannot see")
    print("   the crossing itself for the structural reason stated above.")
    print()
    print("WHAT THIS IS NOT. Local differences only. It does not fit a surface and must not be read")
    print("as one: three attempts at a separable resource map were disconfirmed in this project")
    print("(96 points, 0 hits; 13 non-monotone slices). A knob's ratio here is what it did between")
    print("two specific measured points, not a coefficient.")
