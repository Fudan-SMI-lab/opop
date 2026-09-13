"""Did the expanded space actually beat its own parent, or is it reporting the parent's incumbent?

WHY THIS NEEDS CHECKING. The treatment arm's expansion of `cand-5e033365` produced `sp-8acfe096` with a
best of 3.5108 ms -- IDENTICAL to `sp-7051cb7b`'s 3.5108 ms, to four decimals, after 40 fresh trials in a
wider domain. Two readings, with very different consequences:

  (a) the expansion genuinely found nothing better and the incumbent carried over -- 40 trials bought a
      negative result, which is information;
  (b) the v2 space REPORTED the parent's incumbent without re-measuring it -- the recorded
      `k-retune-cannot-disconfirm-incumbent` shape, in which case the 40 trials tell us nothing about
      that configuration and the equality is an artefact.

The discriminator is whether a trial IN THE V2 SPACE actually produced 3.5108 ms, and whether its params
match the parent's winner. If the winning params appear among v2's own trials, it re-measured; if the
best figure has no matching trial in v2, it was carried over.

Also prints the widened knob's values as ACTUALLY DRAWN. An expansion that adds COMBINE_NUM_STAGES 4 and
5 has bought nothing if TPE never sampled 4 or 5 -- that is a different failure from "sampled them and
they were worse", and `never drawn` is the one that makes the extra 40 trials wasted rather than
informative.

Run on box4:
  /root/autodl-tmp/orch-venv/bin/python /root/probe-clean/expansion_paid_off.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/opop-workspace/opop-glm/runs-v3")
RUN = "run-l3-43-20260913-202332"


def _robust_ms(lat: dict | None) -> float | None:
    """Reproduce LatencyStats.robust_ms: median if present, else mean. It is a @property and is never
    serialised, so every reader has to redo it -- reading `mean` alone would rank differently."""
    if not isinstance(lat, dict):
        return None
    v = lat.get("median")
    if v is None:
        v = lat.get("mean")
    return float(v) if isinstance(v, (int, float)) else None


def main() -> int:
    # Across every expansion: did the ADDED values ever produce the space's best point?
    verdicts: list[tuple[str, str, str, str, float | None]] = []
    for arm in ("s7-treatment", "s7-control"):
        ev = []
        for line in (BASE / arm / RUN / "events.jsonl").open(encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    ev.append(json.loads(line))
                except ValueError:
                    pass

        # space -> candidate, in publication order, so v1/v2 pairs are identifiable
        order: list[tuple[str, str]] = []
        domains: dict[str, dict[str, list]] = {}
        for e in ev:
            if e.get("type") != "SPACE_PUBLISHED":
                continue
            p = e.get("payload") or {}
            sp = p.get("space") or p
            sid = str(sp.get("space_id") or "?")
            order.append((sid, str(sp.get("candidate_id") or p.get("candidate_id") or "?")))
            doms = sp.get("domains")
            d: dict[str, list] = {}
            if isinstance(doms, dict):
                for k, v in doms.items():
                    d[str(k)] = list(v.get("choices") or []) if isinstance(v, dict) else list(v or [])
            elif isinstance(doms, list):
                for it in doms:
                    if isinstance(it, dict) and it.get("name"):
                        d[str(it["name"])] = list(it.get("choices") or [])
            domains[sid] = d

        # trials per space, with params and robust latency
        trials: dict[str, list[tuple[float, dict]]] = {}
        for e in ev:
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            if tr.get("status") != "complete":
                continue
            sid = str(tr.get("space_id") or "?")
            ms = _robust_ms(tr.get("latency_ms"))
            if ms is None:
                continue
            vals = ((tr.get("params") or {}).get("values")) or {}
            trials.setdefault(sid, []).append((ms, dict(vals)))

        by_cand: dict[str, list[str]] = {}
        for sid, cid in order:
            by_cand.setdefault(cid, []).append(sid)

        print("=" * 78)
        print(arm)
        for cid, sids in by_cand.items():
            if len(sids) < 2:
                continue
            for a, b in zip(sids, sids[1:]):
                # Key on the latency alone. A bare `sorted` on the tuples falls through to comparing the
                # params dicts whenever two trials tie on latency, which raises TypeError -- the same
                # trap `check_arm_search_parity.py` documents for its `worst` list.
                ta = sorted(trials.get(a, []), key=lambda t: t[0])
                tb = sorted(trials.get(b, []), key=lambda t: t[0])
                print("  %s   %s (v1, %d trials) -> %s (v2, %d trials)"
                      % (cid, a, len(ta), b, len(tb)))
                if not ta or not tb:
                    print("    one side has no complete trials -- nothing to compare")
                    continue
                best_a, pa = ta[0]
                best_b, pb = tb[0]
                print("    v1 best %.4f ms   v2 best %.4f ms   delta %+.2f%%"
                      % (best_a, best_b, 100 * (best_a - best_b) / best_a))

                # Did v2 have the winning point among its own trials?
                same = [ms for ms, p in tb if p == pa]
                if same:
                    print("    v1's winning params appear among v2's trials at %s ms"
                          % ", ".join("%.4f" % m for m in sorted(same)[:3]))
                else:
                    print("    v1's winning params NEVER appear among v2's trials => v2 never")
                    print("    evaluated the incumbent (k-retune-cannot-disconfirm-incumbent).")
                if abs(best_a - best_b) < 1e-9:
                    # An earlier version of this probe called a bit-identical tie "the incumbent was
                    # carried over without re-measurement", while the line directly above it reported
                    # the winning params present in v2 -- two contradictory claims in one block. The
                    # real explanation is a third one: an expanded space RE-ENQUEUES the parent's
                    # measured points as anchors, so one MEASUREMENT is shared by both spaces. It is
                    # neither two measurements that happened to tie (re-eval noise of +-2-4% makes that
                    # implausible) nor an unmeasured carry-over.
                    print("    THE TWO BESTS ARE BIT-IDENTICAL, and the winner is present in both =>")
                    print("    ONE measurement shared by both spaces (the expansion re-enqueues the")
                    print("    parent's points as anchors). Not a coincidental tie, and not an")
                    print("    unmeasured carry-over -- so this is NOT evidence the 40 trials were")
                    print("    wasted; it means the wider domain produced nothing better.")

                # were the ADDED values ever drawn?
                for k in sorted(set(domains.get(a, {})) | set(domains.get(b, {}))):
                    old, new = domains.get(a, {}).get(k, []), domains.get(b, {}).get(k, [])
                    added = [v for v in new if v not in old]
                    if not added:
                        continue
                    drawn = {p.get(k) for _, p in tb if k in p}
                    hit = [v for v in added if v in drawn]
                    print("    widened %-18s added %s ; drawn in v2: %s => %s"
                          % (k, added, sorted(x for x in drawn if x is not None),
                             "SAMPLED %s" % hit if hit
                             else "NEVER SAMPLED -- the wider domain bought nothing"))
                    if not hit:
                        verdicts.append((arm, cid, k, "never sampled", None))
                        continue
                    best_at = min((ms for ms, p in tb if p.get(k) in hit), default=None)
                    if best_at is None:
                        verdicts.append((arm, cid, k, "no latency at added value", None))
                        continue
                    delta = 100 * (best_a - best_at) / best_a
                    print("      best latency at an added value: %.4f ms (%+.2f%% vs v1 best)"
                          % (best_at, delta))
                    # Is the space's OWN best at an added value? That is the only reading under which
                    # the wider domain, rather than the extra 40 trials, produced the improvement.
                    owns = abs(best_at - best_b) < 1e-9
                    print("      the v2 optimum %s at an added value"
                          % ("IS" if owns else "is NOT"))
                    verdicts.append((arm, cid, k,
                                     "optimum at added value" if owns else "worse than v1 best",
                                     delta))

    print()
    print("=" * 78)
    print("DID THE WIDER DOMAIN ITSELF BUY ANYTHING? (per widened knob)")
    print("  %-14s %-16s %-18s %-24s %s" % ("arm", "candidate", "knob", "verdict", "delta vs v1"))
    for arm, cid, k, verdict, delta in verdicts:
        print("  %-14s %-16s %-18s %-24s %s" % (
            arm[-12:], cid[:16], k, verdict,
            "%+.2f%%" % delta if delta is not None else "-"))
    owned = [v for v in verdicts if v[3] == "optimum at added value"]
    never = [v for v in verdicts if v[3] == "never sampled"]
    print()
    print("%d of %d widened knobs put the space's OWN optimum at a newly added value."
          % (len(owned), len(verdicts)))
    if never:
        print("%d were NEVER SAMPLED, so for those the wider domain cannot have helped at all."
              % len(never))
    if verdicts and not owned:
        print("NONE did. So where an expanded space DID improve, the improvement came from the extra")
        print("`trials_per_space` it was granted, NOT from the wider domain -- which is the measured")
        print("version of 'the expansion helped but only half of it is attributable'. It also means an")
        print("expansion is not a route to values the incumbent could not reach: every added value that")
        print("was sampled came out WORSE than the parent's best.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
