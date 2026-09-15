"""Fourth tier of the brief-attribution ladder: did the rewrite actually MOVE the axis the
brief pointed at?

Tiers 1-3 (delivered / parent-briefed / parent had a gate-passing slope) are about what
the mechanism made AVAILABLE. This one is about UPTAKE, and it is the only tier that can
support "the slope steered the rewrite". Measured separately because the three earlier
tiers were all non-zero while this one was zero -- reporting any single number would
either overstate the mechanism or erase its reach.

HOW UPTAKE IS JUDGED, and why it is deliberately generous. A rewrite is structural: the
new file may have different PARAMS keys entirely, so "moved the axis" cannot mean "the
same knob has a different value". Three signals, printed separately rather than merged:

  names_axis    the change_summary mentions the axis at all
  keeps_fixed   it mentions the axis in a phrase that says it is UNCHANGED ("same knobs
                ... NUM_STAGES"), which is the opposite of uptake and is what the live
                P1 rewrites did
  axis_in_new   the axis appears in the rewrite's own PARAMS (from the registered source)

Naming an axis to say "I did not touch it" must not count as uptake, so `keeps_fixed`
subtracts. Where the signals disagree the row is printed as AMBIGUOUS rather than scored:
a judgement call about one rewrite belongs to a human reading the diff, not to a grep.

Usage: python axis_uptake.py <run_dir> [<run_dir> ...]
"""
import json, re, sys
from pathlib import Path

FIXED_PAT = re.compile(
    r"(same|unchanged|identical|keeps?|preserv\w*|retain\w*|inherit\w*)[^.;]{0,120}",
    re.I)

def main():
    for arg in sys.argv[1:]:
        run = Path(arg)
        ev = []
        for line in (run / "events.jsonl").open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                try: ev.append(json.loads(line))
                except ValueError: pass

        parents, src_of = {}, {}
        for e in ev:
            if e.get("type") == "CANDIDATE_REGISTERED":
                c = (e.get("payload") or {}).get("candidate") or {}
                cid = c.get("candidate_id")
                if cid:
                    parents[cid] = [str(x) for x in (c.get("parent_ids") or [])]
        # Axes a parent had a MINTED (gate-passing) contrast on, with direction.
        minted = {}
        for e in ev:
            p = e.get("payload") or {}
            if e.get("type") == "SCAN_BLOCK_DONE" and p.get("kind") == "C4" and p.get("direction"):
                minted.setdefault(str(p.get("candidate_id")), set()).add(
                    (str(p.get("axis")), str(p.get("direction"))))

        print("=" * 76)
        print("%s   (%s)" % (run.parent.name, run.name))
        rows = 0
        for e in ev:
            if e.get("type") != "REWRITE_PRODUCED":
                continue
            p = e["payload"]
            cid = str(p.get("candidate_id"))
            summary = str(p.get("change_summary") or "")
            pars = parents.get(cid, [])
            axes = sorted({a for q in pars for (a, d) in minted.get(q, set())})
            dirs = {a: d for q in pars for (a, d) in minted.get(q, set())}
            if not axes:
                continue
            rows += 1
            print("  %s  hyp=%s  parent=%s" % (cid, p.get("hypothesis_id"), ",".join(pars)))
            for ax in axes:
                names = ax.lower() in summary.lower()
                fixed = any(ax.lower() in m.group(0).lower()
                            for m in FIXED_PAT.finditer(summary))
                # `expectations` names resource dimensions, not axes, so it cannot settle
                # uptake; it is printed for context only.
                exp = [d.get("dimension") for d in (p.get("expectations") or [])
                       if isinstance(d, dict)]
                verdict = ("KEPT FIXED (anti-uptake)" if fixed else
                           "NAMED" if names else "NOT MENTIONED")
                print("      axis %-14s brief said %-8s -> %s" % (ax, dirs[ax], verdict))
                if exp:
                    print("        (declared dimensions: %s)" % ", ".join(map(str, exp)))
        if not rows:
            print("  no rewrite descends from a parent with a gate-passing slope yet")
    return 0

if __name__ == "__main__":
    sys.exit(main())
