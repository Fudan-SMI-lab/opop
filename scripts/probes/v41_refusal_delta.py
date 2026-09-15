"""Did a rewrite REDUCE its parent's shared-memory refusal rate?

Why this exists. The fourth attribution tier asks whether the rewrite NAMED the brief's
axis; that is a text judgement and it read 0. This asks a measured question instead: the
conditioned brief's content is "axis X is REFUSED at value v under these partners", so the
mechanism's own stated purpose is to make that refusal go away. A rewrite that relieves the
wall without naming the axis still did the thing the brief asked for, and a rewrite that
names it while refusing just as often did not.

WHAT IT MUST NOT DO. Read the delta as C2 evidence on its own. Every arm rewrites, and a
rewrite changes the whole file, so refusal rates move for reasons that have nothing to do
with a brief. The off arm is the control: the SAME statistic computed there prices what
ordinary analyst-driven rewriting does to refusal rate. Only the difference between arms is
about the mechanism, and with a handful of rewrites per run neither number carries a claim
by itself -- it is a descriptive read, printed with its denominators.

Usage: python refusal_delta.py <run_dir> [<run_dir> ...]
"""
import collections, json, sys
from pathlib import Path


def load(p: Path):
    out = []
    for line in (p / "events.jsonl").open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def main() -> int:
    for arg in sys.argv[1:]:
        run = Path(arg.rstrip("/"))
        ev = load(run)
        meta, briefed = {}, set()
        for e in ev:
            p = e.get("payload") or {}
            if e.get("type") == "CANDIDATE_REGISTERED":
                c = p.get("candidate") or {}
                if c.get("candidate_id"):
                    meta[str(c["candidate_id"])] = (
                        str(c.get("origin")), [str(x) for x in (c.get("parent_ids") or [])])
            elif e.get("type") == "CONDITIONED_BRIEF_DELIVERED" and p.get("candidate_id"):
                briefed.add(str(p["candidate_id"]))
        tot, ref, best = collections.Counter(), collections.Counter(), {}
        for e in ev:
            if e.get("type") != "TRIAL_DONE":
                continue
            t = e["payload"]["trial"]
            cid = str(t.get("candidate_id"))
            tot[cid] += 1
            if t.get("failure_kind") == "infeasible_shared_memory":
                ref[cid] += 1
            if t.get("status") == "complete":
                m = (t.get("latency_ms") or {}).get("median")
                if m is not None and (cid not in best or m < best[cid]):
                    best[cid] = m

        def rate(c):
            return (100.0 * ref[c] / tot[c]) if tot[c] else None

        print("=" * 96)
        print("%s   (%s)" % (run.parent.name, run.name))
        print("  %-16s %-9s %-16s %8s %8s %9s %9s"
              % ("rewrite", "parent", "brief?", "best_ms", "d_best", "shmRef%", "d_shmRef"))
        rows = []
        for cid, (origin, parents) in meta.items():
            if origin != "rewrite" or cid not in best:
                continue
            par = parents[0] if parents else None
            pb, pr, cr = best.get(par), rate(par), rate(cid)
            if par is None or pb is None or pr is None or cr is None:
                continue
            db = 100.0 * (best[cid] - pb) / pb
            dr = cr - pr
            rows.append((cid, par, par in briefed, best[cid], db, cr, dr,
                         tot[cid], tot[par]))
            print("  %-16s %-9s %-16s %8.4f %+7.1f%% %7.0f%% %+8.1fpp   (n=%d vs %d)"
                  % (cid, par[:9], "BRIEFED" if par in briefed else "-",
                     best[cid], db, cr, dr, tot[cid], tot[par]))
        if not rows:
            print("  no rewrite with a measurable parent")
            continue
        for label, sel in (("parent BRIEFED", [r for r in rows if r[2]]),
                           ("parent NOT briefed", [r for r in rows if not r[2]]),
                           ("ALL rewrites", rows)):
            if not sel:
                print("  %-19s n=0" % label)
                continue
            dr = sorted(r[6] for r in sel)
            db = sorted(r[4] for r in sel)
            med = lambda a: a[len(a) // 2] if len(a) % 2 else (a[len(a)//2-1] + a[len(a)//2]) / 2
            print("  %-19s n=%d  median d_shmRef %+.1fpp  relieved %d/%d  median d_best %+.1f%%"
                  % (label, len(sel), med(dr), sum(1 for x in dr if x < 0), len(dr), med(db)))
        print("  READ: the off arm's rows are the control for this statistic. A negative")
        print("  d_shmRef in ONE arm is not attribution; the arm DIFFERENCE is the reading,")
        print("  and at these denominators it is descriptive, not a test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
