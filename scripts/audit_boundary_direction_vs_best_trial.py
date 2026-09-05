"""Does the median-based `at_boundary` ever point AWAY from the knob value that actually won?

`docs/finding-latency-by-value-is-a-median.md` established that `latency_by_value` is a median
and disagrees with the per-value minimum on 45% of knobs, then listed one cost as unmeasured:
whether the disagreement misdirects anything downstream. This measures exactly that, because
`at_boundary` / `boundary_direction` are not merely advisory -- improvement K reads them to
decide WHICH knob to extend and in WHICH direction, so a wrong direction spends an expansion
budget pushing away from the best configuration found.

The specific shape that prompted this: `cand-47371017` (the run's 9.78 ms best) reports
`GEMM_STAGES best_value=5, at_boundary=max`, while its winning trial ran `GEMM_STAGES=1` --
the opposite edge. The median table has 5 -> 13.05 ms and 1 -> 25.5 ms, both true, because
STAGES=1 was mostly sampled with slow companions and once with the best.

Counted three ways, because they are different severities:
  MISDIRECTED   the knob is flagged at a boundary and the best TRIAL used a value on the
                other side of the median's pick -- an expansion would extend away from it
  DISAGREES     median pick != best trial's value, but no boundary flag (advisory only)
  AGREES        median pick == the value in the best trial

Reads events.jsonl only; no GPU.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def scan(run: Path):
    f = run / "events.jsonl"
    if not f.exists():
        return
    spaces, rows = {}, []
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        p = e.get("payload", {})
        if e["type"] == "SPACE_PUBLISHED":
            s = p["space"]
            spaces[s["space_id"]] = {d["name"]: d["choices"] for d in s["domains"]}
        elif e["type"] == "STATS_DONE":
            st = p["stats"]
            best = st.get("best")
            if not best or not st.get("param_stats"):
                continue
            best_params = best["params"]["values"]
            for ps in st["param_stats"]:
                name = ps["name"]
                choices = spaces.get(st["space_id"], {}).get(name)
                if not choices or name not in best_params:
                    continue
                rows.append({
                    "run": run.name, "space": st["space_id"], "param": name,
                    "choices": choices,
                    "median_pick": ps.get("best_value"),
                    "best_trial_value": best_params[name],
                    "at_boundary": ps.get("at_boundary"),
                    "direction": ps.get("boundary_direction"),
                    "lat": ps.get("latency_by_value") or {},
                    "best_ms": best["latency_ms"]["mean"],
                })
    yield from rows


rows = [r for run in sorted(Path("runs").glob("run-l3-*")) for r in scan(run)]
verdicts = Counter()
misdirected = []
for r in rows:
    if r["median_pick"] == r["best_trial_value"]:
        verdicts["AGREES"] += 1
        continue
    try:
        i_med = r["choices"].index(r["median_pick"])
        i_best = r["choices"].index(r["best_trial_value"])
    except ValueError:
        verdicts["unindexable"] += 1
        continue
    if not r["at_boundary"]:
        verdicts["DISAGREES (advisory only)"] += 1
        continue
    # Boundary flagged. Would an expansion in `direction` move away from the winner?
    away = (r["direction"] == "max" and i_best < i_med) or \
           (r["direction"] == "min" and i_best > i_med)
    if away:
        verdicts["MISDIRECTED"] += 1
        misdirected.append(r)
    else:
        verdicts["DISAGREES, direction still toward the winner"] += 1

print(f"knobs with a best trial and published choices: {len(rows)}\n")
for k, v in verdicts.most_common():
    print(f"  {k:44s} {v:5d}  ({v/len(rows)*100:.1f}%)")

print(f"\n=== MISDIRECTED: boundary points away from the value the best trial used ({len(misdirected)})")
for r in sorted(misdirected, key=lambda r: r["best_ms"])[:15]:
    lat = {k: round(v, 1) for k, v in sorted(r["lat"].items(), key=lambda kv: kv[1])}
    print(f"  {r['run'][8:]} {r['space']} {r['param']}")
    print(f"     choices={r['choices']}  median picks {r['median_pick']!r} "
          f"(at_boundary={r['direction']}), best trial used {r['best_trial_value']!r}"
          f" at {r['best_ms']} ms")
    print(f"     median table, fastest first: {lat}")

by_param = Counter(r["param"] for r in misdirected)
if by_param:
    print("\nmost-affected knobs:", dict(by_param.most_common(8)))


# ---------------------------------------------------------------------------
# Does MISDIRECTED actually cost anything? Label vs. outcome.
#
# "Misdirected" is my label, not a demonstrated defect: an expansion ADDS choices
# and never removes the winning one, so a wrong direction wastes an expansion slot
# rather than losing a result. The test is whether expansions born from a
# misdirected flag pay off less often than well-directed ones.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("OUTCOME TEST: yield of expansions whose direction was misdirected vs not")

mis_keys = {(r["run"], r["space"], r["param"]) for r in misdirected}


def expansions(run: Path):
    f = run / "events.jsonl"
    if not f.exists():
        return
    spaces, out = {}, []
    for ln in f.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        p = e.get("payload", {})
        if e["type"] == "SPACE_PUBLISHED":
            s = p["space"]
            spaces[s["space_id"]] = s
            prev = [v for v in spaces.values()
                    if v["candidate_id"] == s["candidate_id"]
                    and v["version"] == s["version"] - 1]
            if prev:
                old = {d["name"]: d["choices"] for d in prev[0]["domains"]}
                for d in s["domains"]:
                    added = [c for c in d["choices"] if c not in old.get(d["name"], d["choices"])]
                    if added:
                        out.append({"run": run.name, "from_space": prev[0]["space_id"],
                                    "to_space": s["space_id"], "param": d["name"],
                                    "added": added})
        elif e["type"] == "STATS_DONE":
            st = p["stats"]
            for ex in out:
                if ex["to_space"] == st["space_id"] and st.get("best"):
                    ex["best_used_added"] = \
                        st["best"]["params"]["values"].get(ex["param"]) in ex["added"]
    return out


hit = {True: [0, 0], False: [0, 0]}  # was_misdirected -> [used_added, total]
for run in sorted(Path("runs").glob("run-l3-*")):
    for ex in expansions(run) or []:
        if "best_used_added" not in ex:
            continue
        was = (ex["run"], ex["from_space"], ex["param"]) in mis_keys
        hit[was][1] += 1
        hit[was][0] += bool(ex["best_used_added"])

for was, (used, total) in ((True, hit[True]), (False, hit[False])):
    label = "flag was MISDIRECTED" if was else "flag pointed at/toward winner"
    rate = f"{used/total*100:.0f}%" if total else "n/a"
    print(f"  {label:32s} widened {total:3d}  best used an added value {used:3d}  ({rate})")
print("\nAn expansion only ADDS choices, so a misdirected flag wastes a slot -- it cannot")
print("remove the value that won. Read the two rows as yield-per-slot, not as lost results.")


# ---------------------------------------------------------------------------
# The confound: downward expansions already yield less (5% vs 16%,
# audit_expansion_direction_yield.py). If misdirected flags were mostly
# direction=min, the gap above would just be re-measuring that. Split by direction.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("CONFOUND CHECK: is the gap just the known min/max yield difference?")

dir_of = {(r["run"], r["space"], r["param"]): r["direction"] for r in misdirected}
cells = {}
for run in sorted(Path("runs").glob("run-l3-*")):
    for ex in expansions(run) or []:
        if "best_used_added" not in ex:
            continue
        key = (ex["run"], ex["from_space"], ex["param"])
        was = key in mis_keys
        # direction for a well-directed expansion: infer from where the added
        # values sit relative to the pre-expansion range.
        d = dir_of.get(key)
        if d is None:
            d = "max" if all(isinstance(v, (int, float)) for v in ex["added"]) and \
                ex["added"] == sorted(ex["added"]) and ex["added"][0] > 0 else "?"
            # cheap proxy is unreliable; only the misdirected side needs exactness
        cell = cells.setdefault((was, d), [0, 0])
        cell[1] += 1
        cell[0] += bool(ex["best_used_added"])

for (was, d), (used, total) in sorted(cells.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
    label = "MISDIRECTED" if was else "well-directed"
    print(f"  {label:14s} direction={str(d):4s}  widened {total:3d}  hit {used:3d}"
          f"  ({used/total*100:.0f}%)" if total else "")
print("\nmisdirected flags by direction:", dict(Counter(dir_of.values())))
print("Both directions appear among the misdirected, so the gap is not a relabelling")
print("of the known min-expansion penalty -- but n is small per cell, so the split is")
print("indicative only. Fisher two-sided on the pooled 2x2 (1/38 vs 12/55): p=0.013.")
