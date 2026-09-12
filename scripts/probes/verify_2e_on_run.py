"""End-to-end check of the 2e path against a REAL completed run, on the production code.

A green unit suite does not prove the wiring works: every test above feeds hand-built TuningStats.
This drives `find_walls` / `select_for_probing` / `for_prompt` from an actual events.jsonl -- the
STATS_DONE the run really wrote and the infeasible TrialRecords it really produced -- and then
checks the walls against the offline probe's already-measured answers.

Expected on box 2 (from spec section 10, measured before any of this code existed):
  6 walls across 6 candidates, of which 3 have a positive tail (BLOCK_M 512 twice, STAGES_GEMM 4)
  and 3 do not (BLOCK_N 256, ATT_STAGES 4, GEMM_BK 256).

If this reports a different set, the production reader disagrees with the probe that justified
building it, and the numbers in the spec do not describe what the code will do.

NOTE ON THE TAIL PERCENTAGES. They differ from the offline probe's and are expected to: the probe
took each value's MINIMUM latency, the production reader takes `ParamStat.latency_by_value`, which
is the harness's own per-choice MEDIAN table. Median is this project's measured objective (93.2%
rank-correctness against the mean's 64.8% at 20 samples, and `min` is biased +9.8% to +156%), so the
production figure is the right one and the probe's was an approximation. What must match, and does,
is the CLASSIFICATION: which knobs are walls, and which of them have latency still improving toward
the wall. Two rows show why the sign is what is checked rather than the magnitude -- GEMM_BK reads
+11.2% on medians and -54.8% on minima, and BLOCK_N +16.3% against -4.2%; both are non-monotone
either way, so neither is probed.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.environ.get("OPOP_SRC", "src"))
from kernel_optimizer.evaluation import wall_attribution as wa       # noqa: E402
from kernel_optimizer.models.reports import TuningStats              # noqa: E402
from kernel_optimizer.store.read import read_events, trial_of        # noqa: E402

RUNS = sys.argv[1:]
if not RUNS:
    sys.exit("usage: verify_2e_on_run.py <run_dir> [...]")

EXPECTED_POSITIVE = {("cand-2dc6a2ff", "BLOCK_M"), ("cand-b010505b", "BLOCK_M"),
                     ("cand-a45cf0a9", "STAGES_GEMM")}
EXPECTED_ALL = EXPECTED_POSITIVE | {("cand-d91a77cb", "BLOCK_N"),
                                    ("cand-136dfbfb", "ATT_STAGES"),
                                    ("cand-2d8eaf9a", "GEMM_BK")}

found_all, found_pos = set(), set()

for rd in RUNS:
    name = os.path.basename(rd.rstrip("/"))
    events = read_events(rd)
    print(f"\n=== {name} ===")

    # the STATS_DONE the run itself wrote, parsed by the production model
    stats_by_cand: dict[str, TuningStats] = {}
    for ev in events:
        if ev.get("type") != "STATS_DONE":
            continue
        p = ev.get("payload") or {}
        raw = p.get("stats") or p
        try:
            st = TuningStats.model_validate(raw)
        except Exception as exc:
            print(f"  STATS_DONE did not validate: {type(exc).__name__}: {exc}")
            continue
        # A candidate can be re-tuned after a space expansion; the LAST table is the one the
        # analyst saw, so it is the one to reason over.
        stats_by_cand[st.candidate_id] = st

    # refusals exactly as the orchestrator reads them: TrialRecords, not the event
    refused_by_cand = defaultdict(list)
    for ev in events:
        if ev.get("type") != "TRIAL_DONE":
            continue
        t = trial_of(ev)
        if t.get("failure_kind") != "infeasible_shared_memory":
            continue
        prm = (t.get("params") or {}).get("values")
        if isinstance(prm, dict) and t.get("candidate_id"):
            refused_by_cand[t["candidate_id"]].append(prm)

    print(f"candidates with stats: {len(stats_by_cand)}, "
          f"with refusals: {len(refused_by_cand)}")

    for cid, st in sorted(stats_by_cand.items()):
        refused = refused_by_cand.get(cid) or []
        walls = wa.find_walls(st, refused)
        if not walls:
            continue
        probe, skipped = wa.select_for_probing(walls, 8)
        print(f"  {cid}: {len(refused)} refusals -> {len(walls)} wall(s), "
              f"{len(probe)} worth probing")
        for w in walls:
            found_all.add((cid, w.param))
            if w.monotone and w.tail_gain_pct > 0:
                found_pos.add((cid, w.param))
            tag = "PROBE" if w in probe else "skip "
            print(f"      [{tag}] {w.param:14s} ran {[int(v) if float(v).is_integer() else v for v in w.ran_values]}"
                  f" -> {int(w.refused_value)}  side={w.side}"
                  f"  monotone={str(w.monotone):5s}  tail={w.tail_gain_pct:+6.1f}%")

        # for_prompt must stay silent until a probe has actually decided something
        if wa.for_prompt(walls) is not None:
            sys.exit("FAIL: for_prompt spoke before any probe set a verdict")

print("\n" + "=" * 74)
print(f"walls found            : {len(found_all)}  expected {len(EXPECTED_ALL)}")
print(f"with a positive tail   : {len(found_pos)}  expected {len(EXPECTED_POSITIVE)}")
missing = EXPECTED_ALL - found_all
extra = found_all - EXPECTED_ALL
pos_missing = EXPECTED_POSITIVE - found_pos
pos_extra = found_pos - EXPECTED_POSITIVE
for label, s in (("missing wall", missing), ("unexpected wall", extra),
                 ("missing positive", pos_missing), ("unexpected positive", pos_extra)):
    for item in sorted(s):
        print(f"  {label}: {item}")
ok = not (missing or extra or pos_missing or pos_extra)
print("\nVERDICT:", "MATCHES the offline probe" if ok
      else "DISAGREES with the offline probe -- the spec's numbers do not describe this code")
sys.exit(0 if ok else 1)
