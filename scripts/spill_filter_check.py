"""Should we filter configs by register spills, the way PyTorch Inductor does?

Inductor skips benchmarking any config with n_spills > 16 -- but ONLY for kernels it generated
itself. Verified in our installed torch 2.9.1 (triton_heuristics.py:832-847), the guard reads:

    # we don't skip configs with spilled registers when auto-tuning custom
    # (user-written) Triton kernels, as (i) we don't have any knowledge or
    # control over the kernel code; (ii) there is empirical evidence that
    # for some (complicated) custom Triton kernels, a register-spilling
    # config may yield the best latency.

Our kernels are LLM-written, i.e. exactly Inductor's `custom_kernel` case, so PyTorch's own
authors would decline to apply the filter to us. This checks whether OUR data agrees: did any
winning or near-winning configuration spill?

If spilling configs win in our runs, a spill filter would destroy results and must not be
built -- and we would have independent confirmation of PyTorch's empirical note.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs = [Path(a) for a in sys.argv[1:]]

print("%-34s %10s %9s %8s %8s %s" %
      ("run", "best ms", "spills", "regs", "shared", "would a spill>16 filter kill it?"))
grand = {"winners_with_spills": 0, "winners": 0, "near_with_spills": 0, "near": 0}
detail = []

for run in runs:
    ev = run / "events.jsonl"
    if not ev.exists():
        continue
    trials = []
    with open(ev, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") != "TRIAL_DONE":
                continue
            tr = (e.get("payload") or {}).get("trial") or {}
            if (tr.get("status") or "").lower() != "complete":
                continue
            m = (tr.get("latency_ms") or {}).get("median")
            if not isinstance(m, (int, float)):
                continue
            pr = tr.get("profile") or {}
            trials.append((m, pr.get("n_spills"), pr.get("n_regs"),
                           pr.get("shared_bytes"), tr.get("candidate_id")))
    if not trials:
        continue
    trials.sort(key=lambda t: t[0])
    best = trials[0]
    grand["winners"] += 1
    killed = isinstance(best[1], int) and best[1] > 16
    if killed:
        grand["winners_with_spills"] += 1
    print("%-34s %10.4f %9s %8s %8s %s" %
          (run.name[:34], best[0], best[1], best[2], best[3],
           "YES -- would have been skipped" if killed else "no"))

    # near-ties: within 5% of best
    band = [t for t in trials if t[0] <= best[0] * 1.05]
    spilling_in_band = [t for t in band if isinstance(t[1], int) and t[1] > 16]
    grand["near"] += len(band)
    grand["near_with_spills"] += len(spilling_in_band)
    if spilling_in_band:
        fastest_spiller = min(spilling_in_band, key=lambda t: t[0])
        pct = 100 * (fastest_spiller[0] - best[0]) / best[0]
        detail.append((run.name, len(spilling_in_band), len(band),
                       fastest_spiller[0], fastest_spiller[1], pct))

print()
print("configs inside 5%% of best that spill >16 registers:")
if not detail:
    print("  none in any run")
for name, ns, nb, ms, sp, pct in detail:
    print("  %-34s %d of %d in band; fastest spiller %.4f ms (%+.1f%% off best) with %d spills"
          % (name[:34], ns, nb, ms, pct, sp))

print()
print("SUMMARY")
print("  winning configs that a spill>16 filter would have discarded: %d of %d"
      % (grand["winners_with_spills"], grand["winners"]))
print("  near-tie configs (within 5%%) that spill >16: %d of %d"
      % (grand["near_with_spills"], grand["near"]))
print()
print("Decision rule: if either count is nonzero, DO NOT build a spill filter -- and we have")
print("our own confirmation of the empirical note in PyTorch's source.")
