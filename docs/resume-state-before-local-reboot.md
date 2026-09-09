# Resume state — written before a local reboot, 2026-09-09

The local Windows machine is being rebooted; this session may be interrupted. The Linux
experiment boxes are NOT being touched. This file is what to read on resume.

## Answer to "will the reboot affect the experiments": no

Verified rather than assumed, on both boxes:

| check | box 1 (L3:21) | box 2 (L3:43) |
|---|---|---|
| orchestrator pid | 645571 | 245054 |
| PPID | **1** (reparented to init) | **1** |
| controlling terminal | **`?`** — none | **`?`** — none |
| session / process group | own (645487) | own (244972) |
| stdin | `/dev/null` | `/dev/null` |
| stdout+stderr | its own log file on disk | its own log file on disk |
| `sshd` ancestor | **none** | **none** |
| opencode server | pid 645636, owned by the orchestrator (ppid 645571) | pid 245119, ppid 245054 |

Nothing in either run's process tree descends from an ssh connection. A local reboot drops my
ssh sessions; SIGHUP is delivered to processes with a controlling terminal in the terminating
session, and these have neither. The agents call a *local* opencode server on 127.0.0.1:4096 on
each box, spawned by and owned by the orchestrator itself, so agent calls keep working too.

Cleaned up before rebooting: three leftover monitor poll loops on box 1 (pids 706693, 807784,
832443) that WERE children of my ssh sessions. They were read-only pollers, so orphaning them
was harmless, but they would have lingered. Confirmed after killing them that L3:21 was
unaffected (1035 events, VERDICT WORKING).

**What DOES stop at reboot:** every Monitor task in this session. They live in the local CLI, not
on the boxes. So no completion notification will arrive during the downtime — the runs will
simply carry on, and their events.jsonl is the record. Re-arm monitors on resume.

## Snapshot at the moment of writing

### box 1 — L3:21 (`run-l3-21-20260908-232211`, started 2026-09-08 23:22)

- **655.8 min elapsed of a 720 min (12 h) budget** — this run will END during the downtime.
- 1036 events, 700 trials: 512 complete, 141 correctness_mismatch, 47 infeasible_shared_memory
- **best trial 3.6050 ms** (tuned). Incumbent to beat: **6.92 ms** → 1.92x
- This run's own baselines: eager 15.581, eager_tf32 13.864, torch_compile 14.154,
  **torch_compile_tf32 11.387** (the strongest)
- Measurement quality checked: per-trial CV median 0.6%, 2 near-ties of 314, combined SEM 0.07%
  of best. The leader is genuinely separated, unlike L3:48's earlier run.
- Running commit `e2d3d32` — so it does NOT have the ceilings fix (`9e8066d`) or the rewriter
  backend switch (`1ef142d`). It is the CONTROL for both, per
  `docs/decision-l3-21-continues-without-ceilings-fix.md`.

### box 2 — L3:43 (`run-l3-43-20260909-015247`, started 2026-09-09 01:52)

- 505.3 min elapsed of 720 min — will still be running after the downtime.
- 811 events, 520 trials: 450 complete, 68 infeasible_shared_memory, 1 mismatch, 1 runtime_error
- **best trial 2.9972 ms** — this has just passed the **3.0126 ms** incumbent (`final_reeval_ms`
  of run-l3-43-20260908-053708). Do NOT quote it as a win yet: that is `tuned_ms`, and the
  comparison must be made on this run's own post-termination re-eval.
- Running commit `e840642` — the first run whose kernel-writing agents see the measured ceilings
  (verified live: its generator's `device.md` carries `DRAM 0.911 TB/s` and
  `fp16 164.4 TFLOP/s = 1.85x tf32`) and the first where a rewrite can switch backend.

## On resume, in this order

1. **Check both boxes are still alive and what changed:**
   ```bash
   ssh autodl  '/root/orchestrator_running.sh -v'
   ssh autodl2 '/root/orchestrator_running.sh -v'
   ssh autodl  '/root/autodl-tmp/orch-venv/bin/python /root/run_summary.py \
                /root/autodl-tmp/opop-workspace/opop-glm/runs-l3/run-l3-21-20260908-232211'
   ssh autodl2 '/root/autodl-tmp/orch-venv/bin/python /root/run_summary.py $(cat /root/box2.rundir)'
   ```

2. **L3:21 will likely have finished.** Analyse it per the runbook's seven-point order
   (`docs/runbook-two-box-l3-experiments.md` §5). Specifically:
   - `final_reeval_ms`, never `tuned_ms`
   - whether it ENDED on wall clock or stopped early with budget unused (only 1 of 19 earlier
     runs was ended by its clock; four "converged" having used 0-2 of 12 rewrite rounds)
   - `fp64_rescued_trials` on the winner, out of `quick_correctness_trials` (3), not
     `correctness_trials` (5)
   - the per-precision trial table — `bf16` was 0-for-22 here, see deferred finding D2
   - **backend counts must come from box 2, not this run** (deferred finding D3)

3. **Then start L3:48 on box 1**, at whatever HEAD is current, after applying only fixes that
   clear the four-part bar (on-disk evidence; general not case-specific; a negative control that
   fails on revert; suite green on a box with a GPU). Incumbent: 1.41 ms / 9.29x.

4. **Re-arm monitors** using `orchestrator_running.sh` (never an inline `pgrep -f
   kernel_optimizer.cli`, which matches its own wrapper — that bug silently swallowed a
   completion notification once already).

## Open items

Deferred findings are in `docs/deferred-findings-during-l3-runs.md`: D1 (dead knob values keep
being drawn, 10.8% of trials), D2 (bf16 0-for-22, gate is correct), D3 (backend switch
unverified), D4 (ceilings' behavioural effect has no clean experiment), D5 (F5's prescreen is
net −9.6 min on L3:21 because its cost is per kernel, not per variant).

Everything through `277e5bd` is pushed to `https://github.com/Fudan-SMI-lab/opop.git` branch
`v2`. Both boxes and the Windows checkout are on one lineage.

**At delivery, remind the operator to rotate**: the plaintext API keys in
`.opencode/opencode.jsonc`, `opencode_backup.jsonc`, `kimi-provider.yaml`, the global
`~/.config/opencode/opencode.jsonc`, and every GLM run directory's sandbox `opencode.json`
(0644); plus the GitHub PAT. The key has been on two shared machines.
