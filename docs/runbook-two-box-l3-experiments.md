# Two-box L3 experiment runbook — one task at a time, fixes between tasks

Written 2026-09-08 against branch `v2` @ `1ef142d`. Both experiment boxes are RTX 4090 (sm_89),
both reachable from the orchestrating session, so this document is the **operating procedure**,
not a handoff to a human. It is written so that an agent given only this file plus ssh access can
run its half correctly.

If you are picking this up on a box that is behind, read
`docs/handoff-update-a-linux-box-to-current.md` FIRST — it carries the four silent-failure
checks and the update procedure. This document assumes you have done that.

---

## 0. Why one task at a time, and not a chain

The previous plan launched `level3:43 → 21 → 48` as an unattended 36-hour chain. That is the wrong
shape for this phase: it spends the whole budget before any of it can be read, so a defect found in
task 1 is inherited by tasks 2 and 3 and the three results are no longer independent evidence about
the same framework.

**The rule for this phase: after each task completes, analyse it, apply only fixes that are
certain and low-risk, and start the next task on the fixed version.** Anything uncertain, or
anything needing a large change, is recorded and deferred to a consolidated round — do not carry a
speculative fix into a 12-hour run.

A fix qualifies as "certain and low-risk" only if all four hold:

1. The defect is demonstrated from **on-disk evidence** (events.jsonl, a run artifact, a probe),
   not inferred from a notification or a plausible story.
2. The fix is **general**, not case-specific. No hardcoding for one task, one candidate, one
   shape, one dtype. If the change reads "when task == 43, do X", it is disqualified regardless of
   how well it works.
3. It has a **negative control**: revert the fix, and a test fails. A test that passes both before
   and after proves nothing.
4. The full suite passes on the box that has a GPU (see §1.3 — the skip count matters).

---

## 1. Both boxes: bring to current

### 1.1 Update without destroying local work

Both boxes carry local Linux-port edits to three files. **Never `scp` a mainline file onto a box
without checking first** — this has already destroyed work once.

```bash
cd /root/autodl-tmp/opop-workspace/opop
git status --porcelain                       # expect the 3 linux variants + untracked configs
```

The three expected modifications are `src/kernel_optimizer/cli.py`,
`src/kernel_optimizer/agents/runtime.py`, `src/kernel_optimizer/gpu/worker_client.py`, and each
should be **byte-identical (modulo CRLF) to its committed variant** under `linux-server/`:

```bash
for pair in "cli:src/kernel_optimizer/cli.py" \
            "runtime:src/kernel_optimizer/agents/runtime.py" \
            "worker_client:src/kernel_optimizer/gpu/worker_client.py"; do
  v="${pair%%:*}"; m="${pair##*:}"
  printf "%-46s " "$m"
  diff -q <(sed 's/\r$//' "$m") <(sed 's/\r$//' "linux-server/$v.linux.py") >/dev/null \
    && echo OK || echo "DIFFERS -- investigate before updating"
done
```

If all three say OK, the working tree holds nothing unique and the update is safe:

```bash
S=/root/box-backup/pre-merge-$(date +%Y%m%d-%H%M%S); mkdir -p $S
tar czf $S/dirty.tar.gz $(git status --porcelain | awk '{print $2}') 2>/dev/null
git rev-parse HEAD > $S/HEAD.txt; git status --porcelain > $S/status.txt

git fetch https://github.com/Fudan-SMI-lab/opop.git v2
git merge-base --is-ancestor HEAD FETCH_HEAD && echo "behind, safe to ff" || echo "DIVERGED - stop"
git stash push -u -q -m "linux variants + local configs"
git merge --ff-only FETCH_HEAD
git stash pop                                # expect 0 conflicts
```

If `git stash pop` reports conflicts, resolve **by provenance, not by preference**: a conflicted
file that is a snapshot of an older upstream commit takes the upstream side (`git checkout --ours`
after a stash pop, since a stash pop's "ours" is the newly merged HEAD); a file that is a Linux
variant takes the box side and must then match `linux-server/` exactly. Verify with the loop above
afterwards.

### 1.2 The prompt-doc files reach a RUNNING experiment

`_contract_doc()` and `_triton_pitfalls_doc()` call `read_text` per invocation with no caching, and
both boxes are editable installs. So **editing `candidate_contract.md` changes what the next agent
call in an already-running experiment sees.** Driver code (orchestrator, worker, config) does not:
that needs a restart. When you fix something mid-run, know which of the two you just did, and say
so in the analysis — a "fix verified" claim has to name the clock it applies to.

### 1.3 The suite, on the box with the GPU

```bash
/root/autodl-tmp/orch-venv/bin/python -m pytest tests/ -q
```

**Expect `376 passed, 2 skipped`.** If you see 9 skipped, the orchestrator venv has no torch and
seven GPU-semantics tests are silently not running:

```bash
/root/autodl-tmp/orch-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu129
```

This is not pedantry. `test_relaxed_close_semantics` asserted the wrong verdict from the commit
that introduced it and stayed green for weeks, because it skips wherever torch is absent. A suite
that skips its GPU assertions is not the suite you think you are running.

### 1.4 Preflight, every time, before a 12-hour run

```bash
grep -n '"external_directory": "allow"' src/kernel_optimizer/agents/sandbox.py
grep -n 'OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX' configs/experiments_l3_glm_linux.yaml
python -c "import sys; sys.path.insert(0,'src'); \
  from kernel_optimizer.gpu.worker_main import capture_timing_samples; print('median fix present')"
python scripts/probe_opencode_providers.py     # must list zhipuai with glm-5.3 and a set apiKey
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # must be idle
pgrep -af 'kernel_optimizer.cli' || echo "no orchestrator running"
```

Each of these has cost a real run:

- `external_directory` omitted → a headless permission ask nobody answers → three dead agent
  calls, ~90 minutes for zero output. The verified values are `"allow"` and `{"*": "allow"}`;
  `"always"` was never tested, do not "fix" it to that.
- `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` unset → glm-5.3 truncated at 32000 tokens on its
  **first** L3 call, producing zero files. It is environment-only; there is no config-file entry
  for it other than the `server_env` block. L1 smoke peaks around 5589 tokens, so smoke cannot
  detect this.
- Two orchestrators sharing one GPU destroys the timings, which are the entire deliverable.

### 1.5 Check the calibration numbers, not the word "cached"

```bash
python - <<'EOF'
import json, pathlib, sys
run = sys.argv[1]
for ln in pathlib.Path(run, "events.jsonl").read_text(encoding="utf-8").splitlines():
    e = json.loads(ln)
    if e["type"] in ("CALIBRATION_LOADED", "CALIBRATION_MEASURED"):
        p = e["payload"]
        print(e["type"], "source=", p.get("source"))
        for k in ("dram_tbs","fp32_tflops","tf32_tflops","fp16_tflops","bf16_tflops"):
            print(f"   {k:<14} {p.get(k)}")
EOF
```

`fp16_tflops` and `bf16_tflops` **must be non-zero** on a card with tensor cores. A zero there
means a low-precision candidate is scored against the tf32 denominator — roughly half its real
ceiling on a 4090 — so it reads as *saturated* when it has headroom. That inverted 13 of 25
verdicts on L3:43. A cache below `CALIBRATION_SCHEMA_VERSION` is now refused automatically, but
read the numbers anyway: the next added measurement will have the same failure mode if someone
forgets to bump the constant.

---

## 2. Task assignment

Two boxes, three tasks, and the tasks are **not** interchangeable — each has a different incumbent
and probes something different. Run one task per box at a time.

| task | incumbent to beat | why this task | notes |
|---|---|---|---|
| **level3:43** MinGPTCausalAttention | **3.0126 ms** (`final_reeval_ms`, run-l3-43-20260908-053708) | 69.09x fusion headroom — the largest task-level lever of the three, so the strongest test of whether task_cost/bottleneck feedback actually steers the agent | the reference run was terminated at 11.73 h of 12 h, so its number is a **lower bound** |
| **level3:21** EfficientNetMBConv | **6.92 ms** tuned / 2.18x | train-mode BatchNorm: the reference runs in TRAIN mode, so batch statistics are required and `running_mean`/`running_var` are wrong. Tests whether `eval_semantics.md` reaches the agent | a previous run stopped at 2.05 h of 12 h with 2 of 6 rewrite rounds used |
| **level3:48** Mamba2ReturnY | **1.41 ms** / 9.29x, 94.7% of measured DRAM roof | bandwidth-bound at ~90%, lowest dual-precision noise floor (0.9554) — most likely to spend budget on the correctness gate rather than optimization | little headroom left; a null result here is informative, not a failure |

**Suggested split:** box 2 takes `level3:43` first (it holds the reference run and its artifacts);
box 1 takes `level3:21` first. Whichever finishes first, analyse, fix, then that box takes
`level3:48`.

Do **not** compare a result from one box against a number measured on the other without
re-measuring the baseline: both are 4090s, but `final_reeval_ms` is the only number worth quoting
and it is measured per run.

---

## 3. Launching one task

```bash
cd /root/autodl-tmp/opop-workspace/opop
source /root/autodl-tmp/orch-venv/bin/activate
LOG=/root/autodl-tmp/opop-workspace/opop-glm/l3-<task>-$(date +%Y%m%d-%H%M%S).log
nohup python -m kernel_optimizer.cli --config configs/experiments_l3_glm_linux.yaml \
      run --task level3:43 > $LOG 2>&1 &
echo $LOG
```

Notes that matter:

- **`nohup ... &`, never a foreground `timeout`.** A foreground run dies with your shell, and a
  two-minute tool timeout will kill it (observed: exit 143).
- The orchestrator has **no signal handler**, so a `SIGTERM` leaves an orphaned opencode server
  behind. If you must stop a run, kill it and then check `pgrep -af opencode` and clean up.
- `run.runs_dir` in the config points at `opop-glm/runs-l3`; runs are timestamped so they do not
  collide.
- Before launching, confirm the config's `device:` block matches THIS box. `_device_doc()` writes
  it verbatim into every agent prompt, so a wrong block tells every agent the wrong hardware and
  nothing errors. A config naming an "RTX 5080 Laptop (sm_120)" was once used on a 4090.
- `load_config` reads **one** YAML. `default.yaml` is not a base layer — an omitted key falls back
  to the dataclass default, not to `default.yaml`. That is how a whole run's agents were told the
  GPU was `"unknown"`.

---

## 4. Watching a run without wasting attention

Do not tail the log. Two scripts read the run's own events from disk:

```bash
python scripts/run_summary.py <run-dir>   # event histogram, trials, best median, prescreen, calib
python scripts/run_pace.py    <run-dir>   # per-step wall clock, per-module agent cost, ETA inputs
```

Observed pace on L1:42 smoke (glm-5.3, box 2) for calibrating expectations — L3 will be slower:

| step | cost |
|---|---|
| generator | 6.7 min |
| parameterizer | 2.4 min median |
| analyst | 2.3 min median |
| rewriter | **15.1 min median** |
| tuning trial | 64–175 s |
| F5 prescreen | ~60 s per candidate (16 configs, one worker process) |

**Only trust on-disk `events.jsonl`.** Notification content has been malformed and inconsistent
before; re-verify anything surprising against the file. Two specific traps:

- A report can say both `PROVISIONAL` and "killed or crashed" for the same run. The `ended` lines
  are unreliable for an in-flight run — check the process list and the last event's timestamp.
- `pgrep -f kernel_optimizer.cli` matching your own ssh command line is a false positive; look at
  the actual `python -m` process.

---

## 5. When a task finishes: the analysis, in this order

1. **The number.** `final_reeval_ms` from the independent post-run re-eval, never `tuned_ms`
   (systematically optimistic by +1.5–6.7%, though twice it came out *faster*, so do not "discount
   tuned by N%" either — quote the re-eval).
2. **Did it actually end, or stop?** Read the report's two `ended` lines and the wall clock. Of 19
   earlier runs only ONE was ended by its wall clock; four "converged" having used 0–2 of 12
   rewrite rounds. A run that stopped with budget unused is a finding about the stopping rule, not
   a result about the task.
3. **Correctness accounting.** `fp64_rescued_trials` on the winner, out of
   `quick_correctness_trials` (3) for tuning-derived counts — not `correctness_trials` (5). A
   winner needing 5/5 rescues is a weaker result than one that cleared the absolute gate.
4. **Per-precision trial table.** A precision with zero completed trials means its whole branch
   was never measured; the dominant `failure_kind` says whether that was shared memory,
   correctness, or something else. Do not assume one cause for all of them.
5. **Backend.** New this round: `RewriteCandidate` now carries `backend`, the registered value
   comes from `_detect_backend(source)`, and a declaration/source disagreement is journalled as
   `BACKEND_DECLARATION_MISMATCH`. Count both — whether any candidate chose `cuda`, and whether
   any label disagreed with its file. **35 of 35 candidates being Triton was a consequence of the
   prompts, not a finding**; this is the first run where the option is expressible, so the count
   is the evidence. Zero CUDA candidates is a legitimate outcome to report, not a failure.
6. **F5 prescreen yield.** `SPACE_PRESCREENED` carries `configs_probed`/`infeasible`. The screen
   exists because 180 of 1004 trials on one L3:43 run failed with `out of resource: shared memory`
   (18% of the budget). If `infeasible` is 0 across a whole L3 run, the screen is costing ~60 s
   per candidate and buying nothing — say so.
7. **Loop D.** `NOVELTY_PRODUCED` / `NOVELTY_REJECTED`. Loop D executed zero times in the first 18
   runs. Note that a rewrite rejection used to be logged as `NOVELTY_REJECTED`, so an old log
   overcounts it.

Write the analysis as `docs/result-l3-<task>-<what>.md`, with every number attributed to the file
it came from.

---

## 6. Feedback

Both boxes are driven from one session, so "feedback" means: after each task, report

- the headline number and the incumbent it is compared against,
- which of the seven checks above surfaced something,
- the fixes applied (with their negative controls) and the fixes deferred and why,
- the commit the next task will run on.

Push every fix to `https://github.com/Fudan-SMI-lab/opop.git` branch `v2` before starting the next
task, so both boxes and the Windows checkout stay on one lineage. Use the PAT in a one-time inline
URL only — never persist it in a remote, `git config`, or a committed file.

---

## 7. Standing constraints

- **Never run an unscoped recursive `grep`/`find` at the workspace root.** There is a 14 GB `.db`
  file there. Scope every search to a subdirectory.
- Windows side: always pass `encoding='utf-8'` to Python `open()`; the GBK default crashes.
- Plaintext API keys exist in `.opencode/opencode.jsonc`, `opencode_backup.jsonc`,
  `kimi-provider.yaml`, the global `~/.config/opencode/opencode.jsonc`, and in every GLM run
  directory's sandbox `opencode.json` (0644). The key has been on two shared machines. **Rotation
  is a delivery-time task for the operator, not something to do mid-experiment.** Do not tar or
  scp a runs directory off a box without knowing it carries keys.
- Explicitly out of scope by prior decision: cross-candidate report/hypothesis sharing; banning a
  dtype; early-pruning or greedy seed selection; `REPAIR_REVERTED`; and the change proposed in
  `docs/finding-unreachable-correctness-gate.md` (judged too dangerous to implement without an
  explicit decision).
- Agents get **per-block** device limits only. `device.md` does not carry SM count, L2 size,
  bandwidth, or the task's shapes, so any agent claim resting on occupancy is unverifiable from
  what it was given. Treat such claims as hypotheses.
