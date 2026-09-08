# Updating a Linux experiment box to the current framework

You are running an older checkout. This tells you what to pull, what changed that affects your
results, and the places where getting it wrong is silent rather than loud.

Written 2026-09-08 against branch `v2` @ `1f18cd1` (updated as fixes land; the commit is what the
text was verified against). Everything here was verified on a live box
(RTX 4090, sm_89) — where a number must be measured on *your* hardware instead of copied from
here, it says so.

---

## 1. Update

```bash
cd <your checkout>
git fetch origin v2
git log --oneline HEAD..origin/v2 | wc -l     # how far behind you are
git status --porcelain                        # MUST be empty before you pull
```

**Stop if `git status` is not empty.** A previous box had 567 lines of uncommitted local port work
that a `git pull` would have destroyed. If you have local changes, back them up first
(`git diff > /tmp/local.patch` plus a tarball of the tree) and say so, rather than stashing and
hoping.

```bash
git pull --ff-only origin v2
python -m pytest tests/ -q                    # expect 376 passed, 2 skipped
```

If the test count is materially lower than 376, your pull did not complete — do not start an
experiment. Note the skip count differs by platform and that is expected: on a Linux GPU box 2 skip
(the two WSL/9p regression guards, retired by the port), on the Windows orchestrator host 9 skip
(everything gated on `importorskip("torch")`). If you see 9 skips on a Linux box, you are running
without torch and the GPU tests are not being exercised.

That platform gap is not cosmetic. `test_relaxed_close_semantics` asserted the wrong verdict from
the commit that introduced it and stayed green for weeks, because it skips on Windows and had never
run anywhere else. **A suite that skips a third of its GPU assertions is not the suite you think you
are running** — run it on the box that has the GPU.

**You no longer need to port anything.** The native-Linux port is in the repo now
(`linux-server/*.linux.py`, `linux-server/configs/*.yaml`). If you previously hand-ported
`worker_client.py` / `runtime.py` / `cli.py`, diff yours against `linux-server/` and prefer the
committed version; it has an AST name-completeness guard in the test suite that caught a missing
`import threading` which killed a run at its first agent call.

---

## 2. What changed that affects your results

Five of these change numbers you may already have recorded. Runs from before and after are **not
comparable** on the affected quantities.

### 2.1 The tuning objective was the mean; it is now the median

This is the one that matters most. `robust_ms` reads the median and **falls back to the mean**, so
a missing median was never an error — it was a silent downgrade to an estimator measured at **64.8%
ranking accuracy** against the median's 93.2%. And the median was missing on every timing path
except one: KernelBench computes `elapsed_times`, hands it to `get_timing_stats`, then discards it.

Fixed by intercepting that one function (`capture_timing_samples()` in `gpu/worker_main.py`, called
from `main()`). It only ADDS a key; KernelBench's own mean/std/min/max are untouched, so the
`423217d` pin's evaluation semantics are intact and no vendored file is edited.

**Check it took effect on your box** — do not assume, and do not trust a memory or note that says
"fixed", because such a claim is only true of the checkout it was verified on:

```bash
python scripts/probes/probe_timing_samples_real_kernelbench.py   # needs the WORKER venv
```

It must print `PASS` for both routes with an unpatched control showing no median. Run it from the
repo (or set `KOPT_SRC`), and **not from a directory holding stray scripts**: a leftover `/tmp/nt.py`
of mine shadowed Python's own `ntpath` import and the probe died with a traceback that looked like a
harness fault. Give ad-hoc scripts their own directory. Then, on any run
you produce, the non-expiring test is to count them in that run's own log:

```python
# trials with a median vs without — must be N and 0
import json, collections
c = collections.Counter()
for ln in open("<run>/events.jsonl", encoding="utf-8"):
    e = json.loads(ln)
    if e["type"] == "TRIAL_DONE":
        lat = (e["payload"].get("trial") or {}).get("latency_ms") or {}
        if lat:
            c["with" if lat.get("median") is not None else "without"] += 1
print(c)
```

### 2.2 `speedups_median` now populates

Baselines have medians now, so this field appears where it used to be suppressed. It is safe: the
headline `speedups` and `honest_verdict` stay mean-over-mean, and the median field's gate is
symmetric (both sides need a real median). Measured on L1:42, the two conventions agree to 0.4%.
**Still quote `final_reeval_ms`**, not `tuned_ms` — see §4.3.

### 2.3 A crash you may have hit

`_best_profile` read `latency_ms.median` raw while every other selection reads `robust_ms`. Since
`median` is Optional, this killed a run at its first analyst step with
`TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'`. If your logs contain
that, this is the cause and it is fixed.

### 2.4 Rewrites are no longer discarded on a transport timeout

`rescue_from_sandbox` returns `None` in the base class and had to be overridden per module — so only
the rewriter had it, and `novelty`/`generator` silently threw away finished candidate files when a
call timed out. Measured: a novelty call wrote `nv_1.py` and was killed **9m15s later**, costing a
second full attempt. All three producers now rescue, and a rescued candidate is marked as such in
the lineage (`"[recovered from sandbox after a transport failure...]"`). Seeing that string in a
report is expected, not a bug.

### 2.5 A rewrite rejection is no longer logged as `NOVELTY_REJECTED`

It is now `REWRITE_REJECTED`. If you count `NOVELTY_REJECTED` to ask "did Loop D run", an older log
will mislead you — the rewrite path borrowed that event type, distinguished only by an
`origin: "rewrite"` payload field. `report.py` reads both names, so replaying an old log still shows
its rejections.

### 2.6 Tier 1 facts appear in every bottleneck verdict

`occupancy`, `occupancy_limiter`, `n_spills`, `n_regs` used to be recorded only inside the
`near_limit` block, which four of the five verdicts return before reaching. So a `compute_bound`
verdict advised "add more independent accumulators for ILP" to a kernel already at 255/255 registers
with 10 spills and 8% occupancy — advice that makes it strictly worse. They are now recorded before
any verdict can return.

---

## 3. Things that fail SILENTLY

These are ordered by how much of a run they waste.

### 3.1 The `device:` block in the config is hardcoded to a 4090

`linux-server/configs/experiments_l3_glm_linux.yaml` says:

```yaml
device:
  name: NVIDIA GeForce RTX 4090 (sm_89)
  vram_gb: 23
  max_regs_per_thread: 255
  max_shared_bytes_static: 49152
  max_shared_bytes_optin: 101376
  max_threads_per_block: 1024
```

`_device_doc()` (`agents/modules.py`) writes these **verbatim into every agent prompt**. If your box
is not a 4090, every agent is told the wrong hardware and will size tiles against limits that do not
exist — and nothing errors. This has already happened once: a config naming an "RTX 5080 Laptop
(sm_120)" was used on a 4090.

Measure yours and edit the block:

```bash
python - <<'EOF'
import torch
p = torch.cuda.get_device_properties(0)
print("name:", p.name, "| cap:", p.major, p.minor)
print("vram_gb:", round(p.total_memory / 1024**3, 2))
print("regs_per_sm:", p.regs_per_multiprocessor)
print("shared_per_sm:", p.shared_memory_per_multiprocessor)
print("max_threads_per_sm:", p.max_threads_per_multi_processor)
print("SMs:", p.multi_processor_count)
EOF
```

Note `max_shared_bytes_optin` is the per-BLOCK opt-in limit, which is not the per-SM figure torch
reports. On a 4090 it is 101376 against a per-SM 102400.

Also confirm the config is not silently falling back on defaults: `load_config` reads **one** YAML
file. `default.yaml` is **not** a base layer — an omitted key falls back to the field default, not
to `default.yaml`. That is how a whole run's agents were told the GPU was `"unknown"`.

### 3.2 `external_directory` must be `"allow"`

```bash
grep -n 'external_directory' src/kernel_optimizer/agents/sandbox.py
```

It must be present and `"allow"`. A key omitted from `PERMISSION_CONFIG` does **not** default to
allow — it falls through to opencode's own `{"*": "ask"}`, and a headless server has nobody to
answer. Measured: three dead agent calls, each idling the full timeout, ~90 minutes for zero output.

**Do not "fix" this to `"always"`.** A config comment used to say that; `"always"` was never tested.
The verified values are `"allow"` and `{"*": "allow"}` (10.5 s and 10.1 s against a control that hung
240 s).

### 3.3 The output-token ceiling is environment-only

`OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` must be set (the L3 configs set it to `131072` under
`opencode.server_env`). There is no config-file entry for it. Without it, glm-5.3 was truncated at
32000 tokens on its **first** L3 call and produced zero files. An L1 smoke will not reveal this — its
peak output was 5589 tokens.

### 3.4 Calibration is cached per box, and a cache can predate a measurement

The cache lives beside the runs (`<runs_dir>/calibration.json`) and is keyed on device identity, so
it will not silently cross a hardware change. But **do not copy one between boxes**: every threshold
in the classifier is a fraction of these measured ceilings.

```bash
python -m kernel_optimizer.cli --config <your config> calibrate    # or --recalibrate
```

Expect a `CALIBRATION_MEASURED` or `CALIBRATION_LOADED` event carrying `tiers`, `thresholds`,
`tf32_tflops`, `empty_launch_floor_ms`. If `thresholds` is absent the classifier falls back to
documented defaults and says so — usable, but your verdicts are then not calibrated to your box.

**The second staleness axis, which the identity key does not cover.** Every field on `Calibration`
has a permissive default so that an old cache still loads — necessary, since a box that cannot
measure bf16 must still classify — and the price is that a **newly added** measurement is served as
`0.0` forever on a box whose identity never changed, silently. This happened: the fp16/bf16 ceilings
landed 2026-09-08 and this box's `runs-l3/calibration.json` was written 09-07, so a fresh run
journalled `fp16_tflops: 0.0`. A low-precision candidate is then scored against the **tf32**
denominator — about half its real ceiling on a 4090 — and reads as saturated with headroom left.
That is the `impossible_fraction` defect that carried 13 of 25 verdicts on L3:43.

`load_cached` now refuses any cache below `CALIBRATION_SCHEMA_VERSION`, so this repairs itself on the
next run. Two obligations remain:

- **When you add a measurement to `run_calibrate`, bump `CALIBRATION_SCHEMA_VERSION`**
  (`evaluation/calibration.py`). Forgetting it puts the new field back to a permanent `0.0`.
- **Read the numbers, don't trust the word "cached".** On the first run after an update:

```bash
python - <<'EOF'
import json, pathlib, sys
run = sys.argv[1] if len(sys.argv) > 1 else "<run dir>"
for ln in pathlib.Path(run, "events.jsonl").read_text(encoding="utf-8").splitlines():
    e = json.loads(ln)
    if e["type"] in ("CALIBRATION_LOADED", "CALIBRATION_MEASURED"):
        p = e["payload"]
        print(e["type"], "source=", p.get("source"))
        for k in ("dram_tbs", "fp32_tflops", "tf32_tflops", "fp16_tflops", "bf16_tflops"):
            print(f"   {k:<14} {p.get(k)}")
EOF
```

`fp16_tflops` and `bf16_tflops` must be non-zero on any card that has tensor cores. A zero there is
not a harmless gap — it inverts the advice the agent receives on precisely the candidates fast
enough to lead.

---

## 4. Running an experiment

### 4.1 Preflight, every time

```bash
python -m pytest tests/ -q                # expect 376 passed, 2 skipped on Linux
python -m kernel_optimizer.cli --config <cfg> doctor          # includes the tier block
grep -n 'external_directory' src/kernel_optimizer/agents/sandbox.py
grep -n 'OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX' <cfg>
python -c "import sys; sys.path.insert(0,'src'); \
  from kernel_optimizer.gpu.worker_main import capture_timing_samples; print('median fix present')"
```

Note the **argument order**: `--config` comes BEFORE the subcommand.
`cli --config X run --task Y`, not `cli run --task Y --config X` (the latter errors out).

### 4.2 One run per GPU, always

Timing takes an exclusive lock. Two concurrent runs do not merely halve throughput, they
**contaminate each other's latency measurements**, which are the entire deliverable. If you want
several tasks in sequence, use `linux-server/scripts/run_l3_chain.sh` as a model — note it waits on
the **orchestrator process**, not on `nvidia-smi`: a run sitting between GPU jobs (mid agent call)
shows an idle GPU while still owning the experiment.

Also: the orchestrator has no SIGTERM handler, so killing it leaves orphans. Check for stray
`opencode` and worker processes before starting.

### 4.3 Reading the result

```bash
python scripts/analyze_l3_run.py <run_dir>
```

It answers, from `events.jsonl` only: how the run ended, `final_reeval_ms` vs `tuned_ms`, Loop C
rounds and per-family gains, whether Loop D fired, freezes, impossible fractions, median coverage,
failures/rescues, and whether each space expansion is merely an improvement or actually
*attributable*.

Rules that have each cost a wrong conclusion:

- **Quote `final_reeval_ms`, never `tuned_ms`.** Tuned is *usually* optimistic (1.5–6.7% on prior
  runs) but this is a tendency, not a law — on L1:42 the re-eval came out 2.65% **faster**. So the
  instruction is "use the re-eval figure", not "discount tuned by ~N%".
- **`stop_kind` in the summary can be `None`** while the `CONVERGENCE_DECIDED` stream has the real
  answer. Read the decision stream.
- **Only trust on-disk `events.jsonl`.** Notification content and report prose have both been wrong;
  a report has called an in-flight run "killed or crashed".
- **`frozen_converged` with `rewrite_rounds_used: 0`** means "we never looked", not "no headroom
  left". The analyzer flags this.
- **A rewrite's stated hypothesis is not evidence for why it worked.** Check whether the winning
  trial actually uses the value the hypothesis was about. On L3:43, a rewrite built to unlock
  `BLOCK_N=512` won by 41.7% — and its 512 configs were *slower*; the gain came from an unrelated
  side effect of the same change.

---

## 5. One known defect, deliberately unfixed

**Calibration has no fp16/bf16 arithmetic ceiling.** It measures fp32 and tf32 only
(`gpu/worker_main.py`, the `run_calibrate` matmul block). An fp16 or bf16 kernel is therefore scored
against tf32, and on a 4090 fp16 dense throughput is roughly 2x tf32. The consequence inverts the
advice:

| denominator | 95.95 TFLOP/s reads as | tells the agent |
|---|---|---|
| tf32 88.88 (current) | 107.8% | at the ceiling, stop |
| fp16 ~160 (1.8x tf32) | 60.0% | 40% still available |

It degrades loudly: `impossible_fraction` fires and the `disagreement` text tells the agent not to
read it as "at the ceiling", naming both possible causes. **If you see `pct_of_compute_peak > 100`,
this is why** — check the winning trial's dtype knob before concluding the candidate is saturated.

Not fixed yet because changing calibration shifts every derived threshold, which must not happen
between tasks of a comparison set. The fix, when taken: measure fp16/bf16 ceilings alongside fp32 and
tf32, add them to `Calibration`/`DevicePeaks`, and have `classify` pick the denominator from
`_candidate_precision` (which already distinguishes fp16/bf16/tf32/ieee_fp32 — only the ceiling is
missing).

---

## 6. Security

`.opencode/opencode.jsonc`, `opencode_backup.jsonc`, `kimi-provider.yaml`, the global
`~/.config/opencode/opencode.jsonc`, and **every GLM run sandbox's `opencode.json`** contain a
plaintext API key. `runs_dir` therefore multiplies it, and a `tar`/`scp` of a runs directory carries
the key off the box. Do not publish a runs directory, and treat the key as needing rotation rather
than assuming the directory is private.
