# Running the KernelBench experiments on a fresh Linux GPU server

Target: a Linux box with one NVIDIA GPU (the immediate case is an RTX 4090, sm_89, 24 GB),
reached over SSH, running the GLM arm of this harness.

**Read this first: the harness does not currently run on Linux.** It was built with the
orchestrator on Windows dispatching GPU work into WSL2, and that split is wired into seven
places in the source. This document covers (A) the port, (B) the environment, (C) the
acceptance tests that must pass before an experiment is trustworthy, and (D) how to run and
read an experiment. Section A is real work — budget a day — and skipping any of C means you
will not be able to tell a bad result from a good one.

Everything below was verified by reading the code at commit `fa3e7ba`; each claim carries a
`file:line`. Where a value must be measured on the actual hardware rather than copied from
here, it says so.

---

## A. Port the harness to native Linux

On Linux the Windows/WSL split collapses: the orchestrator and the GPU worker are the same
OS, so the `wsl.exe` wrapper, the drive-letter translation, and `taskkill` all go away.

### A1. `to_wsl_path` must become the identity — `src/kernel_optimizer/gpu/worker_client.py:18-23`

```python
def to_wsl_path(p: Path | str) -> str:
    p = Path(p).resolve()
    drive = p.drive.rstrip(":").lower()
    rest = p.as_posix().split(":", 1)[1]     # IndexError on Linux
    return f"/mnt/{drive}{rest}"
```

On a Linux path there is no `":"`, so `split(":", 1)[1]` raises `IndexError`. `Path.drive`
is `''` too. Callers: `:129`, `:130`, `:150`. On Linux this must return
`str(Path(p).resolve())` unchanged.

### A2. The GPU dispatch must not go through `wsl.exe` — `worker_client.py:159-166`, `:181-189`

`["wsl.exe", "-d", distro, "bash", "-lc", cmd]` is the single dispatch path for **all** GPU
work — static check, correctness, baseline, perf, env probe. It raises `FileNotFoundError`
on Linux, i.e. the entire evaluation half of the harness.

Three shell-only mechanisms in `_build_command` (`:118-131`) are load-bearing and must be
replaced explicitly, not just dropped:

| mechanism | why it works today | Linux replacement |
|---|---|---|
| `VAR=x cmd` env prefix | only a shell does this | `env={**os.environ, "TRITON_CACHE_DIR": …, "PYTHONPATH": …}` |
| `~` in `venv` / `triton_cache_dir` (`config.py:223,229`) | `bash -lc` expands it | `os.path.expanduser` — Python never does |
| `-l` login shell | sources the profile | make the interpreter path absolute |

Miss the `~` expansion and every job dies as `worker_crash` on a literal `./~/…` path.
Also note `_build_command` does no shell quoting: a run directory containing a space breaks
it today and would break a naive port too. Prefer a list argv, which sidesteps quoting
entirely.

### A3. Fix the timeout kill while you are here — `worker_client.py:181-189`

`_kill_workers` runs `pkill -f worker_main.py`, which kills **every** worker on the box.
With `gpu.concurrency.enabled: true` and `max_shared_jobs: 2` (`config.py:209-210`) a
timeout in one shared-lane job kills the other, healthy job, which then reports
`worker_crash` ("no result file", `:170-175`). This is a live bug on WSL today, not a Linux
regression, and the port is the natural moment to fix it: spawn each job with
`start_new_session=True` and kill only `os.killpg(os.getpgid(proc.pid), SIGKILL)`.

### A4. `shell=True` must be removed — `src/kernel_optimizer/agents/runtime.py:73`, `:105`; `src/kernel_optimizer/cli.py:57`

This is the most dangerous item because **it does not raise — it misbehaves silently.**

```python
Popen(["opencode", "serve", "--hostname", h, "--port", p], shell=True, ...)
```

On POSIX, `shell=True` with a list argv runs `/bin/sh -c "opencode"` and passes the rest as
`$0, $1, …` to the shell — **the arguments are discarded.** The server launches with no
`--port`, binds its default, and `_wait_healthy` (`:79`) then times out against a URL
nothing is listening on. It reads exactly like a network problem. The comment at `:73`
states the Windows-only reason (`opencode is a .cmd shim on Windows`).

Note `tests/test_improvements.py:1136-1168` monkeypatches `Popen` and never inspects argv,
so **no test catches this.**

### A5. Server shutdown needs a process-group kill — `runtime.py:110-123`

`taskkill /F /T /PID` raises `OSError` on Linux, which *is* caught at `:118` falling back to
`self.proc.kill()`. But that kills only the direct child — under `shell=True` that is
`/bin/sh`, leaving the real server orphaned: a leaked process and port per run. Use
`start_new_session=True` at spawn, then `SIGTERM` the group, `wait(timeout=…)`, escalate to
`SIGKILL`. Prefer graceful `SIGTERM` first: opencode holds a sqlite session store and
`SIGKILL` can leave it mid-write.

### A6. `doctor`'s WSL probe — `cli.py:49-50`

`["wsl.exe", "-l", "-q"]` decoded as `utf-16-le`. Replace with a native GPU check
(`nvidia-smi`, or the env-probe job) or drop it.

### A7. Retire three tests that assert Windows/WSL-ness

They *should* fail after the port — repairing them would mean re-asserting the thing you
removed:

- `tests/test_worker_protocol.py:16-18` `test_to_wsl_path` — delete, or invert to assert identity.
- `tests/test_improvements.py:634-645` `test_wsl_paths_are_on_ext4_not_9p` — `:645` requires
  `kernelbench_src.startswith("/mnt/")`. The whole 9p/ext4 distinction dissolves on native
  Linux, and with it the harness's largest documented cost: **26.8 s → 2.4 s of per-worker
  startup, about 6.7 h of an 11.7 h L3 run** (`configs/default.yaml:69`). That is the single
  biggest win of this migration.
- `tests/test_improvements.py:648-651` `test_setup_script_refuses_a_9p_venv` — asserts on
  the text of `scripts/setup_wsl_venv.sh`.

Run pytest **from `v2/`**: about 28 sites read `src/…` and `configs/…` by relative path
(e.g. `test_improvements.py:2026,2049`), so a different working directory fails them en
masse in a way that looks like a port bug but is not.

### A8. Two coverage gaps to know about

No test anywhere references `taskkill`, mocks `wsl.exe`, or asserts on `_build_command`
output. **Nothing will fail when you rewrite the dispatch and kill paths, and nothing will
confirm you got them right.** Write the acceptance checks in section C before trusting the
port.

---

## B. Environment setup

### B1. Get the code

```bash
git clone -b v2 https://github.com/Fudan-SMI-lab/opop.git
cd opop/v2
```

The repo contains `src/ tests/ scripts/ configs/ docs/`. It deliberately does **not**
contain `runs/` (experiment output), `.opencode/` (provider config with a plaintext API
key), or the KernelBench checkout.

### B2. Pin KernelBench

The harness reads reference problems from a KernelBench checkout it does not vendor.

```bash
git clone https://github.com/ScalingIntelligence/KernelBench.git
cd KernelBench && git checkout 423217d && cd ..
```

`423217d` is the pin all existing results were measured against. Commit `b08d959` ("scaled
up the tensor sizes for level 1, 2 and 3, so that all problems run above 1ms") changed
problem sizes drastically, so results across that boundary are **not** comparable — see B6.

### B3. Python environment

`scripts/setup_wsl_venv.sh` is WSL-specific but its dependency choices carry over:

```bash
python3 -m venv ~/kernel-opt-venv
source ~/kernel-opt-venv/bin/activate
pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu129
pip install "triton==3.5.*" numpy
```

cu129 wheels cover sm_89 (Ada predates Blackwell), so the version pin is fine on a 4090.
The orchestrator side uses `uv` (`pyproject.toml`); `uv sync` in `v2/`.

Keep the venv and the triton cache on **local disk**. That was forced on WSL for the 9p
reason above; on a native server it is simply normal, but avoid NFS for the same reason.

### B4. opencode and the GLM provider

The harness spawns `opencode serve` itself (`runtime.py:56-77`) and talks to it over HTTP.
Install opencode, then create the provider config. **This file holds an API key in
plaintext** — keep it out of git (`.gitignore` covers `*.jsonc`, `opencode.json`).

`<GLM_ROOT>/.opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "zhipuai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Zhipu AI",
      "options": {
        "baseURL": "https://open.bigmodel.cn/api/coding/paas/v4",
        "apiKey": "<YOUR_KEY>"
      },
      "models": {
        "glm-5.3": {
          "family": "glm", "reasoning": true, "temperature": true, "tool_call": true,
          "options": { "reasoningEffort": "max" }
        }
      }
    }
  }
}
```

**Why the directory layout matters, and it is not obvious.** opencode resolves providers by
walking **up** from the session's working directory. Agent sessions run with
`directory=<sandbox>`, and each sandbox gets its own `opencode.json`, which makes it a
project root and **stops** the upward search. So:

- `launch_cwd` and `runs_dir` must both sit under the tree containing `.opencode/`.
- `sandbox_config_path` must point at the provider file so each sandbox gets a copy.

Get this wrong and every call fails `ProviderModelNotFoundError: Model not found:
zhipuai/glm-5.3` — verified live. On the Windows box that tree is `v2-glm/`; replicate the
shape, e.g. `~/opop-glm/.opencode/opencode.jsonc`.

**Set the output-token ceiling.** opencode computes the per-turn cap as
`min(model.limit.output, OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX ?? 32000)`, and there is no
config-file key for it — the environment variable is the only route. At the 32000 default,
glm-5.3 at `reasoningEffort: max` spent the entire budget on reasoning and wrote **zero
files** on three consecutive attempts, killing a whole run
(`run-l3-21-20260906-084636`). The configs pass it via `opencode.server_env`
(`config.py:52` → `runtime.py:67`); 131072 is zhipuai's advertised maximum (more returns
HTTP 400 code 1210).

### B5. Write a server config — do not reuse the Windows ones

`load_config` reads **exactly one file**; `configs/default.yaml` is *not* a base layer
(`config.py:257-268`). Any key you omit falls back to the pydantic field default, and those
defaults are Windows paths. Five existing configs (`experiments_l3.yaml`,
`smoke_l1*.yaml`) set no paths at all and would silently inherit `D:/…` and `/mnt/d/…`.

Copy `configs/experiments_l2_37_glm.yaml` and change every path plus the whole `device:`
block:

```yaml
kernelbench_root: /home/<user>/opop/KernelBench      # config.py:245 default is D:/...
run:
  runs_dir: /home/<user>/opop-glm/runs-l2-37
opencode:
  launch_cwd: /home/<user>/opop-glm                   # config.py:21 default is D:/...
  sandbox_config_path: /home/<user>/opop-glm/.opencode/opencode.jsonc
  server_env:
    OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX: "131072"
wsl:                       # class name is now a misnomer; the keys still apply
  venv: ~/kernel-opt-venv
  kernelbench_src: /home/<user>/opop/KernelBench/src  # config.py:225 default is /mnt/d/...
  triton_cache_dir: ~/.triton-cache-kopt
device:
  name: RTX 4090 (sm_89)          # MEASURE the rest -- see C3
  vram_gb: 24                     # NOT the 16 in every shipped config
  max_regs_per_thread: 255
  max_threads_per_block: 1024
  max_shared_bytes_static: 49152
  max_shared_bytes_optin: <MEASURED>
```

`kernelbench_root` selects which reference **problem file** is loaded;
`wsl.kernelbench_src` is the evaluation **library** on `PYTHONPATH`. They are independent —
that is what lets you swap one problem's size without touching the eval code.

### B6. Reproducing the level2:37 comparison specifically

The cross-framework comparison on `level2/37_Matmul_Swish_Sum_GroupNorm` uses the
**pre-scaling** sizes (`batch=128, in=512, out=1024, groups=32`, `get_inputs` →
`torch.randn`), confirmed directly by the other team. At the sizes KernelBench ships today
(`32768/1024/4096/64`) the same problem is 1332x slower on our hardware, and their reported
90.30 µs baseline is not physically reachable there — it would need 7.43 TB/s of DRAM
bandwidth.

Build a separate root: copy the pinned tree, then replace exactly that one file from
`git show bb27f27:KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py`. Verify with
`diff -rq` that nothing else differs, and point `kernelbench_root` at it. Expect
`TaskSpec.ref_src_sha` to start `e8946e79d591ca80` (the pinned file is `fe2e8dc5418f1549`
after LF-normalization — `KernelBenchAdapter.load` reads in text mode, so CRLF files hash
as LF). Keep the full tree rather than a lone file so `doctor`'s KernelBench checks still
pass and every other problem resolves to the pinned version.

---

## C. Acceptance tests — run all of these before trusting a result

Ordered so each one's failure is unambiguous. **A negative result is only meaningful if the
check would have failed loudly had the thing been broken** — several of these exist
specifically because a silent pass fooled us before.

### C1. Unit tests

```bash
cd v2 && uv run pytest tests/ -q      # from v2/, see A7
```
Expect the three WSL tests of A7 to fail and everything else to pass (255 pass / 9 skip on
the Windows box at `fa3e7ba`).

### C2. Real GPU probe — not just `torch.cuda.is_available()`

A kernel must actually **launch**. `scripts/setup_wsl_venv.sh:99-127` has an
architecture-neutral triton launch probe; run its equivalent:

```bash
python -c "
import torch, triton, triton.language as tl
print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0),
      torch.cuda.get_device_capability(0))
@triton.jit
def k(x, y, n, B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0,B); m = i < n
    tl.store(y+i, tl.load(x+i, mask=m)*2.0, mask=m)
x = torch.arange(1024, device='cuda', dtype=torch.float32); y = torch.empty_like(x)
k[(1,)](x, y, 1024, B=1024); torch.cuda.synchronize()
assert torch.allclose(y, x*2), 'kernel ran but produced wrong values'
print('triton launch OK')"
```

`get_device_capability` must report `(8, 9)` on a 4090. If it says `(12, 0)` you are on the
wrong machine; if the assert fires, the toolchain is broken in a way `is_available()` will
never tell you.

### C3. Measure the device limits — do not copy them

Every shipped config claims `RTX 5080 Laptop (sm_120)`, `vram_gb: 16`,
`max_shared_bytes_optin: 101376`. Those numbers are written **verbatim into every agent
sandbox** by `_device_doc` (`src/kernel_optimizer/agents/modules.py:46-55`) and are exposed
as names (`MAX_SHARED_BYTES_OPTIN`, …) inside agent-authored constraint expressions via
`DeviceLimits.as_env()` (`models/core.py:180-187`).

So a stale value does two kinds of damage: the guard admits configurations that then fail
to compile with `OutOfResources`, and **the model is actively told it is on Blackwell** and
will emit tensor-core paths that do not exist on sm_89.

```bash
python -c "
import torch
p = torch.cuda.get_device_properties(0)
print('name                  ', p.name)
print('total_gb              ', round(p.total_memory/2**30, 1))
print('sm                    ', (p.major, p.minor))
print('shared_per_block      ', p.shared_memory_per_block)          # -> max_shared_bytes_static
print('shared_per_block_optin', getattr(p, 'shared_memory_per_block_optin', 'MISSING'))
print('threads_per_block     ', p.max_threads_per_multi_processor)  # per-SM, NOT per-block
print('regs_per_sm           ', p.regs_per_multiprocessor)          # per-SM, NOT per-thread
"
```

Verified on the Windows box's WSL (torch 2.9): this prints `shared_per_block 49152` and
`shared_per_block_optin 101376` for the sm_120 laptop card, so the property names are real
and `max_shared_bytes_optin` can be read straight off the device — **measure it on the 4090
rather than reusing 101376.**

Two of the six config fields are *not* directly available and must not be copied from this
output:

- `max_regs_per_thread` — `regs_per_multiprocessor` is per-SM (65536), not the per-thread
  limit. 255 is the correct value on every CUDA architecture in use here; keep it.
- `max_threads_per_block` — there is no such property (`max_threads_per_block` is
  `MISSING`; `max_threads_per_multi_processor` is a different quantity). 1024 is correct on
  both sm_89 and sm_120; keep it.

Put the measured numbers in the config. `tests/test_improvements.py:2003-2038` cross-checks
`device:` blocks between configs for drift, so it will fail loudly if you update one and
forget another — that test is an ally here.

Note `gpu.concurrency.vram_budget_frac` (`config.py:211`) is **declared but never read** —
no reader exists anywhere in `src/`. Going 16 GB → 24 GB does *not* automatically widen
concurrency; `max_shared_jobs` is the only knob that does.

### C4. Baseline sanity, against physics

```bash
uv run kernel-opt --config configs/<yours>.yaml baseline --task level2:37
```

Compute the memory-bandwidth floor for the task's traffic and check the measured baseline is
above it. A "baseline" below the floor means the timing is wrong, not that the hardware is
fast. On the 4090, `(batch*in + batch*out)*4` bytes ÷ ~1.008 TB/s is the floor.

### C5. Timing-noise floor — this decides whether tuning means anything

The tuning objective is the **mean** of `perf_trials` samples (`tuning/tpe.py:85`), and
`min_improvement_pct` defaults to 2.0. If the measurement's standard error is comparable to
2%, the search chases jitter.

Measure it before drawing conclusions (`scripts/probe_l2_37_noise.py` is the pattern):
time the reference ~1000 times, report min / median / p90 / max and a trimmed std. On the
Windows box at the old L2:37 sizes this gave a **22.2% trimmed std** and only 29.10 µs of
summed device time against a 64.96 µs wall median — **55% of wall time was launch overhead,
not kernel execution.** A 4090 on a server (no laptop thermal throttling, no WSL) should be
quieter, but verify rather than assume. If the trimmed std over `sqrt(perf_trials)` is near
`min_improvement_pct`, raise `perf_trials` or the threshold — and say so in the writeup.

### C6. Correctness gate floor — measure the reference against itself

The relaxed gate demands `relaxed_pass_frac: 0.99` of elements within `relaxed_elem_tol`,
but on all three L3 tasks the reference's **own** ieee-vs-tf32 agreement is below that
(0.9554 / 0.9767 / 0.9778), which makes the absolute gate unreachable for a correct
low-precision candidate. That is why `fp64_relative_gate: true` exists (accept when the
candidate's RMSE against an fp64 golden is within a multiplier of the *reference's* own
RMSE). Measure the floor for your task on your hardware before believing a rejection;
`docs/research-tolerance-practice-under-reference-noise.md` has the reasoning.

### C7. One live agent call

```bash
uv run kernel-opt --config configs/<yours>.yaml agent-smoke --module generator --task level1:19
```

Must produce at least one candidate file. This is where a provider-resolution mistake (B4)
or a token-ceiling mistake surfaces — cheaply, on L1, instead of hours into an L3 run.

### C8. A full cheap end-to-end run

```bash
uv run kernel-opt --config configs/smoke_l1_loopc.yaml run --task level1:19   # paths fixed first
```

`smoke_l1_loopc.yaml` is built to exercise Loop C (rewrite) and the convergence verdict;
`smoke_l1_novelty.yaml` exercises Loop D (novelty). Between them all five agent modules run.
Expect `RUN_FINISHED`, `OUTER_LOOP_STUCK = 0`, and at least one
`CONVERGENCE_DECIDED(stop_kind="converged")` on the Loop C config.

### C9. Concurrency cross-talk

With `gpu.concurrency.enabled: true`, confirm that a correctness verdict reached while a
second shared-lane job runs matches the serial verdict, and that exclusive timing
afterwards has normal variance. Timing is supposed to hold the GPU exclusively
(`worker_client.py:26-33`).

**But note a real limitation:** the GPU lock is created at `jobs_dir / "gpu.lock"` where
`jobs_dir = store.run_dir / "jobs"` (`wiring.py:103`, `worker_client.py:116`) — it is
**per-run, not machine-wide**. Two concurrent runs on one GPU do not see each other's lock,
so "exclusive timing" is only exclusive within a run and both would time simultaneously.
**Run one experiment at a time per GPU**, or move the lock to a fixed path as part of the
port. Anything else silently corrupts both runs' numbers.

---

## D. Running an experiment

### D1. Launch

```bash
cd v2 && source ~/kernel-opt-venv/bin/activate
nohup uv run kernel-opt --config configs/<yours>.yaml run --task level2:37 \
      > ~/opop-glm/l2-37-glm.log 2>&1 &
```

Use `tmux`/`screen` over SSH so a dropped connection does not kill a 12-hour run. Confirm
within a minute that `<runs_dir>/run-*/events.jsonl` exists and grows.

### D2. Monitor

```bash
python scripts/watch_run.py <runs_dir>/run-l2-37-<ts> --idle-min 60
```

Prints milestones and failures, exits 0 on `RUN_FINISHED`, exits 1 on stall or dead
process. **One Linux fix needed first:** its liveness check uses Windows `tasklist`
(`scripts/watch_run.py:29-35`) and on Linux the `OSError` is caught as "assume alive", so
`PROCESS GONE` can never fire. Replace with `pgrep -f opencode`. Left as-is, the monitor
still reports events and stalls, but silently loses death detection — which is exactly the
class of blind spot to avoid.

GLM agent calls are slow (a single L3 generator call measured 18m16s), so quiet stretches
are normal; that is why the stall threshold is 60 minutes.

### D3. Read the result — and where the numbers lie

```bash
uv run kernel-opt report --run <runs_dir>/run-<id>    # regenerates purely from events.jsonl
```

Then, in order:

1. **`ended:` and `rewrite rounds spent:`** near the convergence section. A run that used
   <60% of its clock did not converge — a freeze rule ended it, and the report says so.
   Only 1 of 19 historical runs was actually ended by the wall clock.
2. **`final_reeval_ms`, not `tuned_ms`.** `tuned_ms` is systematically optimistic by
   1.5–6.7% because the winning configuration is re-timed in a fresh process. Any claim of
   beating a baseline must cite the re-eval number.
3. **The attribution line** — which kernels the winning configuration actually launched. A
   candidate may hand part of the computation back to PyTorch and still win; that is a real
   result but not the result it appears to be.
4. **`excessive_speedup` flags.** Above 10x vs the in-job reference the run flags but does
   not reject a *correct* candidate (`worker_main.py:906-944`) — correctness decides
   acceptance, speed only annotates. Scrutinize flagged winners.
5. **Trust `events.jsonl` on disk over anything else**, including console output.

### D4. Known GLM-arm behaviours

- **Nested fields returned as JSON text.** glm-5.3 sometimes emits
  `{"space": "{\"domains\": …}"}` where a nested object is required. Decoded generically in
  `AgentModule.invoke` as of `fa3e7ba`; on older code this discarded seed candidates at
  ~$0.04 each with three identical retries. If you see
  `Input should be a valid dictionary` in `SPACE_REJECTED`, you are running pre-`fa3e7ba`.
- **Reasoning-budget truncation.** See B4 — set the token ceiling.
- **Fixes reach a running experiment only on the worker side.** Worker/agent-side changes
  take effect on the next job of a *running* experiment; driver-side changes (orchestrator,
  `base.py`, report) require a restart, because the driver imported the old module at
  launch. Do not expect a driver-side fix to rescue a run in flight.

---

## E. Security

- The provider config (`opencode.jsonc`) holds a plaintext API key. `.gitignore` covers
  `*.jsonc` / `opencode.json`, but note that **`sandbox_config_path` copies the whole
  provider block — key included — into every agent sandbox** under `runs/`. Those run
  directories are secret-bearing: never archive them without stripping, and note the copies
  are `.json`, which the `*.jsonc` rule does not match.
- `runs/` is excluded from git for that reason as well as size.
- Rotate any key that has been on a shared machine or in a shell history.
