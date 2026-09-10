# A800 box: base configuration record

**Box**: `ssh a800` → `connect.nma1.seetacloud.com:46901`, key at `~/.ssh/a800_key` (key-only; the
one-time password was used solely to install the key and is not stored anywhere).

**Configured**: 2026-09-10. Verified state at that date, on this box, at repo commit `f2d4835`.

---

## 1. What this card is, measured (not from a datasheet)

| | 4090 (box 1/2) | **A800 80GB PCIe** | ratio |
|---|---|---|---|
| capability | sm_89 | **sm_80** | — |
| SMs | 128 | **108** | 0.84× |
| VRAM | 24 GiB | **79.3 GiB** | 3.3× |
| DRAM | 911–924 GB/s | **1686 GB/s** | **1.83×** |
| fp32 ieee | 54.8 | **19.0** | **0.35× (much SLOWER)** |
| tf32 | 88 | **111.8** | 1.27× |
| fp16 | 158 | **229** | 1.45× |
| bf16 | 164 | **234** | 1.43× |
| shared/block (optin) | 101376 B | **166912 B** | 1.65× |
| L2 | 72 MiB | **40 MiB** | **0.56× (SMALLER)** |

**Three of these invert conclusions that hold on the 4090**, so any threshold or advice derived on a
4090 must be re-derived here rather than carried over:

- **fp32 is 2.9× SLOWER while tf32 is 1.27× faster.** The ieee-vs-tensor-core tradeoff is far more
  lopsided here (tf32/fp32 = 5.9× versus 1.6× on the 4090), so a candidate locked to
  `input_precision="ieee"` gives up much more on this card.
- **L2 is SMALLER (40 vs 72 MiB) while DRAM is nearly twice as fast.** G7's dormant L2 defect wakes
  at a *different* working-set size here, and the ridge point moves.
- **Shared memory per block is 1.65× larger**, so tile sizes rejected as infeasible on the 4090 may
  compile here — the shared-memory feasibility screen is card-specific.

## 2. G10 confirmed by inversion — the strongest evidence the fix was right

Triton-vs-cuBLAS, every figure correctness-gated against a fp64 reference:

| 精度 | 4090 比值 | **A800 比值** |
|---|---|---|
| fp32 | 0.841 | 0.951 |
| tf32 | 0.982 | **1.032** |
| fp16 | **1.096** | **0.808** |
| bf16 | **1.079** | **0.809** |

**The pattern inverts between cards.** On the 4090 Triton beats cuBLAS at fp16/bf16 and loses at
fp32; on the A800 it loses badly at fp16/bf16 (80.8%, i.e. ~19% of the roof unreachable) and slightly
exceeds cuBLAS at tf32.

So the ratio is a property of the **(card, precision) pair**, not of Triton. Any hardcoded
per-backend ratio would have been wrong on one of the two boxes; `max(cuBLAS, Triton)` measured per
box is right on both. This is also the first real input for **G12** (heterogeneous comparison).

## 3. Software

| item | value |
|---|---|
| Python | 3.12.3 (`/root/miniconda3/bin/python3`) |
| venv | `/root/autodl-tmp/orch-venv` (`--system-site-packages`) |
| torch | 2.8.0+cu128, CUDA works on the A800 |
| triton | 3.4.0 |
| added | optuna 5.0.0, pydantic 2.13.5, pytest |
| node | 22.14.0 at `/root/autodl-tmp/nodejs` (tarball; conda-forge and nodejs.org unreachable, npmmirror worked) |
| opencode | 1.18.30 (npm global; binary is named `opencode.exe` even on Linux — that is normal for this package) |
| tmux | 3.2a (needed `apt-get update` first; the pre-existing lists did not have it) |
| repo | `/root/autodl-tmp/work/opop` @ `f2d4835`, branch v3 |
| KernelBench | `/root/autodl-tmp/work/KernelBench` @ `423217d` (same pin as the other boxes) |

**No token is stored on this box**: the clone used a one-time inline credential and `origin` was
immediately rewritten to the plain URL. Verified: no `ghp_` in `.git/config` or `~/.git-credentials`.

## 4. The PATH trap — fixed at its root

`/etc/profile:28` prepends `/root/miniconda3/bin`, but `/etc/profile` runs for **login shells only**,
and `~/.bashrc` returns early when non-interactive. So `ssh a800 python3` failed with
`command not found` while `ssh a800 'bash -lc python3'` worked — and any subprocess the harness
spawns (tmux panes, worker processes, `opencode`) saw no interpreter at all. This is the documented
cause of a bare `FileNotFoundError` that reads as a missing dependency rather than a missing PATH.

Fixed by prepending a guarded block to `~/.bashrc` **above** the non-interactive early-return, for
both miniconda and node (`~/.bashrc.bak-claude` holds the original). Verified on a fresh connection:
`command -v python3 node opencode` all resolve without `-lc`.

## 5. The Linux port is NOT in the v3 branch — read this before trusting a fresh clone

Three files exist as Linux variants that are **byte-identical to box 1's working tree but differ from
committed v3** (149 insertions uncommitted there):

    src/kernel_optimizer/cli.py                ← linux-server/cli.linux.py
    src/kernel_optimizer/agents/runtime.py     ← linux-server/runtime.linux.py
    src/kernel_optimizer/gpu/worker_client.py  ← linux-server/worker_client.linux.py

A fresh clone of v3 gets the **Windows/WSL** versions, which shell out through `wsl.exe` and
translate paths to `/mnt/d/...` — the harness cannot run at all. Copies of all three are installed
here and recorded under `linux-server/` as the scp-overwrite guard: **before scp'ing any of these
three from a Windows checkout, check `linux-server/` first** — a variant means the mainline file must
not be copied over.

**This is worth committing to the branch**; until then every new Linux box needs this manual step.

## 6. Test suite

**422 passed, 0 failed** (re-run after the G18 merge made `test_to_wsl_path` platform-aware; it
previously failed here by construction, since there is no WSL layer on a native-Linux box).

## 7. Configured and VERIFIED (2026-09-10)

### 7.1 API / provider — done, and proven by a real call

The provider config was transferred **box-to-box over a base64 pipe**, so the key never passed through
a terminal or a model context, and written `0600`. Integrity checked by hash, not by eye:
box 1 sha256 prefix `23907e0378463363` == A800 `23907e0378463363` → **identical**.
`npm install` in `/root/autodl-tmp/work/opop-glm` installed `@opencode-ai/plugin` 1.18.29.

Then the thing that actually matters — `scripts/verify_agent_env.py`, which drives the harness's own
runtime rather than checking that files exist:

```
sandbox providers: ['zhipuai']   ok: resolves from inside a sandbox
permission keys  : [bash, edit, external_directory, webfetch]
server up        : 1.3s        opencode 1.18.30
returned         : 9.8s  finish=tool-calls  cost=0.0023
structured       : {'answer': 4, 'gpu_name': 'NVIDIA A800 80GB PCIe'}
```

**The agent ran `nvidia-smi` itself and read this card back.** That exercises the two failure modes a
file-existence check cannot see:

- a sandbox's own `opencode.json` makes it a project root and **stops** opencode's upward config
  search, so a repo-local provider is invisible from inside a sandbox — the historical
  `ProviderModelNotFoundError: zhipuai/glm-5.3`;
- any permission key missing from `PERMISSION_CONFIG` falls back to `ask`, which is fatal headless
  (the turn idles until the ceiling aborts it).

### 7.2 L3 config — done, and cross-checked against the card

`configs/experiments_l3_glm_a800.yaml`. Every budget and evaluation knob is **copied verbatim** from
box 1's config, so a run here differs as a *machine swap* and nothing else — that is what makes the
two boxes usable for G12. What changed: paths, and the `device:` block.

Two box-specific facts are recorded in the file rather than left to be rediscovered:

| | box 1 (4090) | **box 3 (A800)** |
|---|---|---|
| venvs | two (`orch-venv` + `kernel-opt-venv`; the 30 G system disk cannot hold torch twice) | **one** `orch-venv` serves both roles |
| torch / triton | 2.9.1+cu129 / 3.5.1 | **2.8.0+cu128 / 3.4.0** |
| data disk | 250 G | **50 G** |

The toolchain skew is **not** cosmetic: Triton 3.4 and 3.5 generate different code, so register
counts, shared usage, and even whether a tile compiles can differ between the boxes for reasons that
are *not* the hardware. Any box-1-vs-box-3 comparison must treat the toolchain as a co-varying factor.

`scripts/validate_box_config.py` checks what a YAML parse cannot — every path exists, the venv really
imports torch+triton with working CUDA, and the device block matches what the card reports *now*:

```
name NVIDIA A800 80GB PCIe (sm_80) ~ NVIDIA A800 80GB PCIe   capability sm_80
vram_gb 79 ~ 79.3    shared_optin 166912    sms 108
READY
```

Both boxes pass it and it reads two different cards, so it discriminates rather than always passing.
**Negative control:** box 1's device block validated against the A800 is rejected with all four
mismatches named and exit 1. The first version caught only three — comparing `real_name.split()[0]` is
vacuous, since "NVIDIA" appears in every card name — so the name check now compares the distinctive
model tokens.

## 7.3 The gap every other check missed: the box could not evaluate a kernel

Verified working now, but recorded because it cost a full 6-call G9 re-run. This box passed **422
unit tests**, a **real agent call**, and `validate_box_config.py` **all green** — and still could not
evaluate a single kernel. `kernelbench`'s package `__init__` pulls a chain of its LLM-tooling
dependencies (`dotenv` → `openai` → `litellm` → …), unrelated to evaluation but required to import
the evaluator, and **25 packages were missing** against box 1's working venv. Every candidate came
back `runtime_error`, which reads identically to "the model wrote bad kernels".

Fixed by installing box 1's set, and guarded generally by `scripts/verify_box_can_evaluate.py`,
which evaluates a **known-correct** kernel through the harness's own `quick_test`:

```
candidate : /root/g9-start-l3-21/best.py    (the 3.6050 ms champion measured on the 4090)
ok        : True     latency : 5.4472 ms
VERDICT: READY
```

**Run that before spending agent calls on any new box.** See G29 in the gap register.

**A cross-card datapoint from it**: the same kernel is 3.6050 ms on the 4090 and **5.4472 ms** here
(1.51× slower), against the A800's 2.9× lower fp32 — so this task is not purely fp32-compute-bound.

## 8. What is still NOT done on this box

- **No calibration cached** through the harness's own `kernel-opt calibrate`; the ceilings in §1 come
  from probes. The first orchestrated run will measure and cache it (schema 3, so it will include the
  Triton-reachable ceilings).
- **No L3-scale agent call.** The verification call in §7.1 was trivial (91 input tokens). glm-5.3 at
  L3 prompt scale is where the 32000-token truncation appeared on box 1; neither the raised ceiling
  nor the truncation-specific feedback has been exercised here.
- ~~No GPU worker job has run through `wsl.venv`.~~ **DONE** — see §7.3: a known-correct kernel
  compiled, passed correctness and was timed at 5.4472 ms through the harness's own evaluator.
- **The relaxed + fp64 witness path has never executed here.** Memory is not the constraint on an
  80 GB card, but the code path is unrun.
- **Per-task noise floors are box-1 numbers.** The three ieee-vs-tf32 floors (0.9554 / 0.9767 /
  0.9778) are a *(card, task)* property and must be re-measured here before use.
- **API key rotation is still pending** (user-side). This box now holds the same plaintext key as
  box 1, which raises the count of machines it has been on.

