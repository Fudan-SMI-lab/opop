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

**398 passed, 1 failed.** The single failure is `test_worker_protocol.py::test_to_wsl_path`, which
asserts `D:\x\y` → `/mnt/d/x/y`. There is no WSL layer on a native-Linux box, so this test is
inapplicable by construction and fails identically on box 1. Not a defect of this box.

## 7. What is NOT configured yet

- **No opencode provider config / API key.** `~/.config/opencode/opencode.jsonc` does not exist here,
  so no agent call can run yet. This is deliberate: the key should be the **rotated** one, not the
  currently-shared plaintext key that has already been on two machines.
- **No calibration cached** through the harness's own `kernel-opt calibrate`; the ceilings above come
  from probes. The first orchestrated run will measure and cache it (schema 3, so it will include the
  Triton-reachable ceilings).
- **No L3 config for this box.** `configs/experiments_l3_glm_linux.yaml` on box 1 hardcodes box-1
  paths and a 4090 `device:` block; an A800 config needs its own paths and the measured limits above
  (`max_shared_bytes_optin: 166912`, 108 SMs, 40 MiB L2).
