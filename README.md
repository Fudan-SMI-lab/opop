# kernel-optimizer (v2)

Agent harness for GPU kernel structure search guided by parameter-tuning
feedback. Coding agents (via an opencode server) generate, parameterize,
analyze, and rewrite kernel candidates; the deterministic harness owns
correctness, timing, budgets, convergence, and reporting.

## Architecture

- **Host (Windows)**: orchestrator, opencode client, Optuna TPE tuning, event
  log. No torch on the host.
- **WSL2 (Ubuntu)**: all GPU work as one-shot subprocesses (`gpu/worker_main.py`)
  using the pinned local KernelBench checkout. One job = one process = clean VRAM.
- **GPU lanes**: correctness/compile/static-check jobs may run 2-way concurrent
  (shared lane); **all timing is exclusive** (whole GPU, serialized).
- Every LLM step is a real opencode agent session in a per-call sandbox with
  file-edit/bash freedom, returning schema-validated JSON.

## Setup

```powershell
# host
cd v2
uv sync --extra test

# WSL GPU venv (one-time; ends with a real triton launch probe)
wsl.exe -d Ubuntu -- bash -lc "bash /mnt/d/Pyhon_projects/opop/v2/scripts/setup_wsl_venv.sh"

# health check
uv run kernel-opt doctor
```

## Commands

```powershell
uv run kernel-opt baseline --task level1:19
uv run kernel-opt tune-file --task level1:19 --candidate examples/relu_triton.py --space examples/relu_space.json --trials 8
uv run kernel-opt agent-smoke --module generator --task level1:19
uv run kernel-opt run --task level1:19 --config configs/smoke_l1.yaml
uv run kernel-opt run --task level3:21 --config configs/experiments_l3.yaml
uv run kernel-opt resume --run runs/<id>
uv run kernel-opt report --run runs/<id>
```

## Module map (all replaceable via `ports.py` Protocols; wired in `wiring.py`)

| Requirement | Module |
|---|---|
| 1 candidate generation | `agents/modules.py:CandidateGeneratorAgent` |
| 2 parameterization | `agents/modules.py:ParameterizerAgent` + `paramspace/validation.py` |
| 3 correctness + tuning ceiling | `evaluation/correctness.py` + `tuning/tpe.py` |
| 4 profile records + Bayesian search | `evaluation/profilerx.py` + Optuna TPE |
| 5 blocked-parameter analysis | `tuning/stats.py` (deterministic) + `BottleneckAnalystAgent` |
| 6 structure rewrite | `agents/modules.py:StructureRewriterAgent` (loop C) |
| 7 convergence | `control/convergence.py` (harness-owned) |
| 8 novel family seeds | `NoveltyGeneratorAgent` + `control/families.py` gate (loop D) |
| 9 modularity | `ports.py` + `wiring.py` |

Candidate contract: one `PARAMS = {...}` literal dict; the harness tunes by
rewriting only that span (`paramspace/materializer.py`). See
`src/kernel_optimizer/agents/prompts/candidate_contract.md`.
