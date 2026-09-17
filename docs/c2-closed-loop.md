# T5: exactly two G0/G2 rounds

This entry composes unchanged `acquire` and `run_information` (legacy agent chain
and native retune). It implements **T5 only**, not a 12-hour search, Families-D
loop, or automatic T6 decision. See `result-c2-information-stage.md` for the T4 gate.

## Inputs and commands

For each host/task, use the **same original P1 Shared file** for G0 and G2. Do not
use a selected child from the completed information stage. The runner starts from
exactly the supplied Shared; it does not discover or choose another starting point.
Use the original resolved AppConfig and quality settings (100 full perf samples).
Sampler and evaluation seeds are fixed at 0. Each output directory must be new.

```sh
export V5=/absolute/deployed/v5
export PYTHONPATH="$V5/src:$V5"
export PY=/root/autodl-tmp/orch-venv/bin/python
export CFG=/absolute/inputs/resolved-config.yaml
export P1_SHARED=/absolute/inputs/original-P1/shared.json
export RUNS=/root/autodl-tmp/c2-local-experiment/runs

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_closed_loop \
  --config "$CFG" --shared "$P1_SHARED" --group G0 --run-dir "$RUNS/T5-P1-G0"
CUDA_VISIBLE_DEVICES=1 "$PY" -m scripts.experiments.c2_closed_loop \
  --config "$CFG" --shared "$P1_SHARED" --group G2 --run-dir "$RUNS/T5-P1-G2"
```

These run sequentially as written; the operator may launch one on each GPU.
Repeat on the other task/host: four jobs, at most eight child opportunities and
320 native B40 records. No SSH, deployment, or experiment execution is performed
by the implementation task. No extra attempts are made after the planned two.

## Round behavior

- G0 never calls acquisition. Its required response envelope explicitly contains
  unmeasured axes: no endpoint observations/J, zero calls/cost, `not_acquired`.
- G2 calls the existing acquisition with budget 12 in a new round-local root.
  Actual counts may be lower if the native space/probe rules admit fewer endpoints;
  no observations are fabricated. Round two probes the actual current source/params.
- Each round calls `run_information` unchanged. All ordinary parent information
  remains available; probes never enter tuning history or parent selection.
- A tuning-selected child is promoted only after three complete 100-sample final
  blocks. The new parent uses the selected materialized artifact, best valid native
  trial matching selected params, native published space, backend, and that
  candidate's full tuning history. Reference/task/quality/semantics/device stay fixed.
- Native retune events retain exact original records and witness-reuse tags. For
  round-two agent context, tagged reused trials gain a provenance-only
  `[reused_measurement=true]` note in the existing history detail column. Their
  params/status/latency/profile do not change; the parent record itself remains the
  exact native trial. No cross-child TPE warm start occurs.
- Parent selection, generation failure, witness rejection, or no-best keeps the
  current state and spends the attempt; the second planned opportunity still runs.
- Invalid/missing parent full measurements or invalid selected-child finals are
  terminal integrity failures, not fallback. The failed round remains recorded,
  has no ready timestamp, and earns no additional opportunity. No final reranking.

## Timeline and offline comparison

The single monotonic elapsed clock begins before first-round preparation and
acquisition. It includes acquisition, generation/retries, validation, screening,
tuning, full evaluations, failures, and intervening work; it is never reset.

`result.json` contains:
- `elapsed_total_s`, `terminal_error`, and the initial parent artifact;
- `initial_parent_finals`: an immutable copy of round one's parent full blocks for
  normalization, never replaced by round-two parent measurements;
- `rounds`: `elapsed_started_s`, `elapsed_ready_s`, incumbent native trial, artifact
  path, three fresh full records, error, and relative result/response paths.

`round-N/shared.json` records that round's actual input state. The unchanged
`round-N/opportunity/result.json` and its parent/generation/retune events/jobs retain
all outcomes and costs. G2 acquisition events/jobs are separate under
`round-N/acquisition/`; do not add their component durations twice to total elapsed.
Provider-reported cost omissions remain omissions; elapsed time still includes them.

Ready records are appended; later remeasurement never rewrites earlier ready times
or metrics. A nonterminal failed opportunity can produce a ready retained parent.
Exit 1 means terminal integrity/infrastructure failure; exit 0 means the planned
two rounds completed, **not** that both children succeeded or improved performance.

Offline analysis—not this runner—sets `H=min(t_G0,t_G2)`, uses the latest fully ready
incumbent at or before H, and normalizes to each run's initial fresh parent reference.
The initial source and reference blocks are stored separately; the runner does not
assign retrospectively measured blocks an invented ready timestamp at zero. Raw
parent events remain available if the offline rule needs reference measurement times.
Integrity failures must not be silently dropped. No normalization/gate engine or
statistical-significance claim is implemented here.

```powershell
.\.venv-v5\Scripts\python.exe -m pytest tests/test_c2_closed_loop.py -q
```

Tests replace only external provider/server and GPU worker boundaries; they are
CPU software verification, not new GPU evidence. Existing agent permissions/tools
are unchanged, so harness G0 acquisition=0 is not a claim of OS-enforced tool isolation.
