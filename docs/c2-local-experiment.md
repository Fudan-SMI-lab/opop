# Fixed-parent C2 single-step diagnostic

This driver uses **this v5 source tree**, not the old pipeline's `conditional_scan`
toggle. A receives ordinary pre-cutoff evidence; B receives the same evidence plus
one baked native `TaskResponse` acquisition. No E enqueue, parent B40 retuning,
outer rewrite loop, expansion, repair campaign, or final-measurement reranking.
This is an information-increment experiment, **not untreated-package vs treated-package**
and not a net-benefit or significance test. CPU tests are not GPU evidence.

## Operator inputs

Provide one absolute-path selection JSON per task (values below are examples, not
a task whitelist). Select the parent before observing any new child results.
Prefer a parent from an off run. A parent from active/observe history must use
`conditional_on_treated_state`, even though scan trials are excluded from common history.

```json
{
  "task": "level3:43",
  "historical_run": "/absolute/read-only/history/run-id",
  "parent_id": "cand-EXPLICIT",
  "parent_trial_id": "tr-EXPLICIT",
  "parent_source": "/absolute/read-only/history/run-id/candidates/cand-EXPLICIT/trials/tr-EXPLICIT.py",
  "space": "/absolute/inputs/parent-space.json",
  "cutoff_seq": 545,
  "state": "off",
  "semantics": "/absolute/inputs/semantics.json",
  "backend": "triton"
}
```

`cutoff_seq` is **exclusive**; use the true historical decision/call STARTED seq,
not the output's PRODUCED seq. The explicit trial must be successful, ordinary,
pre-cutoff, and its materialized source's PARAMS must match its recorded params.
The source must match the historical trial artifact. Preparation refuses scan parents;
choose an ordinary parent even for a conditional-on-treated-state comparison.
Historical runs are read only. Do not supply a C2-informed analyst report.

`parent-space.json` uses existing `TaskSpace` JSON:

```json
{"params":[{"name":"BLOCK_SIZE","kind":"int","choices":[64,128,256],"description":"original domain"}],"constraints":[]}
```

Supply the **entire original published space**, not a narrowed range around the winner.
Preparation checks domains and constraints against its pre-cutoff `SPACE_PUBLISHED`
event. It also checks the manifest task/evaluation and, when recorded, semantics/backend.
`semantics.json` contains the pre-decision reference evaluation semantics, for example
`{"training":true,"norm_layers":[]}`. Preserve actual facts and explicit missingness;
do not infer semantics from an analyst's prose. The configured KernelBench adapter
supplies the actual task reference. No arbitrary context/config is sent to direct agents.

The complete machine-readable selection schema can be exported without any runtime:

```sh
"$PY" -m scripts.experiments.c2_local_runner schema --output /absolute/inputs/selection.schema.json
```

The operator must also supply: existing v5 YAML config; provider config path already
read by v5 wiring; correct KernelBench root/src; chosen worker venv; a free GPU per
concurrent job; absolute new input/run roots; and the exact deployed v5 source revision.
All paths in selection/config must be absolute because agent invocation uses an arm-local cwd.
Preparation fingerprints v5/runner source, resolved agent models/retries, seed, transport
policy, and evaluation. Changed substantive settings require a new shared preparation;
run/cache locations and GPU assignment are intentionally not fingerprinted. Keep the
existing provider configuration file unchanged between arms (its secrets are never read
by this fingerprinting code). Each summary requires exactly one A and one B for one path.
No credentials belong in selection, shared JSON, command-line inline provider overrides,
or published results. Existing wiring may copy provider credentials into sandbox
`opencode.json`: **run directories are secret-bearing; do not publish them wholesale.**

## Commands (remote operator only)

Run from the deployed v5 root. Explicit PYTHONPATH is mandatory because the old
editable environment can resolve dirty v2. Do not install/edit a shared environment.

```sh
export V5=/absolute/deployed/v5
export PYTHONPATH="$V5/src:$V5"
export PY=/root/autodl-tmp/orch-venv/bin/python
export CFG=/absolute/inputs/runtime.yaml
export INPUT=/absolute/inputs
export RUNS=/absolute/new-runs

"$PY" -m scripts.experiments.c2_local_runner --config "$CFG" prepare \
  --selection "$INPUT/selection.json" --output "$INPUT/shared.json"

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_local_runner --config "$CFG" \
  smoke-parent --shared "$INPUT/shared.json" --run-dir "$RUNS/smoke"

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_local_runner --config "$CFG" \
  acquire --shared "$INPUT/shared.json" --run-dir "$RUNS/acquire" --probe-budget 12

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_local_runner --config "$CFG" \
  run-arm --shared "$INPUT/shared.json" --run-dir "$RUNS/direct-A" --path direct --arm A
CUDA_VISIBLE_DEVICES=1 "$PY" -m scripts.experiments.c2_local_runner --config "$CFG" \
  run-arm --shared "$INPUT/shared.json" --run-dir "$RUNS/direct-B" --path direct --arm B \
  --responses "$RUNS/acquire/responses.json"

"$PY" -m scripts.experiments.c2_local_runner summarize \
  "$RUNS/direct-A/result.json" "$RUNS/direct-B/result.json"
```

For legacy, use `--path legacy` and new `legacy-A`/`legacy-B` directories, with the
**same shared.json and baked responses.json**. Schedule four jobs only with four
distinct available physical GPUs; do not overlap two jobs on a GPU. On two machines,
copy the same shared/response inputs if running the same task. Different tasks are
separate paired comparisons, not four interchangeable arms.

Example overrides (before the subcommand; keep substantive settings identical in A/B):

```sh
"$PY" -m scripts.experiments.c2_local_runner --config "$CFG" \
  --set wsl.venv=/root/autodl-tmp/kernel-opt-venv \
  --set kernelbench_root=/absolute/KernelBench \
  --set wsl.kernelbench_src=/absolute/KernelBench/src \
  --set opencode.sandbox_config_path=/absolute/existing/opencode.jsonc \
  --set agents.default_model=zhipuai/glm-5.3 \
  --set evaluation.perf_trials=100 \
  smoke-parent --shared "$INPUT/shared.json" --run-dir "$RUNS/new-smoke"
```

Use those same evaluation/device overrides at preparation. A's selected worker is
`/root/autodl-tmp/kernel-opt-venv/bin/python`; B's is
`/root/autodl-tmp/orch-venv/bin/python`. `wsl.venv` takes the **directory**, not the
executable. Provider effort/limits/credentials stay in existing wiring/config; the
runner does not claim wire-level effort verification.

## Protocol and files

- `prepare` is CPU-only. Shared inputs include tuned source, reference, ordinary
  trial records (including failures/resources), regenerated tuning statistics,
  semantics, device, and evaluation settings. Neither probes nor future results
  enter a child's TPE history or choose the parent.
  Absolute paths in historical failure details are redacted before prompt delivery.
- `acquire` uses existing `probe_responses`: fresh legal min/max endpoints per
  parameter, with all other values held at the tuned parent. The explicit budget
  counts actual endpoint attempts. Missing/failed/budget-denied axes remain unknown.
  This is v5's native task-response protocol, **not legacy C4 discovery/validation**.
  B validates the baked envelope against the same shared identity, axes, fixed
  partners, domains, and attempt counts. Failed acquisition with no valid contrast
  remains explicitly `no_valid_contrast`, not evidence of successful treatment.
- Direct calls the unchanged task rewriter once, then native `TaskSearch` with
  B40. Its optional returned space is honored; otherwise the parent space is used.
  The KernelBench adapter still requires runnable ModelNew plus literal PARAMS.
- Legacy regenerates the analyst per arm from common ordinary evidence, with only
  B's brief added, then calls the unchanged rewriter (`n_candidates=1`) and existing
  parameterizer once. Changed keys/domains are permitted. There is no domain narrowing.
- B40 is the **maximum actual trial budget of one published child space**. Existing
  guard/duplicate pruning and finite-space exhaustion can yield fewer than 40 trials;
  the driver does not change the sampler to force forty duplicate measurements.
- Each arm independently full-smokes the fixed parent before generation. Selection
  compares valid child **tuning** observations with the explicit parent's historical
  tuning observation. Failure/no valid child retains the parent, but not zero cost.
- Selection and child params are saved before fresh measurement. Three full final
  blocks (or `--final-blocks N`, N >= 3) alternate parent/child order. Both artifacts
  are measured even when tuning selected the parent, if a valid child exists.
  Full evaluation uses configured 100 performance samples and unchanged correctness.
  Failed final measurements never cause a different candidate to be selected.
- `events.jsonl` records agent events, worker attempts/raw results/wall time,
  trials, published space, selection, and outcomes. `jobs/` retains existing worker
  inputs/outputs. `result.json` separates tuning, frozen selection, fresh parent,
  fresh child, error, and elapsed time. Acquisition has its own store and costs.
  `costs` aggregates worker attempts/failures/wall time and agent calls/attempts,
  tokens, provider-reported costs, and missing final accounting. B links the separate
  `acquisition_costs`; do not charge this shared acquisition once per arm. Parent-smoke
  failures produce terminal arm results with an error and no child generation.
  Existing agent events preserve aggregate token/cost totals over attempts; they
  are not precise per-token billing or GPU busy-time measurements. Provider costs
  unreported after a connection failure cannot be recovered from these events.
- Separate run/job/sandbox/Triton/PyTorch-extension/XDG/temp locations isolate harness
  state. Worker-mode commands scope these process environment variables and restore
  them on exit; native worker subprocesses inherit the scoped values. They are
  **not filesystem access-control boundaries**: inherited agent permissions and
  credentials remain those of existing wiring. Use operator-level isolation if
  protection against an agent deliberately reading other directories is required.
  Distinct GPU assignments remain necessary because locks are run-local.

Summaries show descriptive block medians rounded to four decimals, valid-block
counts, attempts, errors, and wall time. Raw samples remain available. Three timing
blocks are not three independent searches; no statistical significance is claimed.

## CPU verification

```powershell
.\.venv-v5\Scripts\python.exe -m pytest tests/test_c2_local_inputs.py tests/test_c2_local_execution.py -q
.\.venv-v5\Scripts\python.exe -m scripts.experiments.c2_local_runner --help
```

Only provider transport and GPU worker are faked in execution tests. Production
agent validation, TaskSearch, TPE, materialization, correctness wrapper, and
Benchmarker run as existing v5 code. No local model/GPU experiment is implied.
The focused suite includes A and B for both paths, native acquisition at fixed
partners, all-invalid children, generation and parent-smoke failures, frozen final
selection, source/config identity, domain preservation, costs, and cache isolation.
