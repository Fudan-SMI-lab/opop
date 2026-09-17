# Existing-child retuning only

Run one independent B40 study per invocation. This entry calls the existing
`SpaceValidator.validate_and_publish`, registers a normal candidate/family, converts
accepted witness measurements exactly as `_candidate_pipeline` does, then calls
`Orchestrator._tune`. It never calls `_candidate_pipeline`, an agent, expansion,
analysis, novelty, or repair. A rejected witness/space is an outcome, not a repair request.

## Inputs

Use the original **parameterized.py**, not `frozen-child.py`, a later trial artifact,
or a source with follow-up winning PARAMS substituted. For the first-wave legacy
children the saved originals have 19 params each: A has `SPLIT_M=16, DW_WARPS=4`;
B has `SPLIT_M=64, DW_WARPS=8`. Their distinct defaults remain their own validator
witnesses. No special-case parameter names or fixed-point anchors exist in the runner.

Supply a JSON input file:

```json
{
  "task": "level3:21",
  "source": "/absolute/inputs/A-parameterized.py",
  "space": "/absolute/inputs/A-published-space.json",
  "reference": "/absolute/inputs/reference.py",
  "sampler_seed": 0,
  "evaluation_seed": 0,
  "output": "/root/autodl-tmp/c2-local-experiment/runs/retune-A-s0",
  "backend": "triton",
  "helpers": []
}
```

All seven fields through `output` are required. Seeds are nonnegative integers;
backend defaults to `triton` (`cuda` also supported). `helpers` is an optional list
of explicit sibling/helper file paths beneath the original source directory; they
are copied with relative layout into the new run. The current two children are
self-contained. Nothing writes into the old source/reference/history directories.

`space` is the **whole `SPACE_PUBLISHED.payload.space` object**, using the existing
`ParameterSpace` schema: `space_id`, `candidate_id`, `source_sha`, `version`,
`domains` (each name/kind/full choices/description), and `constraints`.
Do not pass the enclosing event or raw string-encoded model proposal. Preserve
the original complete domains and constraints; the normal validator owns acceptance.

`--config` accepts the existing resolved **AppConfig YAML or JSON object** (not a
manifest envelope). Use the evaluation/device/worker settings from the resolved
first-wave manifest/config. No config file is modified. The evaluation seed must
be the fixed historical workload seed; `0` above is an example, not an inferred value.

Machine-readable input JSON Schema:

```sh
"$PY" -c 'import json; from scripts.experiments.c2_retune import RetuneInputs; print(json.dumps(RetuneInputs.model_json_schema(), indent=2))'
```

## Exact execution commands

From the deployed v5 source tree, use explicit PYTHONPATH to avoid the dirty v2
editable install. `PY` is the existing orchestrator interpreter; the worker
interpreter still comes from `wsl.venv` in the selected config.

```sh
export V5=/absolute/deployed/v5
export PYTHONPATH="$V5/src:$V5"
export PY=/root/autodl-tmp/orch-venv/bin/python
export CFG=/absolute/inputs/resolved-config.yaml
export INPUT=/absolute/inputs

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_retune \
  --config "$CFG" --input "$INPUT/retune-A-s0.json"
CUDA_VISIBLE_DEVICES=1 "$PY" -m scripts.experiments.c2_retune \
  --config "$CFG" --input "$INPUT/retune-B-s0.json"
```

Prepare paired files for sampler seeds **0, 1, 2**, retaining the same evaluation
seed and each arm's original source/space/reference. Change only sampler seed and
output root between studies of a structure. Assign at most one active job per GPU;
schedule the six invocations over the four available GPUs. Every output root must
be new. No automatic multi-study launcher or server deployment is included.

## Preserved native behavior

- The default/minimal witnesses and any native permitted witness retry come only
  from `SpaceValidator`. Rejection records the native reason and stops immediately.
- `anchors = tuple(w.params for w in accepted.witnesses)`. A witness with latency
  becomes a cache record via existing `latency_from_result` and profiler APIs.
  Reused anchors count toward the same 40 native tuner trials, just as normal v5.
- `_tune` retains native prescreening, cached shared-memory admission, on-demand
  screens, `constant_liar`, ordered categoricals, deweighting, guard/PRUNED behavior,
  and configured timeouts. Finite-space/guard exhaustion may produce fewer than 40.
- C2 acquisition/E and slope-guide injection are explicitly off in every study.
  This is retuning fixed structures, not a fresh C2 information ablation.
- `cfg.run.seed` controls the native sampler/deweight/prescreen configuration draws.
  Validator sampling and correctness/final evaluator seeds are fixed separately.
  The native compile probe has no job seed field, so the small retune worker entry
  sets KernelBench's seed once before delegating to the unchanged worker. Thus its
  input RNG does not follow sampler seed; native batch traversal is unchanged.
- No runtime server starts. Existing agents are constructed by normal wiring but
  never invoked. Existing provider config may be read by that wiring; no credential
  content is written into this runner's manifest or sent to a model.
- The native family best is selected solely from tuning observations. Its params
  are materialized once in `report/selected.py`, then evaluated in three fresh full
  worker blocks with configured 100 performance samples. Finals never rerank it.

## Outputs and interpretation

`events.jsonl` contains normal registration, published space, prescreen, trial,
reuse, and tuning events, plus `RETUNE_VALIDATION`, `RETUNE_SELECTED`, and final
measurement events. `jobs/` retains native worker payloads/results. `result.json`
separates native `selected`, tuning `trials`, and three `finals`; status is
`complete`, `rejected`, `no_best`, or `final_failed`. Non-complete status exits 1.

All generated sources, worker jobs, cache/temp paths, and final artifacts are under
the selected run root. Retuning creates no protocol/hash/resume infrastructure.
Report the three independent studies separately; three final timing blocks are
not independent searches and imply no statistical significance.

CPU verification (no GPU/model calls):

```powershell
.\.venv-v5\Scripts\python.exe -m pytest tests/test_c2_retune.py -q
```
