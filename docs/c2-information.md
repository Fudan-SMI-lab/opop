# G0 / G1 / G2 single-step information experiment

One invocation produces one child opportunity at a fixed prepared parent. Run the
three groups for each of four parent states for the planned **12 opportunities**.
This entry composes existing `legacy_child` and native `c2_retune.retune`; it does
not call the first-wave handwritten `tune_legacy` loop.

## Information difference only

All groups use the same `Shared` JSON, complete original ordinary history/profile,
stats, reference, semantics, device, config/model/retry settings, and the same
baked `Responses` JSON for that parent. The full acquisition is validated with
`Responses.validate_for(shared)` **before** any group masking, including for G0.
Acquire once per parent using the existing acquisition command; this entry never
changes the probe method or acquires group-specific endpoints.

| Group | Extra agent information |
|---|---|
| G0 | None; ordinary data only |
| G1 | Same acquired endpoint params, validity, J, delta-J, gain, and parameter contrast as G2; fresh resource information withheld |
| G2 | Unchanged full responses from that same acquisition |

G1 clears endpoint `a/b.metrics` and `a/b.detail`, response `resource_deltas`,
`resource_slopes`, and `unknown_resources`. It sets `resource_status="unknown"`
and replaces free-form `reason` with `endpoint_resource_information_withheld`.
This also removes resource-sensitive failure explanations and byte counts while
retaining failure validity and missing endpoints. Empty resource dictionaries are
**not zero measurements**. G1 does not redact old ordinary resource evidence or
static constraints/parameter values. Full unmasked acquisition files and worker
logs are not copied into the agent's input bundle.

The existing legacy analyst is regenerated per opportunity, followed by its
existing rewriter (`n_candidates=1`) and parameterizer, each with the original
bounded retries. No report is reused across groups. The group name is not added
to agent inputs, and underlying prompts/contracts are unchanged—including the
legacy `wall_text` wording used identically for G1 and G2. No G2-only instruction,
forced axis reference, repair campaign, or additional proposal loop is added.

## Execution and selection

1. Declare the parent baseline from `shared.parent.latency_ms.robust_ms` before
   any new measurements. The prepared parent is already materialized at its
   recorded params; there is no parent search.
2. Run three full parent blocks with the existing worker/Benchmarker path, seed 0,
   and the original 100-sample evaluation quality. Keep all results and failures.
   Failed parent verification prevents generation and remains an error outcome.
3. Generate one child through `legacy_child`. Close the generation Runtime before
   launching validation/tuning workers.
4. Export the returned child space as a `ParameterSpace`, preserving all domains,
   constraints, and the original returned source/defaults. Call native `retune`
   in `retune/`, with sampler seed **0**, evaluation seed **0**, B40, native
   witnesses/cache/prescreen/pruning, and three full child final blocks.
5. Select the child only if its **native tuning-selected** latency is strictly
   lower than the predeclared parent baseline. Otherwise retain the parent.
   Fresh parent/child final latencies are not ranking inputs. Native retune fixes
   the child point before its finals; the outer entry derives its result label
   from that tuning snapshot after retune returns. There is no final-data rerank.

Generation errors, unsupported source/params, native witness rejection, no-best,
and failed final blocks remain explicit. There is no operator rewrite, injected
anchor, narrowed domain, or retry around the entire child opportunity. A child
whose final measurement fails stays selected if tuning selected it; the failure
is reported, rather than silently switching to a different output. Legacy children
use the existing single-file contract; unsupported dependencies are not repaired.

## Exact commands

Use the existing resolved AppConfig YAML/JSON and prepared input schemas. Keep
evaluation/device/model/retry settings identical across the three groups. Run from
the deployed v5 tree with an explicit PYTHONPATH; do not install into shared venvs.

```sh
export V5=/absolute/deployed/v5
export PYTHONPATH="$V5/src:$V5"
export PY=/root/autodl-tmp/orch-venv/bin/python
export CFG=/absolute/inputs/resolved-config.yaml
export SHARED=/absolute/inputs/parent-1/shared.json
export RESPONSES=/absolute/inputs/parent-1/responses.json
export RUNS=/root/autodl-tmp/c2-local-experiment/runs/information-parent-1

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_information \
  --config "$CFG" --shared "$SHARED" --responses "$RESPONSES" \
  --group G0 --run-dir "$RUNS-G0"

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_information \
  --config "$CFG" --shared "$SHARED" --responses "$RESPONSES" \
  --group G1 --run-dir "$RUNS-G1"

CUDA_VISIBLE_DEVICES=0 "$PY" -m scripts.experiments.c2_information \
  --config "$CFG" --shared "$SHARED" --responses "$RESPONSES" \
  --group G2 --run-dir "$RUNS-G2"
```

These commands are sequential as written. The operator may schedule jobs across
the four available GPUs, **one active opportunity per GPU**, with distinct new
run directories. Repeat with the other three declared parent states. `wsl.venv`
still selects the worker interpreter from the resolved config. Both seed values
are fixed to zero in this stage; there is no hidden seed sweep.

## Outputs and costs

- `result.json`: group/state, declared parent baseline/params, all three parent
  full blocks, complete native child result (including its finals), selected label
  and artifact, error, generation costs, parent costs, and shared acquisition costs.
- `generation/events.jsonl`: existing agent calls, retries, tokens, and costs.
- `parent/events.jsonl` and `parent/jobs/`: full parent measurement attempts/results.
- `retune/events.jsonl` and `retune/jobs/`: native validation witnesses, prescreen,
  tuning trials/reuse flags, tuning selection, and full final measurements.
- `common/`: unchanged ordinary/reference inputs; `child-space.json`: the full
  generated space before native validation.

Do not count the same baked acquisition cost three times. Validation, tuning, and
final measurement evidence remain separate in native retune events/jobs; reused
witness trials are marked and must not be counted as fresh GPU calls. Missing
provider-reported costs are not fabricated. Nonempty `error` exits 1, but the
opportunity's result and parent fallback remain recorded.

Existing sandbox permissions still apply: input masking and separate run/cache
directories are **not OS-level filesystem access control**. Provider wiring can
place credentials in sandbox config; do not publish whole generation directories.

## Scope and verification

This is a fixed-state, single-step information comparison, not an untreated vs
treated whole-pipeline comparison. Preserve `conditional_on_treated_state` labels
when applicable. **Long closed-loop/12-hour stages are not implemented here** and
must wait for first-stage results; the old active/off flag is not a substitute.
No resource-efficiency scalar or statistical-significance claim is produced.

```powershell
.\.venv-v5\Scripts\python.exe -m pytest tests/test_c2_information_inputs.py tests/test_c2_information_execution.py -q
```

CPU tests fake only external provider/server and GPU-worker boundaries. They run
the real agent validators, `legacy_child`, `retune`, and native `_tune`; they are
software verification, not GPU evidence.
