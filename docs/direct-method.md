# Direct task optimization (v5)

The direct path accepts a project, candidate file, parameter space, and task-owned
evaluation. It optimizes native **J**, optionally measures controlled responses and
rewrites the structure, then **executes the selected winner again** through the same
evaluator before closing it. There is no task/model registry, KernelBench ID,
`ModelNew`, or obligatory `PARAMS` block on this path.

## Run the included CPU example

From `D:/Pyhon_projects/opop/v5`, use the v5 environment (the old `.venv` points to v4):

```powershell
.venv-v5/Scripts/python.exe -m kernel_optimizer.cli optimize-task --project examples/direct_task --candidate candidate.py --space space.json --eval-file eval.py --eval-function assess --direction minimize --context context.json --label "payload delta" --unit bytes --trials 4 --output results/direct-cpu
```

This provided-evaluator, parameter-only command needs no agent server, GPU, torch,
or KernelBench. It selects `width=1`, native J=-1; `--direction maximize` selects
`width=3`, J=1. The incorrect width=9 result is excluded. Final execution runs the
selected width again and reports its independently returned J.

## Enable structural optimization

Append `--rewrite-rounds 1 --probe-budget 4 --goal "Preserve sorted output; minimize payload size"`
to the example command. This starts the **configured agent host** and performs:

1. Real TPE parameter search and strict native min/max best selection.
2. Fresh evaluations at two legal values of each probed parameter, keeping other
   parameters fixed at the incumbent. The probe budget is a total callback-call
   limit across all rewrite rounds, not a per-axis or per-round allowance.
3. A structural agent receives candidate source, goal/context, parameters/space,
   native J responses and available resource observations; it writes a child file.
4. The unchanged evaluator checks and tunes that child. A failed child cannot win.
5. One final evaluation executes the overall selected path with its winning params.

Use repeatable `--resource-metric NAME` for numeric **resource measurements actually
returned by your evaluator**. Declare their units and meaning in context. The
included example does not report resources; it still supports parameter contrasts
and ordinary source-guided rewriting. Missing resources stay unknown.

Numeric slopes use actual nonzero parameter spacing; nominal choices only produce
contrasts. Resource slopes require a selected metric observed at both endpoints and
a nonzero resource delta. These are local empirical responses, not hardware causal
claims or inferred resource limits. Diagnostic probes are reported but do not enter
TPE, best selection, tuning statistics, or the search valid/invalid counts.

`--rewrite-rounds` defaults to **0**, meaning parameter-only optimization, not the
full structural method. Existing family/convergence and configured search wall
budgets still apply. Calls already executing synchronously are not preempted.

## Generate an evaluator

Omit `--eval-file` / `--eval-function` and provide `--goal`:

```powershell
.venv-v5/Scripts/python.exe -m kernel_optimizer.cli optimize-task --project examples/direct_task --candidate candidate.py --space space.json --context context.json --goal "Sort the entire input correctly; minimize payload bytes minus two" --direction minimize --trials 4 --rewrite-rounds 1 --probe-budget 4 --output results/direct-generated
```

Add repeatable `--source reference.py` for reference or other readable project
sources the agents need. Generated evaluation and rewriting share one Runtime.
Agent settings come from the existing configuration; put global `--config PATH`
and `-o KEY=VALUE` before `optimize-task`. Modules use `agents.eval_builder` and
`agents.rewriter`. A builder direction conflicting with explicit `--direction`
fails rather than silently changing the objective. Host checks compile Python
syntax without importing target GPU code; actual evaluation requires its runtime.

## Evaluation and parameter inputs

The exported synchronous callable receives:

```python
def assess(candidate_path, params, context):
    # Execute candidate_path using params and check all outputs independently.
    # Return a finite native int/float J, or TaskEvaluation(score=J, metrics=...).
    # For incorrect outputs, return TaskEvaluation(valid=False, detail="reason").
    ...
```

`candidate_path` is a `Path`; `params` and `context` are mappings. The callback owns
loading, applying parameters, executing the full task, checking correctness and
writing any task output artifacts. It must execute the candidate on each call, not
serve cached evaluation results. It may retain loaded models in its module: one
TaskEvaluator instance is reused for search, probes, children and final execution.
This runs trusted code in process, not in a security sandbox or new worker pool.

Space JSON follows the existing choice-domain format:

```json
{"params": [{"name": "width", "kind": "int", "choices": [1, 2, 3, 9]}],
 "constraints": [{"expr": "width > 0", "rationale": "positive width"}]}
```

Kinds are `int`, `float`, or `str`. CLI context is a JSON object with task-specific
names/data. Candidate, space, evaluator, context and source paths are relative to
`--project` unless absolute; `--output` is relative to the command's working directory.
No custom bytes/throughput/work score is stored as `latency_ms`.

## Completion and output

The output directory contains `summary.json`, `trials.json`, `tuning_stats.json`,
and `report.md`, plus generated agent workspaces when used.

- `best` / `best_candidate`: the historical search winner and its path/params/J.
- `final_execution`: a separate actual execution record of that winner, including
  validity, native score, metrics and failure detail. It is not a tuning trial and
  does not overwrite or rerank historical measurements. Scores may vary across runs.
- `responses`, `probe_calls`, `rewrites`, and `parents`: measured contrasts and
  parent/child results; unknown data are not replaced with zeroes.
- `valid_count` / `invalid_count`: search trials only, excluding diagnostic probes
  and the one final execution.

Exit **0** requires a valid finite final result. No valid search winner, incorrect
final output, callback failure, missing score, or nonfinite final score yields exit
**1**; the saved report retains the failure. There is no silent fallback/reselection.
Configuration/load errors raise explicitly. One final callback is performed after
search whenever a winner exists, in addition to the search/probe budgets; its time
can extend total elapsed time. Keep candidate and generated helper files available:
the saved path is a runnable source location, not a packaged deployment.

## Verified scope and remaining acceptance

```powershell
.venv-v5/Scripts/python.exe -m pytest tests/test_v5_direct_end_to_end.py tests/test_v5_direct_legacy_regression.py -q
```

Tests execute actual CPU candidate and evaluator code through provided/generated
routes, both directions, controlled probes, child retuning and final execution.
Only the model provider/Runtime boundary is fake. These tests do **not** establish
live LLM reasoning quality or real GPU/model performance. Legacy latency behavior
and useful retained packing/scoring APIs remain supported and regression-tested.

**Real 8B single-stream and multi-stream acceptance remains BLOCKED:** no checkpoint/
tokenizer, workload/data, independent reference/evaluation inputs, or suitable
hardware have been supplied. CPU tests are software validation, not substitutes
for that acceptance. No model download or live agent/GPU experiment is required
or performed by these tests.
