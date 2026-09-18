# T6: voluntary formal agent self-test helper

## Scope and evidence boundary

This supports T4/T5; it does not start an optimization campaign. Bash, Python,
read/write tools, permissions, private experiments, and native search budgets are
unchanged. No helper result is automatically admitted to TPE or counted as a B40
trial. The label is always `agent_self_test`.

The supplied audit found shell/miniconda Torch 2.8 / Triton 3.4 versus configured
worker Torch 2.13 / Triton 3.7. The same child source and all default parameters
gave approximately 7.854 ms privately versus 16.911 ms in formal **quick** 20/3
evaluation, not a full 100/5 evaluation. These facts motivate protocol alignment;
they do not isolate an interpreter/cache/timer cause or demonstrate new C2
efficacy. The delivered four-pair result remains two wins and two losses.

Implementation followed `plan-c2-region-followup.md` section 6 and the task's
requirements. `newagent-self-tests.md` was not present in the searched workspace,
so its requested sections 192–235 were unavailable for inspection.

## Operator CLI

Each configured call writes `task/self_test.json`, a separate
`task/self_test_reference.py`, its reachable local dependency snapshot, and
`task/self_test_usage.md`. The usage file contains the configured absolute worker
interpreter and the actual v5 source import path, not shell `python`.

From the agent sandbox on server A, for the existing deployment paths:

```bash
PYTHONPATH=/root/autodl-tmp/opop-workspace/KernelBench/src:/root/autodl-tmp/opop-workspace/v5/src \
  /root/autodl-tmp/kernel-opt-venv/bin/python \
  -m kernel_optimizer.agents.self_test \
  --context task/self_test.json \
  --candidate candidate/parameterized.py \
  --params task/params.json --mode full --output self-tests/full-001
```

Use `/root/autodl-tmp/orch-venv/bin/python` on the server configured with that
worker environment. **The generated usage file is authoritative**, including
any configured `extra_pythonpath`; the deployment path above is an example.
Do not replace it with PATH's Python. The environment assignment is local to
this command, not a global PATH/environment modification. On Windows, run the
generated command inside the configured WSL distro, with the sandbox as cwd;
context paths are translated using the existing worker path mapper.

`--params` is optional. If supplied, it is a complete native `ParamSet`:

```json
{"values": {"BLOCK_M": 32, "dtype": "bf16"}}
```

Those example keys are not universal: use this source's actual keys and values.
Unknown/missing keys fail through the existing materializer, with no guessed
mapping. Without `--params`, defaults are extracted and the source bytes are
preserved. `--mode quick` calls `CorrectnessEvaluator.quick_test`; `--mode full`
calls `Benchmarker.final_reeval`. Optional `--backend cuda` selects the existing
CUDA route; the default is `triton`.

The output directory must be new. Exit 0 requires formal `ok`, `compiled`,
`correct`, and a finite positive latency estimate (median when provided,
otherwise mean). All other formal results exit 1 with `score_ms: null`, even
when raw output contains a fast latency. Input/materialization failures also
produce failed JSON; an existing output directory is never overwritten.

## Protocol and artifacts

- The factory reads the **live** evaluator's seed/config at call time. A TPE
  sampler seed of 2 never substitutes for correctness seed 0. No `AppConfig`,
  provider block, API key, or OpenCode configuration enters helper context.
- Reference bytes/hash and input shape come from the reference snapshot. The
  already-seeded evaluation-semantics document is included when available;
  otherwise semantics are marked unknown. No extra agent or GPU probe is added.
- The helper reuses `WslGpuWorker`, worker Python, configured cache, job builders,
  quality gates including FP64 rescue, timing implementation, and the original
  jobs directory's device arbitration. It introduces no lock or budget default.
- Full/quick counts come from `EvalConfig` (normally 100/5 and 20/3), not constants
  in the helper. Warmup remains worker/KernelBench-owned, not a new config field.
  Installed versions are unknown unless present in raw worker output; no version
  or environment probe is performed.
- Every request gets a fresh materialized candidate, dependency layout, and
  reference outside the helper directory. AST-reachable local Python modules
  are copied without importing them or copying unrelated sandbox experiments.
  Dynamically discovered files, data assets, or dependencies outside the source
  tree still require the existing configured import environment; this is not a
  general project packager.
- `result.json`: validity, compiled/correct status, raw latency, quality mode,
  FP64 gate/rescue count, complete native parameters, source/reference hashes,
  seed/mode, evaluation/cache configuration, and wall time.
- `raw_worker.json`: the unabridged returned worker result, including rescue
  metrics and failure details. `context.json` records the request context.
  `commands.jsonl` records actual worker argv/job/output paths and only the
  `PYTHONPATH` / `TRITON_CACHE_DIR` environment fields. Raw job files remain in
  the existing shared jobs directory. No inherited credential environment is
  serialized.

## API examples and integration notes

Existing constructors remain unconfigured by default. Standard orchestrator
wiring injects the hook for generator, parameterizer, analyst, rewriter, novelty,
and repair. The analyst reuses the task already held by that orchestrator.
The hook runs after `seed_sandbox` and before `render_prompt`, including transport
reseed. It never overwrites an existing `task/ref.py`; parameterizers therefore
receive the formal reference even when their own module did not seed one.

For a custom composition root that already owns a task and live evaluator:

```python
from kernel_optimizer.agents.self_test_context import self_test_context_factory

parameterizer.self_test_context = self_test_context_factory(task, evaluator)
rewriter.self_test_context = self_test_context_factory(task, evaluator)
```

Do not capture `cfg.run.seed` or an early copied evaluation config. Existing
custom/experiment constructors that are not wired explicitly remain compatible
and unconfigured; no experiment/control files were edited by this helper owner.

## Verification choice

**CPU proof only; no new GPU jobs.** The tests execute the real CLI, evaluators,
job builders, worker client, and file protocol with only the GPU process launch
redirected to a narrow stdlib fake worker. They cover full/quick counts, FP64
gate/seed, configured interpreter selection, validity/failure exits, materialized
parameters, dependency layout, constructor compatibility, and secret exclusion.
This proves routing/contracts, not actual GPU numerical correctness or speed.

Any later parent-led GPU verification remains capped at four fixed full jobs:
the audited parent and child defaults twice each, without TPE, new candidates,
LLM calls, or quality relaxation. No such verification was performed here.

### Local validation record

- 14 focused helper tests passed; 42 tests passed including agent schemas,
  materialization, and worker protocol regressions. The combined run emitted
  one existing resource-watch-thread warning from `test_agent_schemas.py`'s
  incomplete `OpencodeClient` mock (`memory_abort_frac` absent).
- Ruff passed on all six changed Python files. LSP reported no errors on them.
  Standalone basedpyright reported zero errors for the three new modules and
  helper tests. Including the pre-existing base/wiring bodies reports nine
  errors at existing untyped dictionaries, generic return, and retry variable
  code; those unrelated bodies were not migrated.
- The programming rules audit passed for all four new Python files (each below
  250 pure LOC). Existing base/wiring audit violations remain, notably the
  already-oversized `AgentModule` file. The hook edits remain minimal rather
  than expanding this task into an agent-base refactor.
- Offline wheel build with `--no-build-isolation` was attempted and blocked by
  missing `hatchling` in the existing environment. No dependency was installed
  and no manifest or lockfile was changed.
