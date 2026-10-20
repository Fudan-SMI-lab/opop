# V6 stage 3: project intake and evidence-driven validation

This stage implements the agreed follow-up to the twelve-run stage-2 audit.
Changes stay on `v6`. Existing structured requests and single-operator tasks
keep their entry points and defaults. Provider usage is observed, never capped.

## Work sequence

1. Record the inputs and prompt before each project-role generation. Provide
   independent fresh-session replay with one explicitly specified treatment.
   A continuation record is not treated as a fresh-session reproduction.
2. Add an optional repository + natural-language goal entry. A real intake
   Agent inspects the launch command, relevant source and necessary data,
   producing the existing `ProjectRequest`. Machine paths, available GPUs and
   profiler locations come from deployment configuration.
3. Measure the adapter differences identified in
   `evidence/v6-stage2-20261020/manual-adapter-audit.json`: Qwen timing boundary,
   ResNet statistics/input lifetime, and DDP gradient clearing. Use isolated
   copies with the same workload and evaluator; preserve original runs.
4. Replay the DDP calibration interpretation problem with a small presentation
   change, initially three fresh generations per arm. Preserve prompts,
   tool-visible input hashes, outputs, usage and invalid results. Sampling
   count defines this experiment, not a limit on development or model use.
5. Screen communication opportunity on the actual two-4090 topology. Verify
   that requested controls affect the real NCCL trace. Report end-to-end time
   separately from summed NCCL kernel time. Manual screening is not autonomous
   optimization.
6. Freeze the implementation for end-to-end tests: three existing projects
   with two goals each and an additional project. Include cross-goal,
   held-out-input and multiple-bundle checks where the generated results make
   those checks meaningful. Preserve all failures and distinguish code-path
   coverage from demonstrated performance gains.

The current all-fallback acceptance behavior is reported as a limitation; this
stage does not silently change acceptance rules. Failure-category routing,
restrictions on Agent information retrieval, and a search-policy redesign remain
deferred.

## New entry points

```sh
kernel-opt --config deployment.yaml optimize-project \
  --project /path/to/project \
  --goal "Reduce complete inference latency while preserving the existing accuracy checks" \
  --command "python infer.py --weights /path/to/weights" \
  --output /path/to/run
```

Use `--context` repeatedly for necessary data paths or constraints.
`--intake-only` saves the resolved request without starting optimization.
If a consequential task detail is absent, intake returns `needs_input` and
questions. Reusing the output with added context resolves those questions; a
completed request is frozen. `--request` remains available without intake.

Optional deployment configuration:

```yaml
project_defaults:
  devices: [] # empty: discover available physical GPUs
  lease_dir: /path/to/shared/device-leases
  tools: {} # explicit nsys/ncu paths override executable discovery
  profiling: {}
  worker_timeout_s: 600
```

Generated descriptions select a workload's GPU count. Merely having two GPUs
does not turn a single-GPU project into a distributed task.

New intake uses `measurement_lifetime: invocation`: evaluator-owned outputs from
quality checks and previous timed invocations are released before the next
measurement. Adapter-owned state is unchanged. Structured requests default to
`legacy`, retaining historical behavior; the option is explicit in the saved
request. This addresses measured-output lifetime, not the search or acceptance
policy. Installed diagnostic metric interfaces are described by the collector
registry, with actual hardware availability checked in workers.

Project-role records are written to `generation-snapshots/`. They preserve input
bytes in the run's content-addressed store and record model, tool settings and
framework hash. External assets and the entire tool-visible filesystem are not
snapshotted. Credentials remain in private deployment configuration.

```sh
python -m kernel_optimizer.project.generation_experiment \
  --config deployment.yaml --snapshot /path/to/generation.json \
  --output /path/to/comparison --edit treatment.json --repeats 3
```

The treatment is a JSON object with `path` (`prompt` or an input path), `old`,
`new`, and optionally `occurrences`. Each arm uses a new session. This command
records generations; optimization claims require separate independent
validation and measurement. Historical reconstructed inputs must be labeled
as reconstructed, never as contemporaneous snapshots.

## Progress

Initial implementation is under validation. No new performance or generalization
claim is made by this document. Stage evidence and final limitations will be
added after the actual runs.
