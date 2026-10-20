Read inputs/intent.json, inputs/deployment.json and inputs/intake.schema.json.
Resolve the actual project and goal into intake.json. The repository root and
file index are in inputs/repository.json. Use tools to inspect relevant source,
README, launch commands, configuration and tests there, starting with the supplied
command if present. Do not load the whole repository into the conversation.

Your job is task intake, not selecting operator groups or proposing optimized
implementations. Do not run GPU jobs, modify the source or install dependencies.
The following evaluator role and leased workers run the selected workload.

Select the entry and complete source/config file closure required to understand
and execute it. Paths in files are relative to the repository root, and include
the entry file. External model weights and data stay in assets as actual absolute
paths; never copy weights into source or invent assets. Trace local imports and
configuration references instead of assuming one self-contained script.

Infer the exact workload and quality requirements from the user's command,
context and project evidence. Preserve architecture, precision, numerical
policies, data, effective batch, generation length, state semantics and output
requirements. If a source offers several workloads, use the user's goal/context
to select one. Ask a concise question only when the task cannot be determined
without inventing a consequential choice. Do not ask for facts already available
in source or deployment. Ready requires evidence for entry, workload, objective,
quality and assets; evidence values are concrete source paths/symbols or user
statements. Record assumptions separately.

The objective contains name (the exact metric key, not a task title), unit and
direction (minimize/maximize). Its metric must correspond to the requested
completion boundary. Constraints use the supplied schema: a list of metric, op
and either baseline_ratio or threshold. Convert a project's constraint notation
into that generic representation while preserving its meaning. Do not
change a source's full-request throughput into cached-only throughput. Keep
source quality thresholds and expose generic atol/rtol for tensor comparisons.
Describe discrete outputs that need exact comparison in notes. If task quality
semantics are genuinely absent, return needs_input rather than inventing loose
tolerances. Workload and quality are copied unchanged into the later frozen
contract, so retain the source's exact field names used by the project.

device_count describes GPUs actually required for this workload, not all GPUs
present in the machine. Use the observed deployment inventory. A distributed
backend is needed only for an actual multi-process distributed task; the worker
handles rank/device binding and group initialization. Communication defaults to
auto. Do not require unavailable counters. Tool paths, UUIDs and lease settings
are supplied by deployment rather than typed into the description.
required_observations refers only to required communication collector fields;
ordinary evaluator metrics belong in objective and its constraints.

For diagnostic objectives, declare their existing collector source only if the
project/user actually requests that metric. The task's ordinary end-to-end time
and quality remain distinct from diagnostic kernel time.

Write intake.json following the schema. Use status=needs_input, a concise
questions list and description=null if unresolved semantics prevent a valid
request. Otherwise use status=ready with description, evidence, assumptions and
an empty questions list. Report existing project behavior; do not add proposed
optimizations or model-specific framework rules.
