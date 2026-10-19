Generate candidate.py for the selected project optimization. Read inputs/request.json, selection.json, targets.json, adapter.py and the supplied source. The goal, workload, quality, dtype, completion boundary and state requirements are frozen. Never change batch, sequence length, generation count, effective training batch, sampling semantics, data or thresholds to manufacture improvement. Do not precompute/capture final answers for the known fixture; the workload must execute on each invocation. Do not modify framework, evaluator, reference inputs or external files.

Follow inputs/proposal.schema.json exactly. In particular, implementation is one enum string such as "composite", never a description; put descriptions in rationale.

Read target-resources.json for rank-local durations, phase breakdown and observed kernel launch configurations. Missing entries are unknown, not zero. Launch registers/shared memory are not measured occupancy, hardware traffic or independent causal effects. The full inventory is available in call-resources.json for contextual queries.

parameter_space follows inputs/parameter-space.schema.json and is evaluated by the existing TaskSearch/TPE loop against the complete project. Use {} for no tunable parameters. If defining parameters, read them from context['parameters']; declare legal choices and constraints without changing frozen task semantics. The selected parameter candidate is independently revalidated with paired project measurements before publication.

candidate.py must export build(context) -> dict[call_id, callable]. Context contains:

* state: the actual adapter state after prepare, before validation runs;
* models: root model mapping from adapter.models(state);
* modules: selected module call_id -> actual nn.Module;
* references: call_id -> the current parent's bound forward, torch operator, or adapter.run;
* targets: call_id -> observed call specification, with kind, target, signatures and semantics;
* parameters: optional tunable values supplied by the coordinator.
* rank/world_size: actual worker identity; targets/modules/references are local to this rank. all_targets retains the complete cross-rank inventory. Return only local declared replacements (an empty mapping is legal on a rank with no replacements), while preserving collective participation on all ranks.

For module targets, the callable has the original forward's args/kwargs, without self. For functional targets it has the exact operator signature; replacements apply only to observed compatible shape/stride/dtype/scalar signatures and otherwise fall back to the parent. For project.invoke it receives state and must return exactly the same result structure as adapter.run. Scope does not change this ABI: a joint candidate returns mutually compatible replacements for multiple sites. Preserve all external outputs and state/alias semantics, including all gradients and parameter updates for training. Avoid in-place rewrites unless alias/state equivalence is established.

One physical module may have separate observed call IDs for phases such as prefill/decode. The framework composes their replacements behind phase/signature guards. Build invariant packed weights once per physical module, sharing them between phase callables where appropriate. Do not install another module.forward wrapper yourself.

The framework installs the whole mapping, counts actual candidate calls during quality/coverage checks, measures the entire project without profiling, and restores the parent on failure. You may transform invariant model parameters during build when valid for the task's warm-model lifecycle; document setup work and memory in proposal.json. Do not cache fixture-dependent outputs. Avoid ad hoc GPU tests; the framework uses the bound worker and returns detailed measured feedback.

Return replacements for a coherent set of selected sites. To optimize a whole execution policy, use a project.invoke target and explicitly report affected groups, state lifecycle and synchronization in proposal.json. If a proposal eliminates inner calls, list eliminated_calls and explain why their consumers remain valid; the framework will recapture the plan. Do not claim missing measurements or unknown transport capabilities as observed facts.

Write proposal.json with implementation (kernel/library/composite/distributed/strategy), replacement_calls (the exact call IDs returned by build), rationale, setup_work, quality_risks, eliminated_calls, requires and replaces (installed bundle IDs, empty arrays if none), and parameter_space (a JSON object, empty if no tuning parameters). Installed bundle metadata is in search.json and their source is supplied. Overlapping call writes require explicit replacement of the earlier whole bundle; preserve its needed behavior yourself. If inputs/previous-feedback.txt exists, explain in rationale how it changes this attempt. Write the two output files and return their names.
