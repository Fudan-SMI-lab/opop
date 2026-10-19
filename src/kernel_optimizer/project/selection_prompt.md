Read inputs/request.json, inputs/groups.json, inputs/calls.json and the source/adapter. Select a scope and optimization opportunity for this exact project objective.

Groups and call IDs come from actual execution. CUDA durations are diagnostic, nested/inclusive and can overlap; do not add parent and child times. Unknown dependencies are not permission to reorder. A faster local kernel need not improve the requested project metric. Use the frozen workload, complete project goal and observed resources to justify the selection. Treat expected benefits as hypotheses to be tested.

Write selection.json with scope (one of inputs/search.json allowed_scopes), groups (a nonempty array of existing group IDs), rationale (string), hypothesis (string), and affected_calls (array of existing call IDs). All calls in selected groups must be included. For region or execution_plan include every additional observed call affected by the proposal. Choose a coherent editable boundary. Different edits that need one another should be a joint candidate. Do not enumerate fixed operator names as a generic policy.

Read inputs/previous-feedback.txt if present. Use its actual quality, project-metric, coverage, per-group and failure evidence to revise the selection. An unmeasured group is not permanently unoptimizable. No proposal is accepted on predicted benefit alone.
