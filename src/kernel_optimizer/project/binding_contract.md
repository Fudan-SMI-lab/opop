Candidate composition and execution contract

The evaluator calls adapter.prepare(), then installs bundles in revision order.

The evaluator adapter preserves the source workload; it does not choose or install
optimization bindings. Selection chooses the editable boundary, the candidate
returns replacements for declared sites, and the framework installs them.
Functional replacements can incur dispatch cost on unrelated operators in the
invocation. Module/project replacements use direct callable boundaries unless
they also install functional replacements. Account for the effective binding
backend when predicting whole-project benefit. Scope names alone do not select
a cheaper backend: the actual returned replacement sites determine it. The
optional binding_cost collector measures parent versus binding-only passthrough
on additive candidates. Its cost is diagnostic and remains part of the score.

The request's optional `binding_backend` defaults to `legacy`. The experimental
`scoped` backend activates functional dispatch only inside observed module
boundaries when every installed functional site has a resolvable boundary.
Unscoped sites and multiple functional bundles retain the existing global
dispatch path and precedence. The setup receipt records the effective backend
and reason. Candidate call interfaces, live signatures, phases, quality and
coverage requirements are unchanged; backend selection does not route search
decisions. A diagnostic backend comparison never publishes a candidate.
build(context) is installation: the default project phase is "run", quality
tracking is off, and the adapter's normal invocation has not started.
context.references contains the current callable at each declared boundary.
It does not implicitly run the adapter's phase contexts or activate inherited
functional dispatch modes. Inherited module replacements have phase/signature
guards too. Calling a raw parent forward at installation is therefore not proof
that every inherited implementation ran.

During an actual adapter invocation, the evaluator enters all installed
functional dispatch modes, and the adapter establishes its phases. Module hooks
provide the observed parent context for matching functional sites. Preserve
those semantics when composing replacements. A graph captured outside that
context can accidentally capture the original work instead of the accepted
parent's work, even when its numerical output is correct.

Quality coverage counts observed Python replacement invocations and calls through
context.references. CUDA Graph replay does not rerun Python hooks; route labels
and installation metadata alone cannot prove inherited native kernels executed.
All inherited required calls remain mandatory unless the earlier whole bundle
is explicitly replaced. Calling unused work solely to fill counters does not
satisfy effective composition. Use runtime traces to corroborate the work that
actually contributes to the returned output.

Lazy setup can be part of a legitimate invocation when needed to capture the
actual parent context, provided continuous execution, reset, cache/alias and
training semantics are preserved. The workload must execute on current inputs
on every invocation. Warm-model timing and lifecycle allocator metrics retain
their frozen definitions; no candidate may change measurement boundaries or
quality thresholds to hide its setup or adaptation costs.

When a proposed execution policy cannot preserve required inherited behavior,
select another coherent rewrite within the permitted closure or explicitly
replace the affected whole bundle. Shared project effects remain joint effects
unless legal independent measurements establish their attribution.
