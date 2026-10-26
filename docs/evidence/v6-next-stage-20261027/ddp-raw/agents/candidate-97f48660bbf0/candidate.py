"""Module-boundary host rendezvous on gpt2.lm_head for DDP rank alignment.

Repairs trial-98426ff10f's measured step_ms failure (214.67 ms vs the 121.95 ms
limit) by moving its validated gloo rendezvous off the two c10d::allreduce_
functional sites and onto the module boundary group b57f55423ea79f2933a2
(call 03e8e3a0a8a549d528e18317, torch.nn.modules.linear.Linear, target
gpt2.lm_head, phase microbatch_forward_backward, 4 invocations per step on both
ranks). The binding_cost AB/BA collector on this exact parent revision
(observations/previous-after/binding_cost-2087a4f7.json) isolated the prior
failure in the binding layer, not the candidate logic: the legacy functional
dispatch mode wraps the whole adapter.run in a TorchDispatchMode (~7,240
dispatcher-visible ops per step) and adds ~97-107 ms before candidate code runs
(parent arm median step_ms 115.7/119.3 ms vs binding-only passthrough arm
217.0/226.3 ms). A module-only bundle uses the direct callable boundary instead
(binding-contract.md): zero functional sites, four Python invocations per step,
no dispatch mode around unrelated operators. That prior trial's alignment logic
itself measurably worked (rank-0 communication median 38.09 -> 36.59 ms) and
was bitwise exact, so the mechanism is retained unchanged in effect.

Replacement semantics (parent-exact passthrough plus rendezvous):

* Call the current parent forward (context.references) with the original
  args/kwargs and return its result unchanged: identical kernels, identical
  autograd graph, identical gradients and DDP state. The four inner aten ops of
  the boundary (view/t/mm/_unsafe_view) still execute inside the parent call
  and are neither replaced nor eliminated. The two c10d::allreduce_ sites are
  not enclosed by this boundary and stay on the parent path.
* After the parent call, execute one gloo barrier on the adapter's existing
  state['control'] group (created in prepare with backend="gloo", timeout 120 s,
  already used by before_run outside timing), on every invocation: 4 symmetric
  rendezvous per step, one right after each microbatch's vocabulary projection,
  before loss/backward. No workload constant is embedded; the barrier fires on
  every call of the boundary.

Measured motivation (current-revision collectors, descriptive not prescriptive):
DDP bucket-allreduce CPU submissions carry a stable 5.2-6.4 ms cross-rank
spread (execution_timeline-1221e205 cross_rank bounds 5.23-6.42 ms), which the
leader's first NCCL kernel absorbs as in-kernel spin-wait (8114 us ~= 6.2 ms
wait + 1.93 ms transfer; nsys-9ed9145c leader extreme 205.7 ms first kernel
with the remaining 12 kernels at floor), while back-to-back bucket kernels run
at the ~34.5-35.1 ms per-step transfer floor (collective_calibration-1af5837f:
1.993 ms x 12 for 28,351,488 B and 10.61-10.65 ms for 157,541,376 B) against
the 38.09/38.40 ms baseline max-rank medians. Per-microbatch grad_ready lag
grows 2.98 -> 5.35 ms across the step, so the skew is created before the
reduction phase; aligning the ranks after each forward caps the lead at about
one microbatch and moves that wait out of NCCL kernel durations into a host
barrier that launches no NCCL or CUDA kernel.

Safety: lm_head is bound at instance level (the 48 distinct Conv1D groups show
instance-level binding), so the evaluator's separate serial-reference model is
untouched. Both ranks execute the identical 4 invocations in every
adapter.run invocation (warmup, repeats, continuous steps, the 3 profiled
diagnostic runs and the inventory-hook run), so barrier counts are symmetric
by construction; the barrier pairs only with the same barrier on the peer rank
in the same invocation. No new process group, no NCCL/NCCL_PROTO or environment
tuning, no build-time collectives or GPU work, no CUDA Graph capture, no
parameter transforms, no cached outputs. Barrier failures never poison the
step: the parent result is already computed, exceptions are caught and
reported as distinct setup/runtime route labels, and the call degrades to
parent-exact passthrough.
"""

from __future__ import annotations

import torch.distributed as dist

LM_HEAD_CALL = "03e8e3a0a8a549d528e18317"


def build(context):
    state = context.get("state") or {}
    references = context.get("references") or {}
    record_route = context.get("record_route", lambda *_: None)

    replacements = {}
    parent = references.get(LM_HEAD_CALL)
    if not callable(parent):
        return replacements

    control = state.get("control") if isinstance(state, dict) else None
    align = None
    try:
        if (
            control is not None
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() >= 2
        ):
            def align(_group=control):
                dist.barrier(group=_group)
    except Exception:
        align = None

    def route(label):
        try:
            record_route(LM_HEAD_CALL, label)
        except Exception:
            pass

    def lm_head_parent_exact_then_gloo_align(*args, **kwargs):
        output = parent(*args, **kwargs)
        if align is None:
            route("setup-no-align-group-parent-exact-passthrough")
            return output
        try:
            align()
            route("parent-exact-then-gloo-control-align")
        except Exception:
            route("runtime-align-failure-parent-exact-passthrough")
        return output

    replacements[LM_HEAD_CALL] = lm_head_parent_exact_then_gloo_align
    return replacements
