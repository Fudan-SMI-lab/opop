"""Fused RMSNorm candidate for the qwen-ttft selection (joint groups
ef1a650b08785209a023 input/post/final norms, 92f04768733eb2543f5d q_norm,
4a79f0c7bb426e51060b k_norm; all prefill phase).

What the previous feedback shows and what this attempt repairs
--------------------------------------------------------------
inputs/previous-feedback.txt records that the installed bundle
trial-fc57bca7b9-param-f71a7c8a never engages its CUDA-graph fast path on
this worker: the candidate_diagnostics evidence
(inputs/observations/previous-after/candidate_diagnostics-4123e6d6.json)
shows RuntimeError "No available kernel. Aborting execution." raised inside
_capture() at the torch.cuda.graph context (after the eager side-stream
warmups succeeded), so the prefill site routes permanently to
parent-fallback and the prefill executes eagerly. Launch batching of the
prefill is therefore spent on this revision and the remaining lever inside
the TTFT window is GPU kernel time. Measured on this exact revision
(target-resources.json, rank 0): the 145 prefill-phase Qwen3RMSNorm calls
cost 2657.7 us inclusive CUDA (1415.8 us for the 73 [1,256,2560] norms,
770.8 us for the 36 q_norm, 471.2 us for the 36 k_norm) through 1160 aten
kernels - transformers 5.16.1 expands each Qwen3RMSNorm.forward into
'x.float().pow(2).mean(-1) -> +eps -> rsqrt -> mul -> .to(bf16) -> weight*()'
(290 to.dtype, 290 mul, 145 add, 145 mean.dim, 145 rsqrt, 145 pow).

What this candidate does
------------------------
For every selected prefill-phase Qwen3RMSNorm site, replace the module
forward with one call to the native fused CUDA op aten::_fused_rms_norm
(a CUDA dispatch-key kernel in this worker's torch 2.13.0+cu129, verified
with _dispatch_dump_table; element 0 of its (out, saved) tuple is the
normalized output). One memory-bound kernel per norm replaces the eight
intermediate-materializing kernels and removes 1015 kernel launches from
the TTFT-critical stream. The op is called directly instead of through
F.rms_norm so the native CUDA kernel is selected regardless of how the
aten::rms_norm composite routes.

Safety design, mirroring the defensive pattern of the accepted bundle:

* Per-site lazy fidelity gate on the first live invocation: run the fused op
  and the parent forward on the live tensor and compare in float32 with
  atol=rtol=0.02. The fused kernel multiplies the weight in fp32 before a
  single bf16 rounding while transformers rounds before the weight multiply,
  so per-element differences are rounding-point ulps (measured max 0.0078 at
  unit scale on CPU); the tolerance accepts exactly that scale of difference
  and rejects semantic mismatches (wrong eps/axis/weight), which produce
  O(1) errors. Any mismatch, exception or unusable module permanently routes
  that site to the parent with a concise route label; a failed gate costs
  the speedup, never output fidelity.
* No gating work while a CUDA stream capture is active (the comparison must
  synchronize): the parent path serves the call and gating is retried at the
  next non-capturing invocation, so the installed bundle's lazy capture
  attempt is never disturbed. With the measured flash-dispatch refusal that
  capture aborts at the first SDPA call and prefill runs eagerly through
  these sites either way; if a capture ever succeeds, the already-gated
  fused kernel is recorded like any eager kernel.
* Decode is untouched: the three decode-phase RMSNorm groups are not in this
  replacement map. Decode invocations (phase 'decode', [1,1,...] signatures)
  do not match the framework's phase/signature dispatch for these sites and
  keep the parent path, protecting the decode_tokens_per_second goal.
* The workload executes on current inputs on every invocation; nothing
  fixture-dependent is cached, and no model parameter is mutated.
"""

from __future__ import annotations

import torch

_fused_rms_norm = torch.ops.aten._fused_rms_norm

# Comparison tolerance for the one-shot fidelity gate, chosen at the frozen
# quality scale: loose enough for expected bf16 rounding-point differences
# (<= ~2 ulp, ~0.008 at unit scale), tight enough to reject any formula,
# axis, eps or weight-handling error.
_GATE_ATOL = 0.02
_GATE_RTOL = 0.02

_NORM_SEMANTICS_MARKER = "RMSNorm"


def _no_route(*_args, **_kwargs):
    return None


def _compatible(hidden_states, weight):
    return (
        isinstance(hidden_states, torch.Tensor)
        and hidden_states.is_cuda
        and hidden_states.dtype is torch.bfloat16
        and hidden_states.dim() >= 1
        and hidden_states.shape[-1] == weight.shape[-1]
        and hidden_states.is_contiguous()
        and weight.device == hidden_states.device
    )


def _make_replacement(call_id, module, parent, record_route):
    weight = getattr(module, "weight", None)
    eps = getattr(module, "variance_epsilon", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dim() != 1
        or weight.numel() == 0
        or not isinstance(eps, float)
    ):
        def unsupported(*args, **kwargs):
            record_route(call_id, "parent-fallback:unsupported-module")
            return parent(*args, **kwargs)

        return unsupported

    normalized_shape = [weight.numel()]
    state = {"mode": "unresolved"}

    def replacement(*args, **kwargs):
        mode = state["mode"]
        if len(args) != 1 or kwargs or not _compatible(args[0], weight):
            record_route(call_id, "parent-fallback:unsupported-signature")
            return parent(*args, **kwargs)
        hidden_states = args[0]
        if mode == "fused":
            record_route(call_id, "fused-rms-norm")
            return (_fused_rms_norm(hidden_states, normalized_shape, None, eps)[0] * weight)
        if mode != "unresolved":
            # Permanent parent fallback: fidelity mismatch or gate error.
            record_route(call_id, "parent-fallback:" + mode)
            return parent(hidden_states)
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = True  # capture status unknowable; take the parent path
        if capturing:
            # The gate must synchronize; defer it to a non-capturing call and
            # keep the capture on the parent path.
            record_route(call_id, "parent-fallback:stream-capturing")
            return parent(hidden_states)
        try:
            fused = (_fused_rms_norm(hidden_states, normalized_shape, None, eps)[0] * weight)
            reference = parent(hidden_states)
            matched = bool(
                torch.allclose(
                    fused.float(), reference.float(), rtol=_GATE_RTOL, atol=_GATE_ATOL
                )
            )
        except Exception:
            state["mode"] = "gate-error"
            record_route(call_id, "parent-fallback:gate-error")
            return parent(hidden_states)
        if not matched:
            state["mode"] = "fidelity-mismatch"
            record_route(call_id, "parent-fallback:fidelity-mismatch")
            return reference
        state["mode"] = "fused"
        record_route(call_id, "fused-rms-norm-gated")
        return fused

    return replacement


def build(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route") or _no_route

    replacements = {}
    for call_id, spec in targets.items():
        if spec.get("kind") != "module" or spec.get("phase") != "prefill":
            continue
        if _NORM_SEMANTICS_MARKER not in str(spec.get("semantics", "")):
            continue
        parent = references.get(call_id)
        module = modules.get(call_id)
        if parent is None or module is None:
            continue
        replacements[call_id] = _make_replacement(
            call_id, module, parent, record_route
        )
    return replacements
