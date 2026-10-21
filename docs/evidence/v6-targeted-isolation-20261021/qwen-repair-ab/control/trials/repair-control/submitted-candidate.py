"""Bit-exact fused RMSNorm repair for the qwen-ttft selection (joint groups
ef1a650b08785209a023 input/post/final norms, 92f04768733eb2543f5d q_norm,
4a79f0c7bb426e51060b k_norm; all prefill phase).

Measured failure being repaired (inputs/previous-feedback.txt +
inputs/observations/previous-after/candidate_diagnostics-4123e6d6.json):

trial-recovered-19ccfe875f replaced the 145 prefill-phase Qwen3RMSNorm
forwards with aten::_fused_rms_norm(...)[0], gated per site by
torch.allclose(fused, parent, atol=rtol=0.02). Every gate passed (routes
"fused-rms-norm-gated" x1 then "fused-rms-norm"), yet output.logits failed
both continuous steps: 1884096/4861952 elements mismatched (38.8%), max abs
4.09375 - the signature of one flipped greedy token (~step 20) with full
trajectory divergence afterwards. Root cause verified on the worker's own
interpreter (torch 2.13.0+cu129, CPU): _fused_rms_norm multiplies the weight
in fp32 with a single bf16 rounding, while transformers 5.16.1 rounds the
normalized activation to bf16 *before* the bf16 weight multiply; the two
formulas differ by up to 1 bf16 ulp per element (0.0078 at unit scale). A
0.02-tolerance per-layer gate accepts exactly that ulp drift, the drift
propagates through 36 layers into the KV cache and every decode step's
logits, and require_exact_greedy_tokens rejects a single near-tie argmax
flip. The tolerance scale of the gate was the defect, not routing.

Repair: keep the numerically order-sensitive part of the parent chain in
aten (the fp32 square tensor and aten::mean.dim reduction) and fuse only
elementwise tails whose arithmetic is identical operation-for-operation to
aten's (IEEE-754 fp32 multiplies, round-to-nearest-even bf16 casts):

  parent : to.f32, pow, mean.dim, add, rsqrt, mul, to.bf16, wmul  (8 kernels)
  level 1: [triton sq: s=float(x)*float(x)], mean.dim, [triton tail:
           r=rsqrt(m+eps); out=(float(w)*((float(x)*r).to(bf16)).to(f32)).to(bf16)]
                                                                   (3 kernels)
  level 2: triton sq, mean.dim, add, rsqrt, triton tail(r=aten)    (5 kernels)
  level 3: to.f32, pow, mean.dim, add, rsqrt, triton tail          (6 kernels)

Every level is verified per site on the first live invocation with a BITWISE
torch.equal against the parent output on the same input - not a tolerance -
so a site only leaves the parent path when its output bits are identical.
For levels 1/2 the fp32 square tensor is additionally compared bitwise
against the parent's own to.f32+pow result, so a sub-bf16 difference that
bf16 output rounding would absorb cannot slip through the variance path.
The first level that matches bitwise wins (preference is remembered per
normalized size); if none matches, or setup/compile fails, the site routes
permanently to the parent with a concise label. Levels 2/3 keep the aten
rsqrt; level 3 additionally keeps the aten square chain, so the worst case
relies only on the fused tail, whose equivalence to aten
mul->to.bf16->wmul was verified bit-exact on the worker interpreter over
randomized bf16 inputs and values.

Numerics guarantees: s is bit-identical to the parent's fp32 squares (level
3 uses the parent's own to.f32+pow calls; levels 1/2 use one IEEE fp32
multiply per element, verified by the gate), mean.dim runs as the identical
aten call on an identical-shape/stride/dtype contiguous input, add/rsqrt
are the parent's own aten calls (levels 2/3), and the fused tail performs
only correctly-rounded fp32 multiplies and RNE bf16 casts with no fusion
across a rounding boundary (verified bitwise by the gate). Bitwise-equal
prefill activations produce a bitwise-equal KV cache, logits and greedy
tokens, so the frozen quality checks (logits atol=rtol=0.02, exact greedy
tokens, negative control) see the parent's own outputs.

Composition with the installed bundle trial-fc57bca7b9-param-f71a7c8a:
its CUDA-graph capture attempt invokes these sites; while a capture is
active the gate (which must synchronize) is deferred and the call is served
by the parent, so capture records the parent chain. After the gate has
passed, already-compiled Triton kernels launch capture-safely. On this
worker the bundle's capture aborts at the first SDPA call
("No available kernel") and routes to parent-fallback, so the prefill -
including these 145 sites - executes eagerly (measured routes confirm the
norm sites were invoked 5-6 times each).

Decode is untouched: no decode-phase RMSNorm sites exist in targets.json,
the signature guard admits only each site's observed prefill shape/stride/
dtype signatures, and everything else (e.g. [1,1,*] decode shapes) falls
back to the parent. The workload executes on current inputs on every
invocation; nothing fixture-dependent is cached and no parameter is
mutated.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _sq_f32_kernel(x_ptr, s_ptr, numel, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(s_ptr + offs, x * x, mask=mask)

    @triton.jit
    def _rms_tail_kernel(
        x_ptr,
        r_ptr,
        w_ptr,
        out_ptr,
        eps,
        N,
        RSQRT_IN_KERNEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        if RSQRT_IN_KERNEL:
            r = tl.rsqrt(tl.load(r_ptr + row) + eps)
        else:
            r = tl.load(r_ptr + row)
        t = (x * r).to(tl.bfloat16)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        out = (w.to(tl.float32) * t.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + row * N + cols, out, mask=mask)

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False

_SQ_BLOCK = 2048
_MAX_TAIL_BLOCK = 65536
_NORM_SEMANTICS_MARKER = "RMSNorm"

_LEVEL_ORDER: dict[int, list[int]] = {}


def _no_route(*_args, **_kwargs):
    return None


def _next_pow2(value):
    result = 1
    while result < value:
        result *= 2
    return result


def _tail_warps(block):
    if block >= 2048:
        return 8
    if block >= 512:
        return 4
    return 2


def _level_order(n_last):
    return _LEVEL_ORDER.setdefault(n_last, [1, 2, 3])


def _prefer_level(n_last, level):
    order = _level_order(n_last)
    if order and order[0] != level:
        order.remove(level)
        order.insert(0, level)


def _make_runner(weight, eps, n_last):
    tail_block = _next_pow2(n_last)
    tail_warps = _tail_warps(tail_block)

    def _tail(x, r_or_m, out, rsqrt_in_kernel):
        _rms_tail_kernel[(x.numel() // n_last,)](
            x,
            r_or_m,
            weight,
            out,
            eps,
            n_last,
            RSQRT_IN_KERNEL=rsqrt_in_kernel,
            BLOCK=tail_block,
            num_warps=tail_warps,
        )

    def _square(x):
        s = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        _sq_f32_kernel[(triton.cdiv(x.numel(), _SQ_BLOCK),)](
            x, s, x.numel(), BLOCK=_SQ_BLOCK, num_warps=4
        )
        return s

    def run(level, x, intermediates=None):
        if level == 1:
            s = _square(x)
            m = s.mean(-1, keepdim=True)
            out = torch.empty_like(x)
            _tail(x, m, out, True)
            if intermediates is not None:
                intermediates["s"] = s
            return out
        if level == 2:
            s = _square(x)
            m = s.mean(-1, keepdim=True)
            r = torch.rsqrt(m + eps)
            out = torch.empty_like(x)
            _tail(x, r, out, False)
            if intermediates is not None:
                intermediates["s"] = s
            return out
        y = x.to(torch.float32)
        s = y.pow(2)
        m = s.mean(-1, keepdim=True)
        r = torch.rsqrt(m + eps)
        out = torch.empty_like(x)
        _tail(x, r, out, False)
        return out

    return run


def _make_replacement(call_id, module, parent, record_route, allowed_shapes):
    weight = getattr(module, "weight", None)
    eps = getattr(module, "variance_epsilon", None)
    if not (
        isinstance(weight, torch.Tensor)
        and weight.dim() == 1
        and weight.numel() > 0
        and weight.dtype is torch.bfloat16
        and weight.is_contiguous()
        and isinstance(eps, float)
    ):
        def unsupported_module(*args, **kwargs):
            record_route(call_id, "parent-fallback:unsupported-module")
            return parent(*args, **kwargs)

        return unsupported_module

    n_last = weight.shape[0]
    if not _TRITON_AVAILABLE or n_last > _MAX_TAIL_BLOCK:
        label = (
            "parent-fallback:triton-unavailable"
            if not _TRITON_AVAILABLE
            else "parent-fallback:unsupported-module"
        )

        def unavailable(*args, **kwargs):
            record_route(call_id, label)
            return parent(*args, **kwargs)

        return unavailable

    run = _make_runner(weight, eps, n_last)
    state = {"level": 0, "label": ""}

    def _compatible(x):
        return (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and x.dtype is torch.bfloat16
            and x.dim() >= 1
            and x.shape[-1] == n_last
            and x.is_contiguous()
            and weight.device == x.device
            and x.numel() % n_last == 0
            and (not allowed_shapes or tuple(x.shape) in allowed_shapes)
        )

    def replacement(*args, **kwargs):
        level = state["level"]
        if len(args) != 1 or kwargs or not _compatible(args[0]):
            record_route(call_id, "parent-fallback:unsupported-signature")
            return parent(*args, **kwargs)
        x = args[0]
        if level > 0:
            try:
                out = run(level, x)
            except Exception:
                state["level"] = -1
                state["label"] = "runtime-error"
                record_route(call_id, "parent-fallback:runtime-error")
                return parent(x)
            record_route(call_id, "bitexact-l%d" % level)
            return out
        if level < 0:
            record_route(call_id, "parent-fallback:" + state["label"])
            return parent(x)
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            capturing = True
        if capturing:
            record_route(call_id, "parent-fallback:stream-capturing")
            return parent(x)
        reference = parent(x)
        parent_squares = None
        for candidate_level in _level_order(n_last):
            intermediates = {} if candidate_level < 3 else None
            try:
                out = run(candidate_level, x, intermediates)
                matched = (
                    out.shape == reference.shape
                    and out.dtype == reference.dtype
                    and torch.equal(out, reference)
                )
                if matched and intermediates is not None:
                    if parent_squares is None:
                        parent_squares = x.to(torch.float32).pow(2)
                    matched = torch.equal(intermediates["s"], parent_squares)
            except Exception:
                matched = False
            if matched:
                state["level"] = candidate_level
                _prefer_level(n_last, candidate_level)
                record_route(call_id, "bitexact-l%d-gated" % candidate_level)
                return out
        state["level"] = -1
        state["label"] = "bitwise-mismatch"
        record_route(call_id, "parent-fallback:bitwise-mismatch")
        return reference

    return replacement


def build(context):
    targets = context.get("targets") or {}
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    record_route = context.get("record_route", _no_route)

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
        weight = getattr(module, "weight", None)
        n_last = weight.shape[0] if isinstance(weight, torch.Tensor) else None
        allowed_shapes = set()
        for signature in spec.get("signatures") or []:
            args = signature.get("args") or []
            if len(args) == 1 and args[0].get("dtype") == "torch.bfloat16":
                shape = tuple(args[0].get("shape") or ())
                if not n_last or shape[-1] == n_last:
                    allowed_shapes.add(shape)
        replacements[call_id] = _make_replacement(
            call_id, module, parent, record_route, allowed_shapes
        )
    return replacements
