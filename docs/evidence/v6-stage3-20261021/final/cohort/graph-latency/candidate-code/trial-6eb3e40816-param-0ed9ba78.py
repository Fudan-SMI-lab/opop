"""CUDA-Graph-replayed fused CSR mean-GraphSAGE encoder candidate.

Repair of the rejected trial-81bc23a8c7 (inputs/previous-feedback.txt): its
Triton CSR segment-sum kernels passed quality (max_abs_error 1.19e-7 over
262144 elements) and cut diagnostic CUDA time from 321.8 us to 110.75 us, but
wall latency regressed to 0.42-0.51 ms because every invocation paid host
dispatch for the fused path (previous-after python_runtime trace: triton
jit.py run 99.3 us self over 6 launches, cuda_utils launch machinery,
torch.cumsum 37.0 us, torch.zeros 29.6 us, torch.empty 20.4 us; profiled
wall 0.686 -> 1.017 ms).  The binding constraint is per-invocation host
launch cost, not kernel time.

This candidate replaces the whole encoder boundary (group
1f40997fb4a53f41b9da, call 113d899004dbbe9147a54672) so both MeanSAGE
layers, the shared CSR build and the inter-layer ReLU form one unit:

  * mean aggregation via the trial-proven Triton CSR segment-sum kernels
    (destination histogram, inclusive cumsum, stable-in-count edge scatter,
    register-resident segment sum with IEEE-correctly-rounded div.rn.f32 by
    max(degree, 1), zero-neighbor rows for nodes without incoming edges,
    duplicate directed edges counted individually); the CSR structure is
    built once per forward from the live destinations/sources tensors and
    shared by both layers, which aggregate over the identical edge list;
  * each nn.Linear projection consumes a directly assembled
    [features | neighbors] buffer; the projections and the inter-layer
    activation are the captured aten ops (F.linear -> cuBLAS GEMM, relu)
    with the observed shapes/dtypes under TF32-disabled 'highest' policy;
  * the first eligible invocation runs the fused path eagerly (which also
    JIT-compiles Triton), warms the exact op sequence on a capture stream,
    then captures the whole two-layer forward on the live input tensors
    with torch.cuda.CUDAGraph; every subsequent invocation performs one
    graph replay and returns a fresh clone of the static output.  Replay
    re-executes the complete computation (CSR rebuild included) on the
    current buffer contents, so the workload executes on each invocation
    and no fixture-dependent output is cached;
  * signature/data_ptr guards route unusual inputs to the eager fused
    path; setup, capture and runtime failures route to the parent bound
    forward, so output correctness never depends on the fast path.
"""

import torch
import torch.nn.functional as tnf

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False

_BLOCK_HIST = 1024
_BLOCK_SCATTER = 1024
_BLOCK_C = 128
_MAX_GRID_Y = 65535
_INT32_MAX = 2 ** 31 - 1

_ENCODER_CALL_ID = "113d899004dbbe9147a54672"

if _HAS_TRITON:

    @triton.jit
    def _opop_hist_kernel(dst_ptr, counts_ptr, E, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < E
        dst = tl.load(dst_ptr + offs, mask=m, other=0)
        tl.atomic_add(counts_ptr + dst, 1, mask=m, sem="relaxed")

    @triton.jit
    def _opop_scatter_kernel(src_ptr, dst_ptr, counts_ptr, ends_ptr,
                             cursor_ptr, sorted_ptr, E, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < E
        dst = tl.load(dst_ptr + offs, mask=m, other=0)
        src = tl.load(src_ptr + offs, mask=m, other=0).to(tl.int32)
        rank = tl.atomic_add(cursor_ptr + dst, 1, mask=m, sem="relaxed")
        end = tl.load(ends_ptr + dst, mask=m, other=0)
        cnt = tl.load(counts_ptr + dst, mask=m, other=0)
        tl.store(sorted_ptr + (end - cnt + rank), src, mask=m)

    @triton.jit
    def _opop_segsum_kernel(feats_ptr, deg_ptr, counts_ptr, ends_ptr,
                            sorted_ptr, buf_ptr, F: tl.constexpr,
                            BLOCK_C: tl.constexpr):
        cb = tl.program_id(0)
        n = tl.program_id(1)
        c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        out_row = n.to(tl.int64) * (2 * F)
        if cb < (F // BLOCK_C):
            v = tl.load(feats_ptr + n.to(tl.int64) * F + c)
            tl.store(buf_ptr + out_row + c, v)
        else:
            cnt = tl.load(counts_ptr + n)
            end = tl.load(ends_ptr + n)
            deg = tl.load(deg_ptr + n)
            den = tl.zeros([BLOCK_C], dtype=tl.float32) + tl.maximum(deg, 1.0)
            acc = tl.zeros([BLOCK_C], dtype=tl.float32)
            base = end - cnt
            c2 = c - F
            for i in range(0, cnt):
                src = tl.load(sorted_ptr + base + i).to(tl.int64)
                acc += tl.load(feats_ptr + src * F + c2)
            res = tl.math.div_rn(acc, den)
            tl.store(buf_ptr + out_row + c, res)


def _eligible(features, sources, destinations, degree, w1, b1, w2, b2):
    if not _HAS_TRITON:
        return False
    if not (features.is_cuda and features.dim() == 2
            and features.dtype == torch.float32):
        return False
    N, F1 = features.shape
    if N <= 0 or N > _MAX_GRID_Y or F1 <= 0 or F1 % _BLOCK_C != 0:
        return False
    if features.stride(1) != 1 or features.stride(0) != F1:
        return False
    for w, b in ((w1, b1), (w2, b2)):
        if not (w.is_cuda and w.dim() == 2 and w.dtype == torch.float32
                and b.is_cuda and b.dim() == 1
                and b.dtype == torch.float32):
            return False
        if w.stride(1) != 1 or w.stride(0) != w.shape[1]:
            return False
        if b.stride(0) != 1 or b.shape[0] != w.shape[0]:
            return False
    F2 = w1.shape[0]
    if 2 * F1 != w1.shape[1] or F2 % _BLOCK_C != 0 or 2 * F2 != w2.shape[1]:
        return False
    if w1.device != features.device or w2.device != features.device:
        return False
    for t in (sources, destinations, degree):
        if not t.is_cuda or t.dim() != 1 or t.stride(0) != 1:
            return False
        if t.device != features.device:
            return False
    if sources.dtype != torch.int64 or destinations.dtype != torch.int64:
        return False
    if degree.dtype != torch.float32 or degree.numel() != N:
        return False
    E = sources.numel()
    if destinations.numel() != E or E > _INT32_MAX:
        return False
    return True


def _fused_encoder_forward(args, weights):
    features, sources, destinations, degree = args
    w1, b1, w2, b2 = weights
    N, F1 = features.shape
    E = sources.numel()
    dev = features.device
    F2 = w1.shape[0]
    # Shared CSR structure over the destination-derived ordering, built once
    # per forward and reused by both layers.
    ibuf = torch.zeros(3 * N, dtype=torch.int32, device=dev)
    counts = ibuf[:N]
    ends = ibuf[N:2 * N]
    cursor = ibuf[2 * N:]
    if E > 0:
        _opop_hist_kernel[(triton.cdiv(E, _BLOCK_HIST),)](
            destinations, counts, E, BLOCK=_BLOCK_HIST, num_warps=4)
    torch.cumsum(counts, 0, dtype=torch.int32, out=ends)
    sorted_src = torch.empty(E, dtype=torch.int32, device=dev)
    if E > 0:
        _opop_scatter_kernel[(triton.cdiv(E, _BLOCK_SCATTER),)](
            sources, destinations, counts, ends, cursor, sorted_src, E,
            BLOCK=_BLOCK_SCATTER, num_warps=4)
    buf1 = torch.empty((N, 2 * F1), dtype=torch.float32, device=dev)
    _opop_segsum_kernel[(2 * F1 // _BLOCK_C, N)](
        features, degree, counts, ends, sorted_src, buf1, F=F1,
        BLOCK_C=_BLOCK_C, num_warps=1)
    hidden = tnf.linear(buf1, w1, b1)
    hidden = torch.relu(hidden)
    buf2 = torch.empty((N, 2 * F2), dtype=torch.float32, device=dev)
    _opop_segsum_kernel[(2 * F2 // _BLOCK_C, N)](
        hidden, degree, counts, ends, sorted_src, buf2, F=F2,
        BLOCK_C=_BLOCK_C, num_warps=1)
    return tnf.linear(buf2, w2, b2)


def _keys(args, weights):
    specs = []
    for t in args + weights:
        specs.append((tuple(t.shape), t.data_ptr()))
    return tuple(specs)


def _capture(state, args, weights):
    features = args[0]
    stream = torch.cuda.Stream(device=features.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _fused_encoder_forward(args, weights)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_out = _fused_encoder_forward(args, weights)
    state["graph"] = graph
    state["out"] = static_out
    state["keys"] = _keys(args, weights)
    state["stream"] = stream


def _make_replacement(module, first, second, parent, record):
    state = {"graph": None, "out": None, "keys": None, "stream": None,
             "broken": False, "attempted": False}

    def replace_encoder(features, sources, destinations, degree):
        try:
            if (torch.is_grad_enabled() or module.training or first.training
                    or second.training):
                record(_ENCODER_CALL_ID, "fallback:signature")
                return parent(features, sources, destinations, degree)
            p1 = first.project
            p2 = second.project
            w1, b1 = p1.weight, p1.bias
            w2, b2 = p2.weight, p2.bias
            if not _eligible(features, sources, destinations, degree,
                             w1, b1, w2, b2):
                record(_ENCODER_CALL_ID, "fallback:signature")
                return parent(features, sources, destinations, degree)
            args = (features, sources, destinations, degree)
            weights = (w1, b1, w2, b2)
            if state["graph"] is not None:
                if _keys(args, weights) == state["keys"]:
                    state["graph"].replay()
                    out = state["out"].clone()
                    record(_ENCODER_CALL_ID, "fused_csr_graph_replay")
                    return out
                out = _fused_encoder_forward(args, weights)
                record(_ENCODER_CALL_ID, "fused_csr_eager")
                return out
            if state["broken"] or state["attempted"]:
                out = _fused_encoder_forward(args, weights)
                record(_ENCODER_CALL_ID, "fused_csr_eager")
                return out
            out = _fused_encoder_forward(args, weights)
            state["attempted"] = True
            try:
                _capture(state, args, weights)
                record(_ENCODER_CALL_ID, "fused_csr_graph_capture")
            except Exception:
                state["broken"] = True
                state["graph"] = None
                state["out"] = None
                state["keys"] = None
                state["stream"] = None
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                record(_ENCODER_CALL_ID, "fallback:capture")
            return out
        except Exception:
            record(_ENCODER_CALL_ID, "fallback:runtime")
            return parent(features, sources, destinations, degree)

    return replace_encoder


def _make_setup_degraded(parent, record):
    def replace_encoder(features, sources, destinations, degree):
        record(_ENCODER_CALL_ID, "fallback:setup")
        return parent(features, sources, destinations, degree)

    return replace_encoder


def build(context):
    record = context.get("record_route", lambda *_: None)
    modules = context.get("modules", {})
    references = context.get("references", {})
    module = modules.get(_ENCODER_CALL_ID)
    parent = references.get(_ENCODER_CALL_ID)
    if module is None or parent is None:
        return {}
    if not _HAS_TRITON:
        return {_ENCODER_CALL_ID: _make_setup_degraded(parent, record)}
    try:
        first = module.first
        second = module.second
        _ = first.project.weight
        _ = second.project.weight
    except Exception:
        return {_ENCODER_CALL_ID: _make_setup_degraded(parent, record)}
    return {_ENCODER_CALL_ID: _make_replacement(module, first, second,
                                                parent, record)}
