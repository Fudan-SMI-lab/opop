"""Fused mean-aggregation GraphSAGE candidate.

Replaces the two observed MeanSAGE module forwards (encoder.first /
encoder.second) with a CSR segment-sum implementation: per-invocation the
destination histogram, its inclusive prefix scan and a stable-in-count edge
reordering are rebuilt, then one Triton kernel gathers source rows per
destination segment, accumulates them in registers, applies the exact fp32
division by max(degree, 1) and writes the [features | neighbors] projection
input directly.  The nn.Linear project module is still invoked normally, so
the Linear module boundary and its aten::linear functional sites execute with
their observed signatures.
"""

import torch

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

_TARGET_CALL_IDS = (
    "206dbc4bff7ae767415599c9",
    "866013201077b272e7156dd4",
)

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


def _eligible(features, sources, destinations, degree):
    if not _HAS_TRITON:
        return False
    if not (features.is_cuda and features.dim() == 2
            and features.dtype == torch.float32):
        return False
    N, F = features.shape
    if N <= 0 or N > _MAX_GRID_Y:
        return False
    if F % _BLOCK_C != 0:
        return False
    if features.stride(1) != 1 or features.stride(0) != F:
        return False
    for t in (sources, destinations, degree):
        if not t.is_cuda or t.dim() != 1 or t.stride(0) != 1:
            return False
    if sources.dtype != torch.int64 or destinations.dtype != torch.int64:
        return False
    if degree.dtype != torch.float32 or degree.numel() != N:
        return False
    E = sources.numel()
    if destinations.numel() != E or E > _INT32_MAX:
        return False
    return True


def _fused_forward(project, features, sources, destinations, degree):
    N, F = features.shape
    E = sources.numel()
    dev = features.device
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
    buf = torch.empty((N, 2 * F), dtype=torch.float32, device=dev)
    grid = ((2 * F) // _BLOCK_C, N)
    _opop_segsum_kernel[grid](
        features, degree, counts, ends, sorted_src, buf, F=F,
        BLOCK_C=_BLOCK_C, num_warps=1)
    return project(buf)


def _make_replacement(call_id, module, parent, record):
    project = module.project

    def replace_mean_sage(features, sources, destinations, degree):
        try:
            if _eligible(features, sources, destinations, degree):
                out = _fused_forward(project, features, sources,
                                     destinations, degree)
                record(call_id, "triton_csr_segsum")
                return out
        except Exception:
            record(call_id, "fallback:runtime")
            return parent(features, sources, destinations, degree)
        record(call_id, "fallback:signature")
        return parent(features, sources, destinations, degree)

    return replace_mean_sage


def _make_setup_degraded(call_id, parent, record):
    def replace_mean_sage(features, sources, destinations, degree):
        record(call_id, "fallback:setup")
        return parent(features, sources, destinations, degree)

    return replace_mean_sage


def build(context):
    record = context.get("record_route", lambda *_: None)
    modules = context.get("modules", {})
    references = context.get("references", {})
    replacements = {}
    for call_id in _TARGET_CALL_IDS:
        module = modules.get(call_id)
        parent = references.get(call_id)
        if module is None or parent is None:
            continue
        if not _HAS_TRITON or not hasattr(module, "project"):
            replacements[call_id] = _make_setup_degraded(call_id, parent,
                                                         record)
            continue
        replacements[call_id] = _make_replacement(call_id, module, parent,
                                                  record)
    return replacements
