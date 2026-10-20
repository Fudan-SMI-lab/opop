"""CUDA-graph follow-up candidate.

Replaces the installed composite bundle trial-678b0307f5-param-9fc9967a at the
whole-forward 'model' boundary (call 80ff8693d7ac3cf2433b10f2).  The accepted
composite pipeline (Triton scan/scatter/segment-mean + split addmm GEMMs, taken
verbatim from the parent implementation) is executed per invocation through a
CUDA graph that is captured lazily during an actual adapter invocation on the
current inputs, never at build() time.  Replay re-executes the identical
9-kernel chain from the live input/weight buffers on every invocation; guards
on input/weight identity trigger recapture or fall back to the eager composite
/ parent, so the workload always executes on current inputs.
"""

import torch
import torch.nn.functional as F

MODEL_CALL_ID = "80ff8693d7ac3cf2433b10f2"

_ARG_NAMES = ("features", "sources", "destinations", "degree")
_SCAN_MAX_BLOCK = 8192
_MAX_CAPTURE_ATTEMPTS = 3

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _scan_kernel(
        degree_ptr, counts_ptr, starts_ptr, ends_ptr, N, BLOCK_N: tl.constexpr
    ):
        offs = tl.arange(0, BLOCK_N)
        m = offs < N
        deg = tl.load(degree_ptr + offs, mask=m, other=0.0).to(tl.int32)
        ends = tl.cumsum(deg, axis=0)
        starts = ends - deg
        tl.store(counts_ptr + offs, deg, mask=m)
        tl.store(ends_ptr + offs, ends, mask=m)
        tl.store(starts_ptr + offs, starts, mask=m)

    @triton.jit
    def _scatter_kernel(
        sources_ptr, dest_ptr, cursor_ptr, src_sorted_ptr, N, E, BLOCK_E: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_E + tl.arange(0, BLOCK_E)
        m = offs < E
        d = tl.load(dest_ptr + offs, mask=m, other=0).to(tl.int32)
        ok = m & (d >= 0) & (d < N)
        p = tl.atomic_add(cursor_ptr + d, 1, mask=ok)
        s = tl.load(sources_ptr + offs, mask=m, other=0)
        tl.store(src_sorted_ptr + p, s.to(tl.int32), mask=ok & (p >= 0) & (p < E))

    @triton.jit
    def _segmean_kernel(
        x_ptr,
        src_sorted_ptr,
        counts_ptr,
        ends_ptr,
        out_ptr,
        E,
        K: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        n = tl.program_id(0)
        deg = tl.load(counts_ptr + n)
        end = tl.minimum(tl.load(ends_ptr + n), E)
        start = tl.maximum(end - deg, 0)
        cols = tl.arange(0, K)
        acc = tl.zeros([K], dtype=tl.float32)
        for p0 in range(start, end, BLOCK_P):
            offs = p0 + tl.arange(0, BLOCK_P)
            pm = offs < end
            s = tl.load(src_sorted_ptr + offs, mask=pm, other=0)
            vals = tl.load(
                x_ptr + s[:, None] * K + cols[None, :], mask=pm[:, None], other=0.0
            )
            acc += tl.sum(vals, axis=0)
        degf = tl.maximum(deg.to(tl.float32), 1.0)
        tl.store(out_ptr + n * K + cols, acc / degf)


def _segmean_warps(k, block_p):
    return max(2, min(8, (block_p * k) // 512))


def _bind_args(args, kwargs):
    if len(args) > 4:
        return None
    if len(args) == 4 and kwargs:
        return None
    vals = list(args)
    for i in range(len(vals), 4):
        name = _ARG_NAMES[i]
        if name not in kwargs:
            return None
        vals.append(kwargs[name])
    if set(kwargs) - set(_ARG_NAMES[:4]):
        return None
    return vals


def _eager_reference(features, sources, destinations, degree, w1, b1, w2, b2):
    def layer(x, w, b):
        messages = x.index_select(0, sources)
        nb = torch.zeros_like(x).index_add_(0, destinations, messages)
        nb = nb / degree.clamp_min(1).unsqueeze(1)
        return F.linear(torch.cat((x, nb), dim=1), w, b)

    hidden = torch.relu(layer(features, w1, b1))
    return layer(hidden, w2, b2)


def _pipeline(features, sources, destinations, degree, st, block_n_override=None):
    N = features.shape[0]
    E = sources.shape[0]
    dev = features.device
    block_n = block_n_override or triton.next_power_of_2(N)
    if block_n <= _SCAN_MAX_BLOCK:
        counts = torch.empty(N, dtype=torch.int32, device=dev)
        ends = torch.empty(N, dtype=torch.int32, device=dev)
        starts = torch.empty(N, dtype=torch.int32, device=dev)
        nw = min(8, max(1, block_n // 1024))
        st["scan"][(1,)](
            degree, counts, starts, ends, N, BLOCK_N=block_n, num_warps=nw
        )
    else:
        counts = degree.to(torch.int32)
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
    src_sorted = torch.empty(E, dtype=torch.int32, device=dev)
    be = st["block_e"]
    if E > 0:
        st["scatter"][(triton.cdiv(E, be),)](
            sources,
            destinations,
            starts,
            src_sorted,
            N,
            E,
            BLOCK_E=be,
            num_warps=min(8, max(1, be // 128)),
        )
    k1 = st["k1"]
    k2 = st["k2"]
    bp = st["block_p"]
    nb1 = torch.empty(N, k1, dtype=torch.float32, device=dev)
    st["segmean"][(N,)](
        features,
        src_sorted,
        counts,
        ends,
        nb1,
        E,
        K=k1,
        BLOCK_P=bp,
        num_warps=_segmean_warps(k1, bp),
    )
    hidden = torch.addmm(st["b1"], features, st["Wa1"])
    hidden.addmm_(nb1, st["Wb1"])
    hidden.relu_()
    nb2 = torch.empty(N, k2, dtype=torch.float32, device=dev)
    st["segmean"][(N,)](
        hidden,
        src_sorted,
        counts,
        ends,
        nb2,
        E,
        K=k2,
        BLOCK_P=bp,
        num_warps=_segmean_warps(k2, bp),
    )
    out = torch.addmm(st["b2"], hidden, st["Wa2"])
    out.addmm_(nb2, st["Wb2"])
    return out


def _compatible(features, sources, destinations, degree, st):
    k1 = st["k1"]
    N = features.shape[0]
    if not features.is_cuda:
        return False
    if features.ndim != 2 or features.dtype is not torch.float32:
        return False
    if features.shape[1] != k1 or features.stride(0) != k1 or features.stride(1) != 1:
        return False
    if N <= 0:
        return False
    dev = features.device
    for t in (sources, destinations):
        if t.device != dev or t.ndim != 1 or t.dtype is not torch.int64:
            return False
        if t.stride(0) != 1:
            return False
    if sources.shape[0] != destinations.shape[0]:
        return False
    if degree.device != dev or degree.ndim != 1 or degree.dtype is not torch.float32:
        return False
    if degree.stride(0) != 1 or degree.shape[0] != N:
        return False
    return True


def _signature_nodes(context):
    try:
        spec = context["targets"][MODEL_CALL_ID]["signatures"][0]
        return int(spec["args"][0]["shape"][0])
    except Exception:
        return None


def _input_key(features, sources, destinations, degree, st):
    return (
        features.device.index,
        tuple(features.shape),
        sources.shape[0],
        features.data_ptr(),
        sources.data_ptr(),
        destinations.data_ptr(),
        degree.data_ptr(),
        st["w1"].data_ptr(),
        st["b1"].data_ptr(),
        st["w2"].data_ptr(),
        st["b2"].data_ptr(),
    )


def build(context):
    record_route = context.get("record_route", lambda *_: None)
    references = context.get("references") or {}
    modules = context.get("modules") or {}
    parent = references.get(MODEL_CALL_ID)
    if parent is None:
        return {}
    params = context.get("parameters") or {}
    try:
        block_p = int(params.get("reduce_block_p", 2))
    except Exception:
        block_p = 2
    if block_p not in (2, 4, 8, 16):
        block_p = 2
    try:
        block_e = int(params.get("scatter_block_e", 512))
    except Exception:
        block_e = 512
    if block_e not in (256, 512, 1024, 2048):
        block_e = 512

    st = {
        "ok": False,
        "why": "setup:unknown",
        "graph": None,
        "static_out": None,
        "graph_key": None,
        "graph_refs": None,
        "attempts": 0,
    }

    if not _HAS_TRITON:
        st["why"] = "setup:no-triton"
    else:
        try:
            module = modules.get(MODEL_CALL_ID)
            if module is None:
                raise RuntimeError("no-module")
            w1 = module.first.project.weight
            b1 = module.first.project.bias
            w2 = module.second.project.weight
            b2 = module.second.project.bias
            hidden = w1.shape[0]
            k1 = w1.shape[1] // 2
            k2 = w2.shape[1] // 2
            if w1.shape[1] != 2 * k1 or w2.shape[1] != 2 * k2:
                raise ValueError("odd-projection-width")
            if k2 != hidden:
                raise ValueError("hidden-mismatch")
            if w1.dtype is not torch.float32 or w2.dtype is not torch.float32:
                raise ValueError("dtype")
            if w1.device.type != "cuda":
                raise ValueError("non-cuda-module")
            st.update(
                w1=w1,
                b1=b1,
                w2=w2,
                b2=b2,
                k1=k1,
                k2=k2,
                Wa1=w1[:, :k1].t(),
                Wb1=w1[:, k1:].t(),
                Wa2=w2[:, :k2].t(),
                Wb2=w2[:, k2:].t(),
                block_p=block_p,
                block_e=block_e,
                scan=_scan_kernel,
                scatter=_scatter_kernel,
                segmean=_segmean_kernel,
            )
            n_sig = _signature_nodes(context) or 4096
            st["n_sig"] = n_sig
            gen = torch.Generator().manual_seed(7)
            dev = w1.device
            f0 = torch.randn(32, k1, generator=gen).to(dev)
            s0 = torch.randint(32, (32,), generator=gen).to(dev)
            d0 = torch.randint(32, (32,), generator=gen).to(dev)
            g0 = torch.bincount(d0, minlength=32).float().to(dev)
            with torch.inference_mode():
                out0 = _pipeline(
                    f0, s0, d0, g0, st, block_n_override=triton.next_power_of_2(n_sig)
                )
                ref0 = _eager_reference(f0, s0, d0, g0, w1, b1, w2, b2)
                torch.testing.assert_close(out0, ref0, atol=1e-5, rtol=1e-4)
                torch.cuda.synchronize(dev)
            del f0, s0, d0, g0, out0, ref0
            st["ok"] = True
            st["why"] = "ok"
        except Exception as exc:
            st["why"] = "setup:" + type(exc).__name__

    def _release_graph():
        graph = st.get("graph")
        st["graph"] = None
        st["static_out"] = None
        st["graph_key"] = None
        st["graph_refs"] = None
        if graph is not None:
            try:
                graph.reset()
            except Exception:
                pass

    def _try_capture(features, sources, destinations, degree, key, label):
        st["attempts"] += 1
        _release_graph()
        side = torch.cuda.Stream(device=features.device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _pipeline(features, sources, destinations, degree, st)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            static_out = _pipeline(features, sources, destinations, degree, st)
        graph.replay()
        st["graph"] = graph
        st["static_out"] = static_out
        st["graph_key"] = key
        st["graph_refs"] = (features, sources, destinations, degree)
        record_route(MODEL_CALL_ID, label)
        return static_out.clone()

    def forward(*args, **kwargs):
        if not st["ok"]:
            record_route(MODEL_CALL_ID, "fallback:setup")
            return parent(*args, **kwargs)
        bound = _bind_args(args, kwargs)
        if bound is None:
            record_route(MODEL_CALL_ID, "fallback:signature")
            return parent(*args, **kwargs)
        features, sources, destinations, degree = bound
        if not _compatible(features, sources, destinations, degree, st):
            record_route(MODEL_CALL_ID, "fallback:signature")
            return parent(*args, **kwargs)
        key = _input_key(features, sources, destinations, degree, st)
        graph = st.get("graph")
        if graph is not None and st.get("graph_key") == key:
            try:
                graph.replay()
            except Exception:
                _release_graph()
            else:
                record_route(MODEL_CALL_ID, "graph:replay")
                return st["static_out"].clone()
        if st["attempts"] < _MAX_CAPTURE_ATTEMPTS:
            label = "graph:capture" if st["attempts"] == 0 else "graph:recapture"
            try:
                return _try_capture(
                    features, sources, destinations, degree, key, label
                )
            except Exception:
                _release_graph()
        try:
            out = _pipeline(features, sources, destinations, degree, st)
        except Exception:
            record_route(MODEL_CALL_ID, "fallback:runtime")
            return parent(*args, **kwargs)
        record_route(MODEL_CALL_ID, "fallback:capture")
        return out

    return {MODEL_CALL_ID: forward}
