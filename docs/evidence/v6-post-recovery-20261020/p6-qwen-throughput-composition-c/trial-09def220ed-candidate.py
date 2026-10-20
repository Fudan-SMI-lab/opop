"""Pointwise stub-threaded composition for the Qwen3-4B decode (IEEE-exact fusions).

Composes with the accepted eager flat-decode region bundle
trial-0b17ae710c-param-a2600a0f (call 323bcbe006b343648b346af9); it is NOT
replaced: the bundle's module replacement keeps executing every decode step and
keeps contributing its GEMM/mean/sdpa kernels to the current-input result.  The
26 functional replacements below intercept only the small elementwise aten
sites that the bundle itself issues inside one decode step (every selected site
has site_parent 323bcbe006b343648b346af9 in current discovery).

Repair of the recorded failures (inputs/previous-feedback.txt and the supplied
GPU probes): the earlier sibling composition (trial-fa955139a3) collapsed into
fallback-only code through a lazy CUDA-graph gate and its resample self-test
crashed in a silu_mul bits-equal check; earlier decode graph trials read
uninitialized capture-pool memory - this candidate uses no stream capture.  The
gate/up packing failure (trial-9cc583939d) is treated as evidence that GEMM
rewrites are risky here, not as a proven causal story; no GEMM, reduction or
attention call is touched.  The leased draft probe (inputs/draft-cuda-probe.json)
found a real CompilationError in the first RoPE kernel (tl.arange needs a
constexpr bound) and measured the native parent final-add output strides as
contiguous (4096,128,128,1)/(1024,128,128,1) while qc keeps the transposed
stride; both are fixed here, together with a verify-chain key error that made
the add site fall to the parent.  The same probe and the full-project screen
(inputs/draft-project-screen.json) showed the draft's transcendental-in-kernel
paths (in-kernel rsqrt from mean+eps, fused silu exp/div) passing the frozen
thresholds with exact greedy tokens but with full-model max_abs_error 0.125 -
not byte-identical.  Those two mechanisms are therefore REMOVED.  Every fused
kernel that remains executes only IEEE float mul/add/neg with explicit
round-to-nearest-even BF16 materialization exactly where the parent aten chain
materializes a BF16 tensor, so there is no implementation freedom left in the
numeric path.

Remaining mechanisms (all values flow through the real parent op sequence every
invocation; nothing is precomputed for the fixture):

* RMSNorm tails (hidden 2560, q 32x128, k 8x128): the x32*rstd mul site launches
  one Triton kernel materializing BOTH the fp32 product and its RN BF16
  back-cast (the exact intermediate rounding aten performs between mul and
  cast); the to.dtype site returns that BF16 tensor; the rsqrt, entry-cast and
  weight-scale mul sites delegate to the real parent operator serving the real
  computation (no filler calls).  3 kernels -> 2 on the mul/cast pair.
* RoPE chains (q 32-head, k 8-head): the first mul (q*cos) launches one kernel
  writing qc = rnd(q*cos), neg(q2) and cat(neg(q2), q1) with the exact strides
  the aten chain produces (qc transposed, neg/cat contiguous); the neg/cat
  sites return those tensors; the rot*sin site launches a second kernel
  writing rs = rnd(rot*sin) and the final add result rnd(qc + rs) with the BF16
  rounding of the product before the add exactly as aten does; the add site
  returns the materialized final tensor with the measured native contiguous
  stride.  5 kernels -> 2.
* SiLU*up and every other selected site: pure parent delegation.  The removed
  draft mechanisms (unread rsqrt placeholder, silu marker) carried non-final
  values; they are gone, so every value any consumer can observe is a genuine
  parent-equivalent tensor with exact parent metadata.

Every replacement checks the full observed shape/stride/dtype signature and
falls back to the parent operator otherwise.  Chain linking uses object
identity plus base-pointer ties, so a signature-matched but unrelated call
never consumes a stub.

Safety: the fused math is value-independent IEEE arithmetic (single-precision
mul/add with explicit intermediate BF16 rounding), so elementwise exactness
does not depend on sampled values; nevertheless each fused class is checked
chain-by-chain against the live parent op chain with torch.equal for a window
covering every chain of one complete run of the frozen workload (chains recur
with identical values each run).  Any mismatch permanently reverts that class
to the parent path.  This window is a per-workload tripwire, not a general
proof; the authoritative acceptance gate remains the framework's full quality
validation on the real workload.  The workload executes on current inputs on
every invocation.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - worker ships triton 3.7.1
    _HAS_TRITON = False

_BF16 = torch.bfloat16
_F32 = torch.float32

_CALLS = {
    "normQ": {"entry": "64982074b04a84f40ad73d92", "rsqrt": "584665e070b5a215e1fa4b94",
              "mul": "3e156c66a7d31d558cc11e6c", "cast": "c5c1b09fe7159ef8ae9c7f1c",
              "wmul": "6c3fe9e81d96d71970c56b67"},
    "normK": {"entry": "31ded45f498a7cf9601f1e78", "rsqrt": "30cdd42f81b8e56c096b67b2",
              "mul": "02ccb458eafde687e522133f", "cast": "12c5763786e09821589d2117",
              "wmul": "c55af535d21f1b163ae3146d"},
    "normH": {"entry": None, "rsqrt": "6a995bb660cd764d2e70cd57",
              "mul": "e4b9fadd2a6698840604b50c", "cast": "1c3eb47689c2400d7251e0f5",
              "wmul": "caf8f9a16b3d5b823ea3fea5"},
    "ropeQ": {"mul1": "d4b0c8808dad80395ce7f9e0", "neg": "730101c557e7bc4313ca7b1c",
              "cat": "c5a25061f382d8c26099bb7f", "mul2": "8665785ccf438a11f52c1493",
              "add": "7c1c33c7f55e62260ea04f57"},
    "ropeK": {"mul1": "a612b60ede69f4e972cdce5a", "neg": "dc2892f1b28df526f6d7d60e",
              "cat": "309621a17042e4cb5cd134a9", "mul2": "9330a63e740ff99f49add9c5",
              "add": "95e0c281309006297e495608"},
    "silu": {"silu": "7fa59f62f13b4eff2ff79848", "mul": "ed7b0f03b3fd6455733fbbfd"},
}

_VERIFY_N = {"normQ": 1200, "normK": 1200, "normH": 2300,
             "ropeQ": 1200, "ropeK": 1200}

if _HAS_TRITON:

    @triton.jit
    def _k_norm(X32, RST, XN, TB, C, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cb = tl.program_id(1)
        offs = cb * BLOCK + tl.arange(0, BLOCK)
        m = offs < C
        rstd = tl.load(RST + row)
        x = tl.load(X32 + row * C + offs, mask=m, other=0.0)
        xn = x * rstd
        tl.store(XN + row * C + offs, xn, mask=m)
        tl.store(TB + row * C + offs, xn.to(tl.bfloat16), mask=m)

    @triton.jit
    def _k_rope1(Q, COS, QC, N2, ROT, D: tl.constexpr, DH: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, D)
        q = tl.load(Q + row * D + d).to(tl.float32)
        c = tl.load(COS + d).to(tl.float32)
        tl.store(QC + row * D + d, (q * c).to(tl.bfloat16))
        d2 = tl.arange(0, DH)
        q1 = tl.load(Q + row * D + d2)
        q2 = tl.load(Q + row * D + DH + d2).to(tl.float32)
        n2 = (-q2).to(tl.bfloat16)
        tl.store(N2 + row * DH + d2, n2)
        tl.store(ROT + row * D + d2, n2)
        tl.store(ROT + row * D + DH + d2, q1)

    @triton.jit
    def _k_rope2(ROT, SIN, QC, RS, FIN, D: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, D)
        r = tl.load(ROT + row * D + d).to(tl.float32)
        s = tl.load(SIN + d).to(tl.float32)
        rs = (r * s).to(tl.bfloat16)
        tl.store(RS + row * D + d, rs)
        qv = tl.load(QC + row * D + d).to(tl.float32)
        tl.store(FIN + row * D + d, (qv + rs.to(tl.float32)).to(tl.bfloat16))


def _sig_ok(t, shape, stride, dtype):
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.dtype is dtype
        and tuple(t.shape) == shape
        and tuple(t.stride()) == stride
    )


def _alloc(shape, stride, dtype, device):
    n = 1
    for i in range(len(shape)):
        n = max(n, shape[i] * stride[i], 1 + (shape[i] - 1) * stride[i])
    flat = torch.empty(n, dtype=dtype, device=device)
    return flat.as_strided(shape, stride)


def _first_tensor(args, kwargs):
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    v = kwargs.get("self")
    return v if isinstance(v, torch.Tensor) else None


def _two_tensors(args, kwargs):
    if len(args) >= 2:
        return args[0], args[1]
    return kwargs.get("self"), kwargs.get("other")


def _to_pair(args, kwargs):
    t = None
    dt = kwargs.get("dtype")
    for a in args:
        if isinstance(a, torch.Tensor):
            if t is None:
                t = a
        elif isinstance(a, torch.dtype):
            if dt is None:
                dt = a
    return t, dt


def _cat_pair(args, kwargs):
    lst = args[0] if args else kwargs.get("tensors")
    if "dim" in kwargs:
        dim = kwargs["dim"]
    elif len(args) > 1:
        dim = args[1]
    else:
        dim = 0
    return lst, dim


def build(context):
    if not _HAS_TRITON:
        return {}
    context = context or {}
    refs = context.get("references") or {}
    route0 = context.get("record_route")
    route = route0 if callable(route0) else (lambda *_: None)

    cls_refs = {}
    for cls, roles in _CALLS.items():
        r = {}
        for role, cid in roles.items():
            if cid is None:
                continue
            ref = refs.get(cid)
            if not callable(ref):
                r = None
                break
            r[role] = ref
        if r:
            cls_refs[cls] = r
    if not {"normQ", "normK", "normH", "ropeQ", "ropeK"} <= set(cls_refs):
        return {}

    S = {
        "route": route,
        "cls_refs": cls_refs,
        "ver": {cls: 0 for cls in cls_refs},
        "vleft": {cls: _VERIFY_N.get(cls, 1200) for cls in cls_refs},
        "norm": {cls: None for cls in cls_refs if cls.startswith("norm")},
        "rope": {cls: None for cls in cls_refs if cls.startswith("rope")},
    }

    def _countdown(cls, ok):
        if not ok:
            S["ver"][cls] = 3
            return
        S["vleft"][cls] -= 1
        if S["vleft"][cls] <= 0:
            S["ver"][cls] = 2

    out = {}

    def _add_norm_class(cls, x_s, x_st, r_s, r_st, w_s, rows, cols, block):
        r = S["cls_refs"][cls]
        c_entry = r.get("entry")
        c_rsqrt, c_mul, c_cast, c_wmul = r["rsqrt"], r["mul"], r["cast"], r["wmul"]
        cid_entry = _CALLS[cls]["entry"]
        cid_rsqrt = _CALLS[cls]["rsqrt"]
        cid_mul = _CALLS[cls]["mul"]
        cid_cast = _CALLS[cls]["cast"]
        cid_wmul = _CALLS[cls]["wmul"]
        grid = (rows, (cols + block - 1) // block)

        def entry_repl(*args, **kwargs):
            try:
                t, dt = _to_pair(args, kwargs)
                if _sig_ok(t, x_s, x_st, _BF16) and dt is _F32:
                    route(cid_entry, "entry_delegate")
                    return c_entry(*args, **kwargs)
                route(cid_entry, "entry_sig_parent")
            except Exception:
                route(cid_entry, "entry_runtime_parent")
            return c_entry(*args, **kwargs)

        def rsqrt_repl(*args, **kwargs):
            try:
                t = _first_tensor(args, kwargs)
                if not _sig_ok(t, r_s, r_st, _F32):
                    route(cid_rsqrt, "sig_parent")
                    return c_rsqrt(*args, **kwargs)
                res = c_rsqrt(t)
                S["norm"][cls] = {"rstd": res, "xn": None, "tb": None}
                route(cid_rsqrt, "rsqrt_delegate")
                return res
            except Exception:
                S["ver"][cls] = 3
                S["norm"][cls] = None
                route(cid_rsqrt, "runtime_parent")
                return c_rsqrt(*args, **kwargs)

        def mul_repl(*args, **kwargs):
            try:
                a, b = _two_tensors(args, kwargs)
                ch = S["norm"].get(cls)
                if not _sig_ok(a, x_s, x_st, _F32) or not _sig_ok(b, r_s, r_st, _F32):
                    S["norm"][cls] = None
                    route(cid_mul, "sig_parent")
                    return c_mul(*args, **kwargs)
                linked = ch is not None and b is ch.get("rstd")
                st = S["ver"][cls]
                if st == 2 and linked:
                    xn = torch.empty(x_s, dtype=_F32, device=a.device)
                    tb = torch.empty(x_s, dtype=_BF16, device=a.device)
                    _k_norm[grid](a, b, xn, tb, cols, BLOCK=block)
                    ch["xn"] = xn
                    ch["tb"] = tb
                    route(cid_mul, "norm_fused")
                    return xn
                if linked and st in (0, 1):
                    p_xn = c_mul(a, b)
                    p_tb = c_cast(p_xn, _BF16)
                    f_xn = torch.empty(x_s, dtype=_F32, device=a.device)
                    f_tb = torch.empty(x_s, dtype=_BF16, device=a.device)
                    _k_norm[grid](a, b, f_xn, f_tb, cols, BLOCK=block)
                    ok = torch.equal(f_xn, p_xn) and torch.equal(f_tb, p_tb)
                    _countdown(cls, ok)
                    ch["xn"] = p_xn
                    ch["tb"] = p_tb
                    route(cid_mul, "norm_verify" if ok else "norm_verify_fail")
                    return p_xn
                route(cid_mul, "norm_parent")
                return c_mul(a, b)
            except Exception:
                S["ver"][cls] = 3
                route(cid_mul, "runtime_parent")
                return c_mul(*args, **kwargs)

        def cast_repl(*args, **kwargs):
            try:
                t, dt = _to_pair(args, kwargs)
                if dt is _BF16 and _sig_ok(t, x_s, x_st, _F32):
                    ch = S["norm"].get(cls)
                    if ch is not None and t is ch.get("xn") and ch.get("tb") is not None:
                        S["norm"][cls] = None
                        route(cid_cast, "cast_stub")
                        return ch["tb"]
                route(cid_cast, "cast_parent")
                return c_cast(*args, **kwargs)
            except Exception:
                route(cid_cast, "runtime_parent")
                return c_cast(*args, **kwargs)

        def wmul_repl(*args, **kwargs):
            try:
                a, b = _two_tensors(args, kwargs)
                if _sig_ok(a, w_s, (1,), _BF16) and _sig_ok(b, x_s, x_st, _BF16):
                    route(cid_wmul, "wmul_delegate")
                    return c_wmul(a, b)
                route(cid_wmul, "sig_parent")
                return c_wmul(*args, **kwargs)
            except Exception:
                route(cid_wmul, "runtime_parent")
                return c_wmul(*args, **kwargs)

        if c_entry is not None:
            out[cid_entry] = entry_repl
        out[cid_rsqrt] = rsqrt_repl
        out[cid_mul] = mul_repl
        out[cid_cast] = cast_repl
        out[cid_wmul] = wmul_repl

    def _add_rope_class(cls, q_s, q_st, cs_s, cs_st, negi_s, negi_st,
                        n2_s, n2_st, cat1_s, cat1_st, rot_s, rot_st,
                        qc_s, qc_st, rows):
        r = S["cls_refs"][cls]
        c_mul1, c_neg, c_cat, c_mul2, c_add = (
            r["mul1"], r["neg"], r["cat"], r["mul2"], r["add"])
        cid_mul1 = _CALLS[cls]["mul1"]
        cid_neg = _CALLS[cls]["neg"]
        cid_cat = _CALLS[cls]["cat"]
        cid_mul2 = _CALLS[cls]["mul2"]
        cid_add = _CALLS[cls]["add"]
        half_bytes = 2 * (q_s[-1] // 2)
        grid = (rows,)

        def _fresh(q):
            qc = _alloc(qc_s, qc_st, _BF16, q.device)
            n2 = _alloc(n2_s, n2_st, _BF16, q.device)
            rot = torch.empty(rot_s, dtype=_BF16, device=q.device)
            return qc, n2, rot

        def mul1_repl(*args, **kwargs):
            try:
                a, b = _two_tensors(args, kwargs)
                if not (_sig_ok(a, q_s, q_st, _BF16) and _sig_ok(b, cs_s, cs_st, _BF16)):
                    route(cid_mul1, "sig_parent")
                    return c_mul1(*args, **kwargs)
                st = S["ver"][cls]
                if st == 2:
                    qc, n2, rot = _fresh(a)
                    _k_rope1[grid](a, b, qc, n2, rot, D=128, DH=64)
                    S["rope"][cls] = {"q": a, "qptr": a.data_ptr(), "qc": qc,
                                      "n2": n2, "rot": rot, "rs": None,
                                      "final": None, "verify": False}
                    route(cid_mul1, "rope_part1")
                    return qc
                if st in (0, 1):
                    p_qc = c_mul1(a, b)
                    f_qc, f_n2, f_rot = _fresh(a)
                    _k_rope1[grid](a, b, f_qc, f_n2, f_rot, D=128, DH=64)
                    S["ver"][cls] = 1
                    S["rope"][cls] = {"q": a, "qptr": a.data_ptr(), "qc": p_qc,
                                      "n2": f_n2, "rot": f_rot, "f_qc": f_qc,
                                      "f_n2": f_n2, "f_rot": f_rot,
                                      "p_qc": p_qc, "rs": None, "final": None,
                                      "verify": True}
                    route(cid_mul1, "rope_verify1")
                    return p_qc
                route(cid_mul1, "rope_parent")
                return c_mul1(a, b)
            except Exception:
                S["ver"][cls] = 3
                S["rope"][cls] = None
                route(cid_mul1, "runtime_parent")
                return c_mul1(*args, **kwargs)

        def neg_repl(*args, **kwargs):
            try:
                t = _first_tensor(args, kwargs)
                if _sig_ok(t, negi_s, negi_st, _BF16):
                    ch = S["rope"].get(cls)
                    if ch is not None and t.data_ptr() == ch["qptr"] + half_bytes:
                        if ch.get("verify"):
                            res = c_neg(t)
                            ch["p_n2"] = res
                            route(cid_neg, "rope_verify_neg")
                            return res
                        if ch.get("n2") is not None:
                            route(cid_neg, "rope_stub_neg")
                            return ch["n2"]
                route(cid_neg, "sig_parent")
                return c_neg(*args, **kwargs)
            except Exception:
                route(cid_neg, "runtime_parent")
                return c_neg(*args, **kwargs)

        def cat_repl(*args, **kwargs):
            try:
                lst, dim = _cat_pair(args, kwargs)
                if (
                    isinstance(lst, (list, tuple))
                    and len(lst) == 2
                    and dim == -1
                    and _sig_ok(lst[0], n2_s, n2_st, _BF16)
                    and _sig_ok(lst[1], cat1_s, cat1_st, _BF16)
                ):
                    ch = S["rope"].get(cls)
                    if ch is not None and lst[1].data_ptr() == ch["qptr"]:
                        if ch.get("verify") and lst[0] is ch.get("p_n2"):
                            res = c_cat(lst, -1)
                            ch["p_rot"] = res
                            route(cid_cat, "rope_verify_cat")
                            return res
                        if not ch.get("verify") and lst[0] is ch.get("n2"):
                            route(cid_cat, "rope_stub_cat")
                            return ch["rot"]
                route(cid_cat, "sig_parent")
                return c_cat(*args, **kwargs)
            except Exception:
                route(cid_cat, "runtime_parent")
                return c_cat(*args, **kwargs)

        def mul2_repl(*args, **kwargs):
            try:
                a, b = _two_tensors(args, kwargs)
                if _sig_ok(a, rot_s, rot_st, _BF16) and _sig_ok(b, cs_s, cs_st, _BF16):
                    ch = S["rope"].get(cls)
                    if ch is not None and a is ch.get("rot") and not ch.get("verify"):
                        rs = torch.empty(rot_s, dtype=_BF16, device=a.device)
                        fin = torch.empty(rot_s, dtype=_BF16, device=a.device)
                        _k_rope2[grid](ch["rot"], b, ch["qc"], rs, fin, D=128)
                        ch["rs"] = rs
                        ch["final"] = fin
                        route(cid_mul2, "rope_part2")
                        return rs
                    if ch is not None and ch.get("verify") and a is ch.get("p_rot"):
                        p_rs = c_mul2(a, b)
                        ch["p_rs"] = p_rs
                        f_rs = torch.empty(rot_s, dtype=_BF16, device=a.device)
                        f_fin = torch.empty(rot_s, dtype=_BF16, device=a.device)
                        _k_rope2[grid](ch["rot"], b, ch["f_qc"], f_rs, f_fin, D=128)
                        ch["f_rs"] = f_rs
                        ch["f_fin"] = f_fin
                        route(cid_mul2, "rope_verify2")
                        return p_rs
                route(cid_mul2, "sig_parent")
                return c_mul2(*args, **kwargs)
            except Exception:
                S["ver"][cls] = 3
                route(cid_mul2, "runtime_parent")
                return c_mul2(*args, **kwargs)

        def add_repl(*args, **kwargs):
            try:
                a, b = _two_tensors(args, kwargs)
                alpha_ok = kwargs.get("alpha", 1) == 1 and (
                    len(args) <= 2 or (len(args) == 3 and args[2] == 1)
                )
                if (
                    _sig_ok(a, qc_s, qc_st, _BF16)
                    and _sig_ok(b, rot_s, rot_st, _BF16)
                    and alpha_ok
                ):
                    ch = S["rope"].get(cls)
                    if (
                        ch is not None
                        and ch.get("verify")
                        and a is ch.get("p_qc")
                        and b is ch.get("p_rs")
                    ):
                        p_fin = c_add(a, b)
                        ok = (
                            torch.equal(ch["f_qc"], a)
                            and ch.get("p_n2") is not None
                            and torch.equal(ch["f_n2"], ch["p_n2"])
                            and ch.get("p_rot") is not None
                            and torch.equal(ch["f_rot"], ch["p_rot"])
                            and ch.get("f_rs") is not None
                            and torch.equal(ch["f_rs"], b)
                            and torch.equal(ch["f_fin"], p_fin)
                        )
                        _countdown(cls, ok)
                        S["rope"][cls] = None
                        route(cid_add, "rope_verify_add" if ok else "rope_verify_fail")
                        return p_fin
                    if ch is not None and a is ch.get("qc") and b is ch.get("rs"):
                        S["rope"][cls] = None
                        route(cid_add, "rope_stub_add")
                        return ch["final"]
                route(cid_add, "sig_parent")
                return c_add(*args, **kwargs)
            except Exception:
                S["ver"][cls] = 3
                route(cid_add, "runtime_parent")
                return c_add(*args, **kwargs)

        out[cid_mul1] = mul1_repl
        out[cid_neg] = neg_repl
        out[cid_cat] = cat_repl
        out[cid_mul2] = mul2_repl
        out[cid_add] = add_repl

    def _add_silu_class():
        r = S["cls_refs"]["silu"]
        c_silu, c_mul = r["silu"], r["mul"]
        cid_silu = _CALLS["silu"]["silu"]
        cid_mul = _CALLS["silu"]["mul"]
        s_s, s_st = (8, 1, 9728), (9728, 9728, 1)

        def silu_repl(*args, **kwargs):
            try:
                t = _first_tensor(args, kwargs)
                if _sig_ok(t, s_s, s_st, _BF16):
                    route(cid_silu, "silu_delegate")
                    return c_silu(*args, **kwargs)
                route(cid_silu, "sig_parent")
                return c_silu(*args, **kwargs)
            except Exception:
                route(cid_silu, "runtime_parent")
                return c_silu(*args, **kwargs)

        def mul_repl(*args, **kwargs):
            try:
                a, b = _two_tensors(args, kwargs)
                if _sig_ok(a, s_s, s_st, _BF16) and _sig_ok(b, s_s, s_st, _BF16):
                    route(cid_mul, "silu_mul_delegate")
                    return c_mul(a, b)
                route(cid_mul, "sig_parent")
                return c_mul(*args, **kwargs)
            except Exception:
                route(cid_mul, "runtime_parent")
                return c_mul(*args, **kwargs)

        out[cid_silu] = silu_repl
        out[cid_mul] = mul_repl

    _add_norm_class("normQ", (8, 1, 32, 128), (4096, 4096, 128, 1),
                    (8, 1, 32, 1), (32, 32, 1, 1), (128,), 256, 128, 128)
    _add_norm_class("normK", (8, 1, 8, 128), (1024, 1024, 128, 1),
                    (8, 1, 8, 1), (8, 8, 1, 1), (128,), 64, 128, 128)
    _add_norm_class("normH", (8, 1, 2560), (2560, 2560, 1),
                    (8, 1, 1), (1, 1, 1), (2560,), 8, 2560, 512)
    _add_rope_class(
        "ropeQ",
        (8, 32, 1, 128), (4096, 128, 4096, 1),
        (1, 1, 1, 128), (5120, 128, 128, 1),
        (8, 32, 1, 64), (4096, 128, 4096, 1),
        (8, 32, 1, 64), (2048, 64, 2048, 1),
        (8, 32, 1, 64), (4096, 128, 4096, 1),
        (8, 32, 1, 128), (4096, 128, 128, 1),
        (8, 32, 1, 128), (4096, 128, 4096, 1),
        256,
    )
    _add_rope_class(
        "ropeK",
        (8, 8, 1, 128), (1024, 128, 1024, 1),
        (1, 1, 1, 128), (5120, 128, 128, 1),
        (8, 8, 1, 64), (1024, 128, 1024, 1),
        (8, 8, 1, 64), (512, 64, 512, 1),
        (8, 8, 1, 64), (1024, 128, 1024, 1),
        (8, 8, 1, 128), (1024, 128, 128, 1),
        (8, 8, 1, 128), (1024, 128, 1024, 1),
        64,
    )
    _add_silu_class()

    return out
