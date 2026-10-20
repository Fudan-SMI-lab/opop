"""Repair candidate for group 0c566a3a363f294c6750 (72 prefill gate_proj/up_proj
aten::linear.default sites, M=256, N=9728, K=2560 bf16 TN).

Previous trial-8ecab37315 was rejected (no_valid_parameter_candidate) because its
hand-written Triton GEMM was numerically wrong on every parameter trial. This
module repairs that failure with two mandatory mechanisms:

1. A canonical row-major TN linear kernel: the [1, 256, 2560] contiguous
   activation is collapsed to A[256, 2560], the weight is consumed directly in
   its native row-major B[9728, 2560] layout as dot(A_tile, B_tile^T) with fp32
   accumulation and a single bf16 rounding at the store. No weight
   transposition, packing, replication or precompute of any kind.

2. An install-time equivalence gate: a tile configuration is activated only
   after it reproduces an fp32 reference within bf16-rounding-scale bounds on
   the exact observed signature, on the current device, over random operand
   sets plus one real module weight. A configuration that fails the gate, an
   illegal configuration, or a missing Triton installation registers guarded
   replacements whose fast path is disabled, so every call falls back to the
   parent operator: a wrong kernel can never reach measurement again.
"""
from __future__ import annotations

import torch

SELECTED_CALLS = (
    "1c152979afa12dd1028378b1",
    "1e3285ee785c5a0c49f5c6d3",
    "214a9efb92c6d0cf43fca86a",
    "21d9ce0486af5a22efb57bd3",
    "27998c76b79cd123f1abe887",
    "28b0adad34d444a2ec6bea50",
    "2f4e7f6b7013a3a4a4f14059",
    "341ea1ebfb19127c61c75b17",
    "34c27963cf3840bd2e8e4d35",
    "3d3a94542fd9e52e82341926",
    "3f661b51860f4d88012ecb3f",
    "4310be8f7fe0dc22a1de55d4",
    "4591185a0aadc7e4f795ff3e",
    "463d2608ce37f0d46453136d",
    "464488b41c20f582480d88e8",
    "4c35941b2dba429ccd6002f5",
    "4d7fd81283c685007c327e24",
    "501df4fd085b34f72f62b682",
    "538035c2fe7be4b1e3616c71",
    "576954be1f0b7e5227a9c493",
    "5ccfc528fb4f1b22fb952473",
    "5e412648827854291e84a743",
    "640e09e9559a9665a48c6983",
    "64e1cb05c4715c314ec73f8c",
    "69afd2184b7b825eb95ad110",
    "71036cb1358cc6ebe8667206",
    "739781a4c2bae5e279950782",
    "748d6b156c365f642ac4d881",
    "7561130ea067bda8f0cce0db",
    "791bfbd975a41c367a0588c8",
    "7bcadf6a22b3dda1b7b7fb19",
    "7c5c770cfdf4ea00ebb29136",
    "7e22e4252f2a704715c3fe0a",
    "7ed2c7d8dbc30aaa8d597286",
    "7eda06f6397fce326ec86401",
    "7fe7748e52c428c8932b9d84",
    "7fedf98ffa4c2c026ddca4be",
    "833a56d15826f9ff0e517344",
    "841c0fc0b6a44e98d83663ec",
    "84306195f01c4a338e2d5745",
    "85b929418199ec380c789115",
    "8b8503f0ee4135801d55425e",
    "9c355ff1a8dfb8caeb3f1cc5",
    "9fa7c349e0ca9df4fdeb7115",
    "a1c07190b38becc3e0583bcb",
    "a5496d793c6fe75468ca0efb",
    "a76e034d73fb1ef8da3af986",
    "b157b1c9efc281fd28d4c651",
    "bdd397cf0ae780917778edc0",
    "be342a31d4a5f061400e274a",
    "cb043c371f102afb518baf48",
    "d0ee06be970d07d0471a9061",
    "d3d2aca50892637dfbbbe395",
    "dc3e9d40abbcaadb8a115348",
    "dca32a951470a8d119e6a946",
    "dff72d0b7184cf1167bfba3f",
    "e0803fc3545919517d70ad1c",
    "e17c0e4328a3b25c54db884d",
    "e3f8f0c3bdd7e2a2e751bb59",
    "e76872452abf9cfa1a71c58f",
    "e9aa24895c670aba8253cf0e",
    "eb1a8da5eba4d4f6946b61b3",
    "eb3dea2a90566ea5f2472eb8",
    "ec61e185cdac47c3341f185f",
    "edab464975cf8b4aec5e32e4",
    "ede72ddb1e958c561e258e22",
    "ede91c385ec66eae79c06a0c",
    "f0c936debbedb18c180c25cb",
    "f12f82645b82416344efd51a",
    "f47c50ad7618ca546011dd7e",
    "f9444b26b52179efff3b81a6",
    "faf83d504b9638b229e91040",
)

_OBS_INPUT_SHAPE = (1, 256, 2560)
_OBS_INPUT_STRIDE = (655360, 2560, 1)
_OBS_WEIGHT_SHAPE = (9728, 2560)
_OBS_WEIGHT_STRIDE = (2560, 1)

_M, _N, _K = 256, 9728, 2560

_DEFAULT_PARAMS = {
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 3,
}

_SMEM_LIMIT = 101376  # sm_89 (RTX 4090) max dynamic shared memory per CTA in bytes
_MIN_CTAS = 128       # device has 128 SMs; require at least one CTA per SM

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - worker without a Triton toolchain
    triton = None
    tl = None


if tl is not None:

    @triton.jit
    def _tn_linear_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, tl.trans(b), acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, c)


def _legal_params(p):
    bm, bn, bk = p["BLOCK_M"], p["BLOCK_N"], p["BLOCK_K"]
    stages = p["num_stages"]
    if min(bm, bn, bk) < 16:
        return False
    if _M % bm or _N % bn or _K % bk:
        return False
    if (bm * bk + bk * bn) * 2 * stages > _SMEM_LIMIT:
        return False
    if (_M // bm) * (_N // bn) < _MIN_CTAS:
        return False
    if p["num_warps"] not in (4, 8):
        return False
    if not 2 <= stages <= 5:
        return False
    if not 1 <= p["GROUP_M"] <= 16:
        return False
    return True


def _make_runner(p):
    bm, bn = p["BLOCK_M"], p["BLOCK_N"]
    grid = ((_M // bm) * (_N // bn),)
    warps, stages = p["num_warps"], p["num_stages"]

    def run(a2d, w):
        out = torch.empty((_M, _N), device=a2d.device, dtype=torch.bfloat16)
        _tn_linear_kernel[grid](
            a2d, w, out,
            _M, _N, _K,
            a2d.stride(0), a2d.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=p["BLOCK_K"],
            GROUP_M=p["GROUP_M"],
            num_warps=warps,
            num_stages=stages,
        )
        return out

    return run


def _normalize_device(device):
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _resolve_device(context):
    state = context.get("state") or {}
    device = state.get("device")
    if isinstance(device, torch.device):
        return _normalize_device(device)
    models = context.get("models") or {}
    for model in models.values():
        for param in model.parameters():
            return _normalize_device(param.device)
    modules = context.get("modules") or {}
    for module in modules.values():
        for param in module.parameters():
            return _normalize_device(param.device)
    return None


def _sample_real_weight(context, ids):
    modules = context.get("modules") or {}
    for cid in ids:
        weight = getattr(modules.get(cid), "weight", None)
        if (
            torch.is_tensor(weight)
            and tuple(weight.shape) == _OBS_WEIGHT_SHAPE
            and weight.dtype is torch.bfloat16
            and weight.is_cuda
        ):
            return weight.detach()
    return None


def _gate_dataset_ok(run, weight, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    a = torch.randn((1, _M, _K), device=device, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        ref = torch.mm(a.view(_M, _K).float(), weight.float().t())
        got = run(a.view(_M, _K), weight).view(1, _M, _N)
    if tuple(got.shape) != (1, _M, _N):
        return False
    if got.dtype is not torch.bfloat16 or got.device != device:
        return False
    if not got.is_contiguous():
        return False
    diff = got.view(_M, _N).float() - ref
    abs_ref = ref.abs()
    tol = 1e-3 + 2e-2 * abs_ref
    frac_bad = (diff.abs() > tol).float().mean().item()
    worst = diff.abs().max().item()
    scale = abs_ref.max().item()
    return frac_bad <= 1e-4 and worst <= 3e-2 * scale + 1e-2


def _equivalence_gate(run, device, real_weight):
    for seed in (0x51CE0001, 0x51CE0002):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        weight = torch.randn(_OBS_WEIGHT_SHAPE, device=device, generator=gen, dtype=torch.float32).to(torch.bfloat16)
        if not _gate_dataset_ok(run, weight, seed ^ 0xA5A5, device):
            return False
    if real_weight is not None:
        if not _gate_dataset_ok(run, real_weight, 0x51CE0003, device):
            return False
    return True


def _parent_call(parent, input, weight, bias):
    if bias is None:
        try:
            return parent(input, weight)
        except TypeError:
            return parent(input, weight, None)
    return parent(input, weight, bias)


def _make_replacement(parent, run, device):
    def _linear(input, weight, bias=None):
        try:
            if (
                run is not None
                and bias is None
                and not torch.is_grad_enabled()
                and input.dim() == 3
                and tuple(input.shape) == _OBS_INPUT_SHAPE
                and tuple(input.stride()) == _OBS_INPUT_STRIDE
                and weight.dim() == 2
                and tuple(weight.shape) == _OBS_WEIGHT_SHAPE
                and tuple(weight.stride()) == _OBS_WEIGHT_STRIDE
                and input.dtype is torch.bfloat16
                and weight.dtype is torch.bfloat16
                and input.device == device
                and weight.device == device
            ):
                return run(input.reshape(_M, _K), weight).view(1, _M, _N)
        except Exception:
            pass
        return _parent_call(parent, input, weight, bias)

    return _linear


def build(context):
    params = dict(_DEFAULT_PARAMS)
    supplied = context.get("parameters") or {}
    for key in _DEFAULT_PARAMS:
        if supplied.get(key) is not None:
            try:
                params[key] = int(supplied[key])
            except (TypeError, ValueError):
                pass

    targets = context.get("targets") or {}
    ids = [cid for cid in SELECTED_CALLS if cid in targets]
    replacements = {}
    if not ids:
        return replacements

    references = context.get("references") or {}
    device = _resolve_device(context)

    run = None
    if tl is not None and device is not None and device.type == "cuda" and _legal_params(params):
        try:
            candidate_run = _make_runner(params)
            if _equivalence_gate(candidate_run, device, _sample_real_weight(context, ids)):
                run = candidate_run
        except Exception:
            run = None

    for cid in ids:
        parent = references.get(cid)
        if not callable(parent):
            parent = torch.nn.functional.linear
        replacements[cid] = _make_replacement(parent, run, device)
    return replacements
