"""Shape-specialized replacement for the selected prefill gate/up projection GEMMs.

Selection (inputs/selection.json): group 0c566a3a363f294c6750 - the 72 prefill
aten::linear.default call sites of Qwen3-4B (one gate_proj and one up_proj per
layer across all 36 layers), each exactly one bf16 TN GEMM with
M=256, N=9728, K=2560 on the observed contiguous layouts.

Replacement: a Triton blocked GEMM that consumes the activation and the weight
in their native row-major layouts (A [256,2560], B [9728,2560], C [256,9728]
bf16 out), accumulating in fp32 with no split-K and no cross-call state. Tile
shape / warps / stages are fixed at build time from context['parameters']
(with safe defaults), so the steady-state call path is one cached kernel launch
per site; the workload executes on every invocation.

Every callable guards the observed signature exactly (shapes, strides, dtype,
device, bias=None, grad mode) and falls back to the parent operator otherwise,
so decode-phase M=1 linears and any unexpected invocation are untouched.
"""
from __future__ import annotations

import torch

try:  # optional dependency; build() returns an empty mapping without it
    import triton
    import triton.language as tl
    _TRITON_OK = True
except Exception:  # pragma: no cover - worker image is expected to ship Triton
    triton = None
    tl = None
    _TRITON_OK = False


# Frozen observed signature of every selected call site (targets.json).
_M, _N, _K = 256, 9728, 2560
_IN_SHAPE = (1, _M, _K)
_IN_STRIDES = (_M * _K, _K, 1)
_W_SHAPE = (_N, _K)
_W_STRIDES = (_K, 1)

# Defaults used when the coordinator supplies no tunable values.
_DEFAULT_PARAMS = {
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "GROUP_M": 4,
    "num_warps": 4,
    "num_stages": 4,
}

# Keep shared memory inside the sm80 dynamic limit (166912 B) with margin.
_MAX_SHARED_BYTES = 155648


if _TRITON_OK:

    @triton.jit
    def _gate_up_gemm(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_bn,
        stride_cm,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # C[M, N] = A[M, K] @ B[N, K]^T with row-major A, B, C. The guarded
        # caller guarantees M % BLOCK_M == N % BLOCK_N == K % BLOCK_K == 0 and
        # unit stride along K for both operands, so no masks are needed.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :]
        b_ptrs = b_ptr + offs_bn[None, :] * stride_bn + offs_k[:, None]

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :]
        tl.store(c_ptrs, acc.to(tl.bfloat16))


def _make_replacement(parent, block_m, block_n, block_k, group_m, num_warps, num_stages):
    grid = (((_M + block_m - 1) // block_m) * ((_N + block_n - 1) // block_n),)

    def linear_repl(input, weight, bias=None):
        if (
            bias is None
            and input.dim() == 3
            and weight.dim() == 2
            and input.dtype is torch.bfloat16
            and weight.dtype is torch.bfloat16
            and input.is_cuda
            and tuple(input.shape) == _IN_SHAPE
            and tuple(weight.shape) == _W_SHAPE
            and input.stride() == _IN_STRIDES
            and weight.stride() == _W_STRIDES
            and input.device.index == torch.cuda.current_device()
            and not (
                torch.is_grad_enabled()
                and (weight.requires_grad or input.requires_grad)
            )
        ):
            out = torch.empty((_M, _N), device=input.device, dtype=torch.bfloat16)
            _gate_up_gemm[grid](
                input,
                weight,
                out,
                _M,
                _N,
                _K,
                _K,
                _K,
                _N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=group_m,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return out.view(1, _M, _N)
        return parent(input, weight, bias)

    return linear_repl


def build(context):
    """Return {call_id: callable} for the selected prefill gate/up GEMM sites."""
    if not _TRITON_OK:
        return {}

    raw = context.get("parameters") or {}
    params = {name: raw.get(name, default) for name, default in _DEFAULT_PARAMS.items()}
    try:
        block_m = int(params["BLOCK_M"])
        block_n = int(params["BLOCK_N"])
        block_k = int(params["BLOCK_K"])
        group_m = int(params["GROUP_M"])
        num_warps = int(params["num_warps"])
        num_stages = int(params["num_stages"])
    except (TypeError, ValueError):
        return {}
    if min(block_m, block_n, block_k, group_m, num_warps, num_stages) <= 0:
        return {}
    if _M % block_m or _N % block_n or _K % block_k:
        return {}
    if (block_m * block_k + block_k * block_n) * 2 * num_stages > _MAX_SHARED_BYTES:
        return {}

    targets = context.get("targets") or {}
    references = context.get("references") or {}
    replacements = {}
    for call_id in _SELECTED_CALLS:
        if call_id not in targets:
            continue
        parent = references.get(call_id)
        if parent is None:
            continue
        replacements[call_id] = _make_replacement(
            parent, block_m, block_n, block_k, group_m, num_warps, num_stages
        )
    return replacements


# Call ids of group 0c566a3a363f294c6750 (inputs/selection.json, rank 0).
_SELECTED_CALLS = (
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
    "faf83d504b9638b229e91040"

)
