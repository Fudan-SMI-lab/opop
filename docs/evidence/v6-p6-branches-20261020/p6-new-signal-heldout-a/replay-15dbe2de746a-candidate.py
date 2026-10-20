"""Candidate: region rewrite of the normalized signal gating pipeline.

Replaces the project.invoke call (adapter.run) with a fused implementation of
the complete, verbatim pipeline body:

    affine     = x * scale + bias
    wave       = sin(affine) + cos(affine)
    mean       = wave.mean(dim=-1, keepdim=True)
    variance   = (wave - mean).square().mean(dim=-1, keepdim=True)
    normalized = (wave - mean) * rsqrt(variance + 1e-5)
    output     = normalized * sigmoid(normalized)

Primary path: one fused Triton kernel per invocation (one program per row)
that loads the row, computes affine -> sin+cos, block-reduces the row mean,
reuses the centered row (the reference subtracts the mean twice; the fused
kernel computes it once), block-reduces the variance, and applies the
rsqrt-normalize + sigmoid-gate epilogue in the same launch.

Fallback chain, selected once at build time (never switched mid-measurement):
  1. fused Triton kernel (requires triton, power-of-two width);
  2. CUDA-graph replay of the verbatim eager op sequence (bit-identical math
     to the parent, removes launch gaps; static buffers are identity-guarded);
  3. reduced-launch eager op sequence (addcmul / var_mean / silu).

Every replacement validates the observed signature properties (shape, dtype,
device, contiguity, no-grad) on each call and falls back to the parent
adapter.run for anything else.  No outputs are cached; the kernel runs on
every invocation.  The only build-time GPU work is the one-time Triton JIT
compilation, a CUDA-graph capture (fallback path only), and a single
equivalence check whose results are discarded.
"""

from __future__ import annotations

import contextlib

import torch

PROJECT_INVOKE_CALL = "5d7d6bb13691b02f533f62b0"

# Internal build-time gate, strictly tighter than the frozen quality
# thresholds (atol=rtol=1e-5).  Both the fused kernel and the eager
# reference are deterministic on the bound device, so passing this gate at
# build time implies the framework's looser check passes at validation.
_GATE_ATOL = 5e-6
_GATE_RTOL = 5e-6


def _phase(name):
    """Best-effort discovery phase annotation, mirroring adapter._phase."""
    try:
        from kernel_optimizer.project.discovery import phase  # type: ignore

        return phase(name)
    except Exception:
        return contextlib.nullcontext()


def _reference_pipeline(x, scale, bias):
    """Verbatim source/adapter math, used only for build-time gating."""
    affine = x * scale + bias
    wave = torch.sin(affine) + torch.cos(affine)
    mean = wave.mean(dim=-1, keepdim=True)
    variance = (wave - mean).square().mean(dim=-1, keepdim=True)
    normalized = (wave - mean) * torch.rsqrt(variance + 1e-5)
    return normalized * torch.sigmoid(normalized)


try:  # Triton ships with PyTorch CUDA builds; treat it as optional.
    import triton
    import triton.language as tl

    @triton.jit
    def _fused_gating_kernel(
        x_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        inv_n,
        WIDTH: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        base = row * WIDTH
        offs = tl.arange(0, WIDTH)
        x = tl.load(x_ptr + base + offs)
        scale = tl.load(scale_ptr + offs)
        bias = tl.load(bias_ptr + offs)
        affine = x * scale + bias
        wave = tl.sin(affine) + tl.cos(affine)
        mean = tl.sum(wave, axis=0) * inv_n
        centered = wave - mean
        variance = tl.sum(centered * centered, axis=0) * inv_n
        denom = tl.rsqrt(variance + 1e-5)
        normalized = centered * denom
        tl.store(out_ptr + base + offs, normalized * tl.sigmoid(normalized))

    _HAS_TRITON = True
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None
    _HAS_TRITON = False


def _tensor_guard(rows, width, device):
    """Signature guard matching the observed call specification."""

    def matches(x, scale, bias):
        return (
            isinstance(x, torch.Tensor)
            and isinstance(scale, torch.Tensor)
            and isinstance(bias, torch.Tensor)
            and x.dim() == 2
            and x.shape[0] == rows
            and x.shape[1] == width
            and scale.dim() == 1
            and scale.shape[0] == width
            and bias.dim() == 1
            and bias.shape[0] == width
            and x.dtype is torch.float32
            and scale.dtype is torch.float32
            and bias.dtype is torch.float32
            and x.is_cuda
            and x.device == device
            and x.is_contiguous()
            and scale.is_contiguous()
            and bias.is_contiguous()
            and not x.requires_grad
            and not scale.requires_grad
            and not bias.requires_grad
        )

    return matches


def _build_triton_runner(state, parent, num_warps):
    """One fused kernel launch per invocation; None if unavailable/unclean."""
    if not _HAS_TRITON:
        return None
    try:
        x = state["x"]
        scale = state["scale"]
        bias = state["bias"]
        rows = int(x.shape[0])
        width = int(x.shape[1])
        if rows <= 0 or width <= 0 or width & (width - 1):
            return None  # tl.arange requires a power-of-two tile
        matches = _tensor_guard(rows, width, x.device)
        if not matches(x, scale, bias):
            return None
        inv_n = 1.0 / width
        grid = (rows,)
        kernel = _fused_gating_kernel
        device_ctx = torch.cuda.device(x.device)

        # One-time JIT compile + equivalence gate on the prepared inputs.
        # The gate result is a boolean only; nothing is cached or replayed.
        with device_ctx:
            probe = torch.empty_like(x)
            kernel[grid](
                x, scale, bias, probe, inv_n, WIDTH=width, num_warps=num_warps
            )
            reference = _reference_pipeline(x, scale, bias)
            if not torch.allclose(probe, reference, atol=_GATE_ATOL, rtol=_GATE_RTOL):
                return None

        def runner(st):
            try:
                xs = st.get("x")
                ss = st.get("scale")
                bs = st.get("bias")
                if matches(xs, ss, bs):
                    with _phase("pipeline"):
                        out = torch.empty_like(xs)
                        kernel[grid](
                            xs,
                            ss,
                            bs,
                            out,
                            inv_n,
                            WIDTH=width,
                            num_warps=num_warps,
                        )
                    return {"output": out}
            except Exception:
                pass
            return parent(st)

        return runner
    except Exception:
        return None


def _build_graph_runner(state, parent):
    """CUDA-graph replay of the verbatim eager ops; bit-identical math."""
    graph = None
    try:
        x = state["x"]
        scale = state["scale"]
        bias = state["bias"]
        if not (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            and isinstance(scale, torch.Tensor)
            and isinstance(bias, torch.Tensor)
        ):
            return None
        with torch.cuda.device(x.device):
            side = torch.cuda.Stream(device=x.device)
            side.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    _reference_pipeline(x, scale, bias)  # capture-safe warmup
            torch.cuda.current_stream(x.device).wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = _reference_pipeline(x, scale, bias)

            # One replay + equivalence gate; values are not reused.
            graph.replay()
            reference = _reference_pipeline(x, scale, bias)
            if not torch.allclose(
                static_out, reference, atol=_GATE_ATOL, rtol=_GATE_RTOL
            ):
                return None

        def runner(st):
            try:
                # Replay reads the captured buffers, so require the exact
                # tensors that were captured; anything else goes to parent.
                if (
                    st.get("x") is x
                    and st.get("scale") is scale
                    and st.get("bias") is bias
                ):
                    with _phase("pipeline"):
                        graph.replay()
                        return {"output": static_out.clone()}
            except Exception:
                pass
            return parent(st)

        return runner
    except Exception:
        if graph is not None:
            try:
                graph.reset()
            except Exception:
                pass
        return None


def _build_eager_runner(parent):
    """Reduced-launch eager fallback; computes from the passed tensors."""

    def runner(st):
        try:
            x = st.get("x")
            scale = st.get("scale")
            bias = st.get("bias")
            if (
                isinstance(x, torch.Tensor)
                and x.is_cuda
                and x.dim() == 2
                and x.shape[1] > 0
                and isinstance(scale, torch.Tensor)
                and scale.dim() == 1
                and scale.shape[0] == x.shape[1]
                and isinstance(bias, torch.Tensor)
                and bias.dim() == 1
                and bias.shape[0] == x.shape[1]
                and x.dtype is torch.float32
                and scale.dtype is torch.float32
                and bias.dtype is torch.float32
                and x.is_contiguous()
                and scale.is_contiguous()
                and bias.is_contiguous()
                and not x.requires_grad
            ):
                with _phase("pipeline"):
                    affine = torch.addcmul(bias, x, scale)
                    wave = torch.sin(affine)
                    wave = torch.add(wave, torch.cos(affine), out=wave)
                    row_mean, row_var = torch.var_mean(
                        wave, dim=-1, keepdim=True, correction=0
                    )
                    centered = torch.sub(wave, row_mean)
                    denom = torch.rsqrt(row_var.add_(1e-5))
                    normalized = torch.mul(centered, denom)
                    return {"output": torch.nn.functional.silu(normalized)}
        except Exception:
            pass
        return parent(st)

    return runner


def build(context):
    """Return replacements for the selected region (project.invoke)."""
    state = context["state"]
    parent = context["references"][PROJECT_INVOKE_CALL]

    parameters = dict(context.get("parameters") or {})
    impl = str(parameters.get("impl", "triton"))
    if impl not in ("triton", "graph", "eager"):
        impl = "triton"
    try:
        num_warps = int(parameters.get("num_warps", 8))
    except Exception:
        num_warps = 8
    if num_warps not in (4, 8, 16):
        num_warps = 8

    runner = None
    if impl == "triton":
        runner = _build_triton_runner(state, parent, num_warps)
        if runner is None:
            impl = "graph"  # graceful degradation, same invocation ABI
    if impl == "graph":
        runner = _build_graph_runner(state, parent)
    if runner is None:
        runner = _build_eager_runner(parent)

    return {PROJECT_INVOKE_CALL: runner}
