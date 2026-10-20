"""Candidate: fused Triton gating kernel executed through a one-node CUDA graph.

Region rewrite of the normalized signal gating pipeline through the
project.invoke call (adapter.run ABI: runner(state) -> {"output": tensor}).
The timed path is the accepted revision's stable replay structure - one
CUDA-graph replay plus one eager fresh-output clone per invocation - where
the graph now contains a single fused Triton kernel implementing the
verbatim pipeline body:

    affine     = x * scale + bias
    wave       = sin(affine) + cos(affine)
    mean       = wave.mean(dim=-1, keepdim=True)
    variance   = (wave - mean).square().mean(dim=-1, keepdim=True)
    normalized = (wave - mean) * rsqrt(variance + 1e-5)
    output     = normalized * sigmoid(normalized)

Build-time (one-time, outside timing): JIT-compile the kernel, gate the
eager probe against the verbatim eager reference on the prepared inputs,
capture the single launch into a CUDA graph with a static output buffer,
gate one replay of that graph, then pre-warm with extra replays and a
synchronize so allocator and clock state are settled before the
framework's own warmup.

Fallback chain, selected once at build time (never switched
mid-measurement):
  1. one-node CUDA graph over the fused Triton kernel (primary);
  2. CUDA-graph replay of the verbatim eager op sequence (bit-identical
     math to the parent, the accepted revision's own path - parity);
  3. the parent reference (adapter.run).

The reformulated eager sequence (var_mean / silu) is deliberately absent:
it failed quality deterministically twice (max_abs_error 0.05610130...).
Every path is identity-guarded to the captured input tensors and falls
back to the parent reference for anything else.  No outputs are cached;
the kernel executes on every invocation via graph.replay() reading the
live input buffers, so the workload runs in full on each invocation.
"""

from __future__ import annotations

import contextlib

import torch

PROJECT_INVOKE_CALL = "5d7d6bb13691b02f533f62b0"

# Internal build-time gate, strictly tighter than the frozen quality
# thresholds (atol=rtol=1e-5).  The fused kernel, the captured graph and
# the verbatim eager reference are all deterministic on the bound device,
# so passing this gate at build time predicts the framework's looser
# check at validation.
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
            and scale.device == device
            and bias.device == device
            and x.is_contiguous()
            and scale.is_contiguous()
            and bias.is_contiguous()
            and not x.requires_grad
            and not scale.requires_grad
            and not bias.requires_grad
        )

    return matches


def _build_fused_graph_runner(state, parent, num_warps, record):
    """One-node CUDA graph over the fused kernel; None if unavailable/unclean."""
    if not _HAS_TRITON:
        return None
    graph = None
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
        route = "fused_triton_cuda_graph"

        with torch.cuda.device(x.device):
            # Eager probe: performs the one-time JIT compilation of this
            # single specialization and gates the fused math.  Only the
            # boolean outcome is kept; probe and reference are discarded.
            probe = torch.empty_like(x)
            kernel[grid](
                x, scale, bias, probe, inv_n, WIDTH=width, num_warps=num_warps
            )
            reference = _reference_pipeline(x, scale, bias)
            if not torch.allclose(probe, reference, atol=_GATE_ATOL, rtol=_GATE_RTOL):
                return None

            # Static output owned by this candidate; every replay
            # overwrites it from the live input buffers.
            static_out = torch.empty_like(x)

            # Side-stream warmup so capture runs against a hot, settled
            # launcher (no compilation or allocation inside capture).
            side = torch.cuda.Stream(device=x.device)
            side.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    kernel[grid](
                        x,
                        scale,
                        bias,
                        static_out,
                        inv_n,
                        WIDTH=width,
                        num_warps=num_warps,
                    )
            torch.cuda.current_stream(x.device).wait_stream(side)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kernel[grid](
                    x,
                    scale,
                    bias,
                    static_out,
                    inv_n,
                    WIDTH=width,
                    num_warps=num_warps,
                )

            # Replay gate: the captured node itself must reproduce the
            # gated math (zero first so the values can only come from the
            # replay).
            static_out.zero_()
            graph.replay()
            if not torch.allclose(
                static_out, reference, atol=_GATE_ATOL, rtol=_GATE_RTOL
            ):
                return None

            # Pre-warm replays plus synchronize so allocator and clock
            # state are settled before the framework's warmup; the
            # values are not retained.
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize(x.device)

        def runner(st):
            try:
                # Replay reads the captured buffers, so require the exact
                # tensors that were captured; anything else goes to parent.
                if (
                    st.get("x") is x
                    and st.get("scale") is scale
                    and st.get("bias") is bias
                ):
                    record(PROJECT_INVOKE_CALL, route)
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


def _build_verbatim_graph_runner(state, parent, record):
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
        route = "verbatim_cuda_graph"
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
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize(x.device)

        def runner(st):
            try:
                if (
                    st.get("x") is x
                    and st.get("scale") is scale
                    and st.get("bias") is bias
                ):
                    record(PROJECT_INVOKE_CALL, route)
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


def _build_parent_runner(parent, record):
    """Last resort: delegate every invocation to the parent reference."""

    def runner(st):
        record(PROJECT_INVOKE_CALL, "parent_delegate")
        return parent(st)

    return runner


def build(context):
    """Return replacements for the selected region (project.invoke)."""
    state = context["state"]
    parent = context["references"][PROJECT_INVOKE_CALL]
    record = context.get("record_route", lambda *_: None)

    parameters = dict(context.get("parameters") or {})
    try:
        num_warps = int(parameters.get("num_warps", 8))
    except Exception:
        num_warps = 8
    if num_warps not in (4, 8, 16):
        num_warps = 8

    runner = _build_fused_graph_runner(state, parent, num_warps, record)
    if runner is None:
        runner = _build_verbatim_graph_runner(state, parent, record)
    if runner is None:
        runner = _build_parent_runner(parent, record)

    return {PROJECT_INVOKE_CALL: runner}
