"""Repair candidate for the rejected trial-61aff8484b.

Selected group fc4ee42c1ef82b000685, call faf88ed785916744a12ba5c9:
aten::conv2d with input [1,256,14,14] (stride (50176,196,14,1)), weight
[512,256,3,3] (stride (2304,9,3,1)), bias None, stride [2,2], padding [1,1],
dilation [1,1], groups 1 -> output [1,512,7,7] (stage-3 -> stage-4
downsampling convolution of ResNet-18).

The rejected predecessor replaced three other convolutions with an
im2col + cuBLAS-GEMM composite; the paired project measurement regressed the
objective from ~1.45 ms to ~4.9 ms, i.e. per-call composite work (extra aten
dispatches and a slow skinny-N batched SGEMM route) dominated any kernel-time
saving on this dispatch-sensitive batch=1 workload.  This candidate follows
the repaired selection exactly: a single-launch, occupancy-aware FP32 CUDA
kernel for this one signature, with no per-call CPU-side dispatch beyond one
C++ extension call.

Kernel design (RTX 4090, 128 SMs, FP32, TF32 disabled by the frozen policy):
  * the cuDNN heuristic (benchmark stays False) serves the call with one
    implicit_convolve_sgemm kernel launched as grid [2,16] x block [8,8]
    (32 CTAs / 2048 threads total, 115.5 diagnostic us, ~1.0 TFLOP/s);
  * the replacement launches grid (16,7) = 112 CTAs x 256 threads: CTA
    (bx, by) owns output channels [32*bx, 32*bx+32) of output row by; it
    stages the three input rows that row `by` touches into 48 KB of shared
    memory (zero padding included), then each thread performs the K=2304
    dot product of one (output channel, output column) pair with fmaf
    accumulation in a fixed, deterministic order;
  * weights are read through float4 loads from a build-time one-per-model
    repack of the invariant [512,256,3,3] weight into a 16-byte aligned
    [512,256,12] layout (taps 9..11 zero-padded); the runtime weight
    argument is checked against the packed source pointer every call and any
    mismatch (or any non-observed signature) falls back to the parent
    aten::conv2d reference.

No global torch/cudnn/cublas flag is touched, no module.forward wrapper is
installed, no fixture-dependent value is cached (the kernel consumes the
live input tensor and live weight identity on every invocation), and the
whole forward pass still executes on each run.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CALL_ID = "faf88ed785916744a12ba5c9"
EXT_NAME = "opop_conv_s2_faf88ed7_v1"

_IN_SHAPE = (1, 256, 14, 14)
_IN_STRIDE = (50176, 196, 14, 1)
_W_SHAPE = (512, 256, 3, 3)
_W_STRIDE = (2304, 9, 3, 1)
_OUT_STRIDE = (25088, 49, 7, 1)

_CPP_SRC = "torch::Tensor conv_s2(torch::Tensor x, torch::Tensor wp);"

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// y[0, co, by, bx_col] = sum_{ci, i, j} w[co, ci, i, j] * xpad[ci, 2*by+i-1, 2*col+j-1]
// x: contiguous [1,256,14,14]; wp: contiguous [512,256,12] (taps 0..8 real, 9..11 zero)
// y: contiguous [1,512,7,7].  grid (16,7), block 256.
__global__ void conv_s2_kernel(const float* __restrict__ x,
                               const float4* __restrict__ w4,
                               float* __restrict__ y) {
  __shared__ float xs[3 * 256 * 16];  // 49152 B; idx = i*4096 + ci*16 + (col+1)
  const int co0 = blockIdx.x << 5;
  const int oy = blockIdx.y;
  const int tid = threadIdx.x;
  for (int idx = tid; idx < 3 * 256 * 16; idx += 256) {
    const int i = idx >> 12;
    const int rem = idx & 4095;
    const int ci = rem >> 4;
    const int cp = rem & 15;
    const int r = (oy << 1) + i - 1;
    const int c = cp - 1;
    float v = 0.0f;
    if (r >= 0 && r < 14 && c >= 0 && c < 14) {
      v = __ldg(x + ci * 196 + r * 14 + c);
    }
    xs[idx] = v;
  }
  __syncthreads();
  const int cl = tid >> 3;   // 0..31  -> output channel co0+cl
  const int xl = tid & 7;    // 0..7   -> output column (7 is idle)
  if (xl < 7) {
    const float4* wr = w4 + (size_t)(co0 + cl) * 768;  // 256*12/4 float4 per co
    float acc = 0.0f;
    for (int ci = 0; ci < 256; ++ci) {
      const float* xb = xs + ci * 16;
      #pragma unroll
      for (int g = 0; g < 3; ++g) {
        const float4 wv = __ldg(wr + ci * 3 + g);
        const float* wf = reinterpret_cast<const float*>(&wv);
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = (g << 2) + e;
          if (t < 9) {
            acc = fmaf(wf[e], xb[(t / 3) * 4096 + (xl << 1) + (t % 3)], acc);
          }
        }
      }
    }
    y[(size_t)(co0 + cl) * 49 + oy * 7 + xl] = acc;
  }
}

torch::Tensor conv_s2(torch::Tensor x, torch::Tensor wp) {
  TORCH_CHECK(x.is_cuda() && wp.is_cuda(), "conv_s2: cuda tensors required");
  TORCH_CHECK(x.scalar_type() == at::kFloat && wp.scalar_type() == at::kFloat,
              "conv_s2: float32 required");
  TORCH_CHECK(x.dim() == 4 && x.size(0) == 1 && x.size(1) == 256 &&
                  x.size(2) == 14 && x.size(3) == 14,
              "conv_s2: input must be [1,256,14,14]");
  TORCH_CHECK(x.is_contiguous(), "conv_s2: input must be contiguous");
  TORCH_CHECK(wp.dim() == 3 && wp.size(0) == 512 && wp.size(1) == 256 &&
                  wp.size(2) == 12 && wp.is_contiguous(),
              "conv_s2: weight pack must be [512,256,12] contiguous");
  auto y = torch::empty({1, 512, 7, 7}, x.options());
  conv_s2_kernel<<<dim3(16, 7), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),
      reinterpret_cast<const float4*>(wp.data_ptr<float>()),
      y.data_ptr<float>());
  return y;
}
"""

# Soft sanity path of the physical module that owns this call site.
_MODULE_PATH = ("resnet", "encoder", "stages", 3, "layers", 0, "layer", 0,
                "convolution")


def _module_weight(model):
    try:
        node = model
        for part in _MODULE_PATH:
            node = node[int(part)] if isinstance(part, int) else getattr(node, part)
        w = node.weight
        if (tuple(w.shape) == _W_SHAPE and tuple(w.stride()) == _W_STRIDE
                and w.dtype is torch.float32 and w.is_cuda and w.is_contiguous()):
            return w
    except Exception:
        pass
    return None


def _pack_weight(w):
    """[512,256,3,3] contiguous -> [512,256,12] float4-aligned, taps 9..11 zero."""
    wp = torch.zeros((512, 256, 12), device=w.device, dtype=torch.float32)
    wp[:, :, :9].copy_(w.reshape(512, 256, 9))
    return wp.contiguous()


def _compile_extension():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=EXT_NAME,
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["conv_s2"],
        verbose=False,
    )


def _self_check(ext, device):
    """One small randomized on-device check of kernel + packing (build time only).

    Uses a private CPU generator so global RNG state is untouched; transients
    are freed before returning.
    """
    x = wt = wp = ref = out = None
    try:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1234567)
        x = torch.randn((1, 256, 14, 14), generator=gen, dtype=torch.float32).to(device)
        wt = torch.randn((512, 256, 3, 3), generator=gen, dtype=torch.float32).to(device)
        wp = _pack_weight(wt)
        with torch.no_grad():
            ref = F.conv2d(x, wt, None, (2, 2), (1, 1))
            out = ext.conv_s2(x, wp)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return bool(
                tuple(out.shape) == (1, 512, 7, 7)
                and tuple(out.stride()) == _OUT_STRIDE
                and out.dtype is torch.float32
                and out.device == x.device
                and torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
            )
    except Exception:
        return False
    finally:
        for t in (x, wt, wp, ref, out):
            if t is not None:
                del t


def build(context):
    references = context.get("references") or {}
    parent = references.get(CALL_ID)
    if parent is None:
        parent = torch.conv2d

    ext = None
    packed = None
    src_ptr = None
    fast_ok = False

    weight = None
    models = context.get("models") or {}
    model = models.get("model")
    if model is not None:
        weight = _module_weight(model)

    if weight is not None and torch.cuda.is_available():
        try:
            ext = _compile_extension()
        except Exception:
            ext = None
        if ext is not None:
            try:
                packed = _pack_weight(weight)
                src_ptr = weight.data_ptr()
                fast_ok = _self_check(ext, weight.device)
            except Exception:
                fast_ok = False
            if not fast_ok:
                packed = None
                src_ptr = None

    def conv2d_replacement(input, weight_arg, bias=None, stride=(1, 1),
                           padding=(0, 0), dilation=(1, 1), groups=1):
        # Fast path only for the exact observed signature and the exact packed
        # (invariant) weight storage; everything else uses the parent op.
        if fast_ok and bias is None and groups == 1 and input.is_cuda:
            try:
                if (stride[0] == 2 and stride[1] == 2
                        and padding[0] == 1 and padding[1] == 1
                        and dilation[0] == 1 and dilation[1] == 1
                        and input.dtype is torch.float32
                        and weight_arg.dtype is torch.float32
                        and tuple(input.shape) == _IN_SHAPE
                        and tuple(input.stride()) == _IN_STRIDE
                        and tuple(weight_arg.shape) == _W_SHAPE
                        and tuple(weight_arg.stride()) == _W_STRIDE
                        and weight_arg.data_ptr() == src_ptr
                        and not (torch.is_grad_enabled() and weight_arg.requires_grad)):
                    return ext.conv_s2(input, packed)
            except Exception:
                pass
        return parent(input, weight_arg, bias, stride, padding, dilation, groups)

    return {CALL_ID: conv2d_replacement}
