"""New project input: normalized signal gating, with no nn.Module or target list.

The synthetic inputs are seeded; all output elements are part of correctness.
The measured operation is pipeline(), on GPU-resident FP32 inputs. Loading,
allocation and reference checking are outside steady-state timing.
"""
import torch


def inputs(rows, width, device, seed=91273):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, width, device=device, dtype=torch.float32, generator=generator)
    scale = torch.randn(width, device=device, dtype=torch.float32, generator=generator)*0.1+1
    bias = torch.randn(width, device=device, dtype=torch.float32, generator=generator)*0.05
    return x, scale, bias


def pipeline(x, scale, bias):
    affine = x*scale+bias
    wave = torch.sin(affine)+torch.cos(affine)
    mean = wave.mean(dim=-1, keepdim=True)
    variance = (wave-mean).square().mean(dim=-1, keepdim=True)
    normalized = (wave-mean)*torch.rsqrt(variance+1e-5)
    return normalized*torch.sigmoid(normalized)


def main():
    import argparse
    import time
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--width", type=int, default=4096)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    tensors = inputs(args.rows, args.width, "cuda")
    reference = pipeline(*tensors)
    torch.testing.assert_close(pipeline(*tensors), reference, atol=1e-5, rtol=1e-5)
    for _ in range(3):
        pipeline(*tensors)
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = pipeline(*tensors)
    torch.cuda.synchronize()
    print({"elapsed_ms": (time.perf_counter_ns()-started)/1e6, "output_elements": output.numel()})


if __name__ == "__main__":
    main()
