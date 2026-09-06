import torch
import torch.nn as nn
import triton
import triton.language as tl


PARAMS = {
    'EXPAND_BLOCK_M': 64,
    'EXPAND_BLOCK_N': 128,
    'EXPAND_BLOCK_K': 16,
    'EXPAND_WARPS': 4,
    'EXPAND_STAGES': 3,
    'PROJECT_BLOCK_M': 64,
    'PROJECT_BLOCK_N': 128,
    'PROJECT_BLOCK_K': 16,
    'PROJECT_WARPS': 4,
    'PROJECT_STAGES': 3,
    'COMPUTE_DTYPE': 'fp16',
    'DW_BLOCK': 256,
    'DW_WARPS': 4,
    'DW_STAGES': 3,
    'MOMENTS_WARPS': 4,
    'FINAL_BLOCK': 1024,
    'FINAL_WARPS': 2,
}


@triton.jit
def _pointwise_produce_moments(
        x, weight, y, partial_sum, partial_sq, batch_hw: tl.constexpr,
        in_channels: tl.constexpr, out_channels: tl.constexpr,
        hw: tl.constexpr, num_partials: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, COMPUTE_DTYPE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = offs_n // hw
    spatial = offs_n % hw
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, in_channels, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < out_channels) & (offs_k[None, :] < in_channels)
        b_mask = (offs_k[:, None] < in_channels) & (offs_n[None, :] < batch_hw)
        a = tl.load(weight + offs_m[:, None] * in_channels + offs_k[None, :],
                    mask=a_mask, other=0.0)
        b = tl.load(x + batch[None, :] * in_channels * hw +
                    offs_k[:, None] * hw + spatial[None, :],
                    mask=b_mask, other=0.0)
        if COMPUTE_DTYPE == "fp16":
            acc += tl.dot(a.to(tl.float16), b.to(tl.float16), input_precision="ieee")
        elif COMPUTE_DTYPE == "bf16":
            acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), input_precision="ieee")
        elif COMPUTE_DTYPE == "tf32":
            acc += tl.dot(a, b, input_precision="tf32")
        else:
            acc += tl.dot(a, b, input_precision="ieee")
    valid_m = offs_m < out_channels
    valid_n = offs_n < batch_hw
    mask = valid_m[:, None] & valid_n[None, :]
    offsets = (batch[None, :] * out_channels * hw +
               offs_m[:, None] * hw + spatial[None, :])
    tl.store(y + offsets, acc, mask=mask)
    values = tl.where(valid_n[None, :], acc, 0.0)
    partial_offsets = offs_m * num_partials + pid_n
    tl.store(partial_sum + partial_offsets, tl.sum(values, axis=1), mask=valid_m)
    tl.store(partial_sq + partial_offsets, tl.sum(values * values, axis=1),
             mask=valid_m)


@triton.jit
def _depthwise_produce_moments(
        x, weight, y, mean, var, gamma, beta, partial_sum, partial_sq,
        batch_out_hw: tl.constexpr, channels: tl.constexpr,
        in_h: tl.constexpr, in_w: tl.constexpr,
        out_h: tl.constexpr, out_w: tl.constexpr,
        kernel_size: tl.constexpr, stride: tl.constexpr,
        padding: tl.constexpr, eps: tl.constexpr,
        num_partials: tl.constexpr, APPLY_NORM: tl.constexpr,
        BLOCK: tl.constexpr):
    channel = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    valid_n = offs_n < batch_out_hw
    batch = offs_n // (out_h * out_w)
    out_spatial = offs_n % (out_h * out_w)
    oh = out_spatial // out_w
    ow = out_spatial % out_w
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    if APPLY_NORM:
        mu = tl.load(mean + channel)
        inv = tl.rsqrt(tl.load(var + channel) + eps) * tl.load(gamma + channel)
        shift = tl.load(beta + channel)
    for kh in range(0, kernel_size):
        ih = oh * stride - padding + kh
        for kw in range(0, kernel_size):
            iw = ow * stride - padding + kw
            in_bounds = valid_n & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
            values = tl.load(x + batch * channels * in_h * in_w +
                             channel * in_h * in_w + ih * in_w + iw,
                             mask=in_bounds, other=0.0).to(tl.float32)
            if APPLY_NORM:
                values = tl.minimum(tl.maximum((values - mu) * inv + shift, 0.0), 6.0)
                values = tl.where(in_bounds, values, 0.0)
            w = tl.load(weight + channel * kernel_size * kernel_size +
                        kh * kernel_size + kw)
            acc += values * w
    output_offsets = (batch * channels * out_h * out_w +
                      channel * out_h * out_w + out_spatial)
    tl.store(y + output_offsets, acc, mask=valid_n)
    values = tl.where(valid_n, acc, 0.0)
    partial_offset = channel * num_partials + pid_n
    tl.store(partial_sum + partial_offset, tl.sum(values, axis=0))
    tl.store(partial_sq + partial_offset, tl.sum(values * values, axis=0))


@triton.jit
def _pointwise_consume_bn_produce_moments(
        x, weight, y, mean, var, gamma, beta, partial_sum, partial_sq,
        batch_hw: tl.constexpr, in_channels: tl.constexpr,
        out_channels: tl.constexpr, hw: tl.constexpr, eps: tl.constexpr,
        num_partials: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        COMPUTE_DTYPE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    batch = offs_n // hw
    spatial = offs_n % hw
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, in_channels, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < out_channels) & (offs_k[None, :] < in_channels)
        b_mask = (offs_k[:, None] < in_channels) & (offs_n[None, :] < batch_hw)
        a = tl.load(weight + offs_m[:, None] * in_channels + offs_k[None, :],
                    mask=a_mask, other=0.0)
        b = tl.load(x + batch[None, :] * in_channels * hw +
                    offs_k[:, None] * hw + spatial[None, :], mask=b_mask, other=0.0)
        mu = tl.load(mean + offs_k[:, None], mask=offs_k[:, None] < in_channels,
                     other=0.0)
        variance = tl.load(var + offs_k[:, None], mask=offs_k[:, None] < in_channels,
                           other=0.0)
        scale = tl.load(gamma + offs_k[:, None], mask=offs_k[:, None] < in_channels,
                        other=0.0)
        shift = tl.load(beta + offs_k[:, None], mask=offs_k[:, None] < in_channels,
                        other=0.0)
        b = tl.minimum(tl.maximum((b - mu) * tl.rsqrt(variance + eps) * scale + shift,
                                  0.0), 6.0)
        b = tl.where(b_mask, b, 0.0)
        if COMPUTE_DTYPE == "fp16":
            acc += tl.dot(a.to(tl.float16), b.to(tl.float16), input_precision="ieee")
        elif COMPUTE_DTYPE == "bf16":
            acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), input_precision="ieee")
        elif COMPUTE_DTYPE == "tf32":
            acc += tl.dot(a, b, input_precision="tf32")
        else:
            acc += tl.dot(a, b, input_precision="ieee")
    valid_m = offs_m < out_channels
    valid_n = offs_n < batch_hw
    mask = valid_m[:, None] & valid_n[None, :]
    offsets = (batch[None, :] * out_channels * hw +
               offs_m[:, None] * hw + spatial[None, :])
    tl.store(y + offsets, acc, mask=mask)
    values = tl.where(valid_n[None, :], acc, 0.0)
    partial_offsets = offs_m * num_partials + pid_n
    tl.store(partial_sum + partial_offsets, tl.sum(values, axis=1), mask=valid_m)
    tl.store(partial_sq + partial_offsets, tl.sum(values * values, axis=1),
             mask=valid_m)


@triton.jit
def _finish_moments(partial_sum, partial_sq, stats,
                    channels: tl.constexpr, count: tl.constexpr,
                    num_partials: tl.constexpr, BLOCK: tl.constexpr):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_partials
    base = channel * num_partials + offsets
    total = tl.sum(tl.load(partial_sum + base, mask=mask, other=0.0), axis=0)
    total_sq = tl.sum(tl.load(partial_sq + base, mask=mask, other=0.0), axis=0)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    tl.store(stats + channel, variance)
    tl.store(stats + channels + channel, mean)


@triton.jit
def _final_bn_residual(x, identity, y, mean, var, gamma, beta,
                       total: tl.constexpr, channels: tl.constexpr,
                       hw: tl.constexpr, eps: tl.constexpr,
                       RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    channel = (offsets // hw) % channels
    values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    mu = tl.load(mean + channel, mask=mask, other=0.0)
    variance = tl.load(var + channel, mask=mask, other=0.0)
    scale = tl.load(gamma + channel, mask=mask, other=0.0)
    shift = tl.load(beta + channel, mask=mask, other=0.0)
    result = (values - mu) * tl.rsqrt(variance + eps) * scale + shift
    if RESIDUAL:
        result += tl.load(identity + offsets, mask=mask, other=0.0)
    tl.store(y + offsets, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True))
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True))
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels))

    def _partial_buffers(self, channels, num_partials, device):
        partials = torch.empty((2, channels, num_partials), device=device,
                               dtype=torch.float32)
        return partials[0], partials[1]

    def _finish_stats(self, partial_sum, partial_sq, channels, count, num_partials):
        stats = torch.empty((2, channels), device=partial_sum.device, dtype=torch.float32)
        _finish_moments[(channels,)](
            partial_sum, partial_sq, stats, channels, count, num_partials,
            BLOCK=triton.next_power_of_2(num_partials),
            num_warps=PARAMS["MOMENTS_WARPS"])
        return stats[0], stats[1]

    def _pointwise_raw(self, x, weight):
        batch, in_channels, height, width = x.shape
        out_channels = weight.shape[0]
        hw = height * width
        count = batch * hw
        num_partials = triton.cdiv(count, PARAMS["EXPAND_BLOCK_N"])
        y = torch.empty((batch, out_channels, height, width), device=x.device,
                        dtype=x.dtype)
        partial_sum, partial_sq = self._partial_buffers(out_channels, num_partials,
                                                        x.device)
        grid = (triton.cdiv(out_channels, PARAMS["EXPAND_BLOCK_M"]), num_partials)
        _pointwise_produce_moments[grid](
            x, weight, y, partial_sum, partial_sq, count, in_channels, out_channels,
            hw, num_partials, BLOCK_M=PARAMS["EXPAND_BLOCK_M"],
            BLOCK_N=PARAMS["EXPAND_BLOCK_N"], BLOCK_K=PARAMS["EXPAND_BLOCK_K"],
            COMPUTE_DTYPE=PARAMS["COMPUTE_DTYPE"], num_warps=PARAMS["EXPAND_WARPS"],
            num_stages=PARAMS["EXPAND_STAGES"])
        return y, self._finish_stats(partial_sum, partial_sq, out_channels, count,
                                     num_partials)

    def _depthwise(self, x, conv, stats=None, bn=None):
        batch, channels, in_h, in_w = x.shape
        kernel_size = conv.weight.shape[2]
        stride = conv.stride[0]
        padding = conv.padding[0]
        out_h = (in_h + 2 * padding - kernel_size) // stride + 1
        out_w = (in_w + 2 * padding - kernel_size) // stride + 1
        count = batch * out_h * out_w
        num_partials = triton.cdiv(count, PARAMS["DW_BLOCK"])
        y = torch.empty((batch, channels, out_h, out_w), device=x.device, dtype=x.dtype)
        partial_sum, partial_sq = self._partial_buffers(channels, num_partials, x.device)
        if stats is None:
            mean = var = gamma = beta = x
            eps = 0.0
            apply_norm = False
        else:
            var, mean = stats
            gamma, beta, eps = bn.weight, bn.bias, bn.eps
            apply_norm = True
        _depthwise_produce_moments[(channels, num_partials)](
            x, conv.weight, y, mean, var, gamma, beta, partial_sum, partial_sq,
            count, channels, in_h, in_w, out_h, out_w, kernel_size, stride,
            padding, eps, num_partials, APPLY_NORM=apply_norm,
            BLOCK=PARAMS["DW_BLOCK"], num_warps=PARAMS["DW_WARPS"],
            num_stages=PARAMS["DW_STAGES"])
        return y, self._finish_stats(partial_sum, partial_sq, channels, count,
                                     num_partials)

    def _project(self, x, weight, bn, stats):
        batch, in_channels, height, width = x.shape
        out_channels = weight.shape[0]
        hw = height * width
        count = batch * hw
        num_partials = triton.cdiv(count, PARAMS["PROJECT_BLOCK_N"])
        var, mean = stats
        y = torch.empty((batch, out_channels, height, width), device=x.device,
                        dtype=x.dtype)
        partial_sum, partial_sq = self._partial_buffers(out_channels, num_partials,
                                                        x.device)
        grid = (triton.cdiv(out_channels, PARAMS["PROJECT_BLOCK_M"]), num_partials)
        _pointwise_consume_bn_produce_moments[grid](
            x, weight, y, mean, var, bn.weight, bn.bias, partial_sum, partial_sq,
            count, in_channels, out_channels, hw, bn.eps, num_partials,
            BLOCK_M=PARAMS["PROJECT_BLOCK_M"], BLOCK_N=PARAMS["PROJECT_BLOCK_N"],
            BLOCK_K=PARAMS["PROJECT_BLOCK_K"],
            COMPUTE_DTYPE=PARAMS["COMPUTE_DTYPE"],
            num_warps=PARAMS["PROJECT_WARPS"], num_stages=PARAMS["PROJECT_STAGES"])
        return y, self._finish_stats(partial_sum, partial_sq, out_channels, count,
                                     num_partials)

    def forward(self, x):
        identity = x
        if hasattr(self, "expand_conv"):
            expanded, expand_stats = self._pointwise_raw(x, self.expand_conv[0].weight)
            x, depthwise_stats = self._depthwise(
                expanded, self.depthwise_conv[0], expand_stats, self.expand_conv[1])
        else:
            x, depthwise_stats = self._depthwise(x, self.depthwise_conv[0])
        x, project_stats = self._project(
            x, self.project_conv[0].weight, self.depthwise_conv[1], depthwise_stats)
        var, mean = project_stats
        y = torch.empty_like(x)
        channels = x.shape[1]
        hw = x.shape[2] * x.shape[3]
        _final_bn_residual[(triton.cdiv(x.numel(), PARAMS["FINAL_BLOCK"]),)](
            x, identity, y, mean, var, self.project_conv[1].weight,
            self.project_conv[1].bias, x.numel(), channels, hw,
            self.project_conv[1].eps, RESIDUAL=self.use_residual,
            BLOCK=PARAMS["FINAL_BLOCK"], num_warps=PARAMS["FINAL_WARPS"])
        return y
