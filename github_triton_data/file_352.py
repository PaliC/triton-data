import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

from torch import nn

try:
    # @manual=//triton:triton
    from triton.language.extra.cuda.libdevice import rsqrt as tl_rsqrt
except ImportError:
    # @manual=//triton:triton
    from triton.language.math import rsqrt as tl_rsqrt


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@triton.jit
def rms_norm_kernel(
    x_ptr,
    h1_ptr,
    w_ptr,
    eps,
    stride,
    N_COLS,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    h1_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(
            x_ptr + cols, mask=cols < N_COLS, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        _mean += a * a
    rstd = tl_rsqrt((tl.sum(_mean, axis=0) / N_COLS) + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask)
        tl.store(h1_ptr + cols, a * rstd * w, mask=mask)


@triton.jit
def rms_norm_add_kernel(
    x_ptr,
    y_ptr,
    h1_ptr,
    w_ptr,
    eps,
    stride,
    N_COLS,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    h1_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        ax = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        ay = tl.load(
            y_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        a = ax + ay
        tl.store(x_ptr + cols, a, mask=mask)
        _mean += a * a
    rstd = tl_rsqrt((tl.sum(_mean, axis=0) / N_COLS) + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask)
        tl.store(h1_ptr + cols, a * rstd * w, mask=mask)


def rms_norm(x, attn_norm_weights, eps):
    assert x.is_contiguous()
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    with torch.cuda.device(x.device):
        rms_norm_kernel[(M,)](
            x_arg,
            out,
            attn_norm_weights,
            eps,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return out


def rms_norm_add(x, y, attn_norm_weights, eps):
    # x, y contiguous of same shape [..., n]
    # output of same shape, normed over the last dim.
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert attn_norm_weights.is_contiguous()
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    y_arg = y.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    with torch.cuda.device(x.device.index):
        rms_norm_add_kernel[(M,)](
            x_arg,
            y_arg,
            out,
            attn_norm_weights,
            eps,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return out


class TritonRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)

    def increment_and_forward_(self, x, y):
        """
        An addition fused with forward.

            z = layer.increment_and_forward_(x, y)

        is equivalent to

            x += y
            z = layer(x)
        """
        return rms_norm_add(x, y, self.weight, self.eps)
