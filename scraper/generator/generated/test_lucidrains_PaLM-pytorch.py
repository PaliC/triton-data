import sys
_module = sys.modules[__name__]
del sys
train = _module
palm_pytorch = _module
autoregressive_wrapper = _module
palm_lite = _module
triton = _module
layernorm = _module
palm = _module
softmax = _module
setup = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, time, torch, torchaudio, torchvision, triton, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import torch


import triton


import triton.language as tl


import torch.nn.functional as F


from torch import einsum


from torch import nn


from torch import autograd


import random


import numpy as np


import torch.optim as optim


from torch.nn import functional as F


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


@triton.jit
def _layer_norm_fwd_fused(X, Y, W, M, V, stride, N, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    X += row * stride
    Y += row * stride
    x = tl.load(X + cols, mask=mask, other=0)
    mean = tl.sum(x, axis=0) / N
    xmean = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xmean * xmean, axis=0) / N
    rstd = 1 / tl.sqrt(var + 1e-05)
    xhat = xmean * rstd
    tl.store(M + row, mean)
    tl.store(V + row, rstd)
    w = tl.load(W + cols, mask=mask)
    y = xhat * w
    tl.store(Y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, X, W, M, V, Lock, stride, N, GROUP_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0)
    dy = tl.load(DY + cols, mask=mask, other=0)
    w = tl.load(W + cols, mask=mask)
    mean = tl.load(M + row)
    rstd = tl.load(V + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = dy * xhat
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dw(DW, FINAL_DW, M, N, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


@triton.jit
def softmax_kernel_forward(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    causal_mask = col_offsets > row_idx % n_cols
    row = row + tl.where(causal_mask, -float('inf'), 0.0)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def softmax_kernel_backward(output_ptr, input_ptr, grad_ptr, grad_row_stride, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    grad_row_start_ptr = grad_ptr + row_idx * grad_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    grad_ptrs = grad_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    probs_row = tl.load(input_ptrs, mask=mask, other=0.0)
    grad_row = tl.load(grad_ptrs, mask=mask, other=0.0)
    dxhat = probs_row * grad_row
    softmax_grad_output = dxhat - probs_row * tl.sum(dxhat, axis=0)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_grad_output, mask=mask)

