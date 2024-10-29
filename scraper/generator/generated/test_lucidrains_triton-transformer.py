import sys
_module = sys.modules[__name__]
del sys
assert = _module
setup = _module
train = _module
triton_transformer = _module
autoregressive_wrapper = _module
bmm = _module
cross_entropy = _module
dropout = _module
layernorm = _module
softmax = _module
transformer = _module
utils = _module

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


from torch import autograd


import torch.nn.functional as F


import triton


import triton.language as tl


from random import randrange


def exists(val):
    return val is not None


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2)], key=['M', 'N', 'K'])
@triton.jit
def bmm_kernel(x_ptr, y_ptr, o_ptr, M, N, K, stride_al, stride_am, stride_ak, stride_bl, stride_bk, stride_bn, stride_ol, stride_om, stride_on, **meta):
    BLOCK_SIZE_M = meta['BLOCK_SIZE_M']
    BLOCK_SIZE_N = meta['BLOCK_SIZE_N']
    BLOCK_SIZE_K = meta['BLOCK_SIZE_K']
    GROUP_SIZE_M = 8
    pid_batch = tl.program_id(0)
    pid = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak + pid_batch * stride_al)
    y_ptrs = y_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn + pid_batch * stride_bl)
    o = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        o += tl.dot(x, y)
        x_ptrs += BLOCK_SIZE_K * stride_ak
        y_ptrs += BLOCK_SIZE_K * stride_bk
    if exists(meta['ACTIVATION']):
        o = meta['ACTIVATION'](o)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + stride_ol * pid_batch
    tl.store(o_ptrs, o, mask=mask)


@triton.jit
def relu_squared_activation(x):
    return tl.where(x > 0, x * x, 0.0)


@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, **meta):
    BLOCK_SIZE = meta['BLOCK_SIZE']
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * 4
    off0 = block_start + BLOCK_SIZE * 0 + tl.arange(0, BLOCK_SIZE)
    off1 = block_start + BLOCK_SIZE * 1 + tl.arange(0, BLOCK_SIZE)
    off2 = block_start + BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE)
    off3 = block_start + BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE)
    mask0 = off0 < n_elements
    mask1 = off1 < n_elements
    mask2 = off2 < n_elements
    mask3 = off3 < n_elements
    x0 = tl.load(x_ptr + off0, mask=mask0)
    x1 = tl.load(x_ptr + off1, mask=mask1)
    x2 = tl.load(x_ptr + off2, mask=mask2)
    x3 = tl.load(x_ptr + off3, mask=mask3)
    r0, r1, r2, r3 = tl.random.rand4x(seed, off0)
    keep0, keep1, keep2, keep3 = r0 > p, r1 > p, r2 > p, r3 > p
    o0 = tl.where(keep0, x0 / (1 - p), 0.0)
    o1 = tl.where(keep1, x1 / (1 - p), 0.0)
    o2 = tl.where(keep2, x2 / (1 - p), 0.0)
    o3 = tl.where(keep3, x3 / (1 - p), 0.0)
    tl.store(output_ptr + off0, o0, mask=mask0)
    tl.store(output_ptr + off1, o1, mask=mask1)
    tl.store(output_ptr + off2, o2, mask=mask2)
    tl.store(output_ptr + off3, o3, mask=mask3)


@triton.jit
def layernorm_kernel_forward_training(output_ptr, mean_centered_ptr, normed_ptr, input_ptr, gamma_ptr, input_row_stride, gamma_row_stride, output_row_stride, mean_centered_row_stride, normed_row_stride, n_cols, stable, eps, **meta):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']
    row_start_ptr = input_ptr + row_idx * input_row_stride
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.0)
    if stable:
        row_max = tl.max(tl.where(mask, row, float('-inf')), axis=0)
        row /= row_max
    row_mean = tl.sum(row, axis=0) / n_cols
    row_mean_centered = tl.where(mask, row - row_mean, 0.0)
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var
    output = normed * gammas
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)
    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets
    tl.store(mean_centered_ptrs, row_mean_centered, mask=mask)
    normed_row_start_ptr = normed_ptr + row_idx * normed_row_stride
    normed_ptrs = normed_row_start_ptr + col_offsets
    tl.store(normed_ptrs, normed, mask=mask)


@triton.jit
def layernorm_kernel_forward_inference(output_ptr, input_ptr, gamma_ptr, input_row_stride, gamma_row_stride, output_row_stride, n_cols, stable, eps, **meta):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']
    row_start_ptr = input_ptr + row_idx * input_row_stride
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.0)
    if stable:
        row_max = tl.max(tl.where(mask, row, float('-inf')), axis=0)
        row /= row_max
    row_mean = tl.sum(row, axis=0) / n_cols
    row_mean_centered = tl.where(mask, row - row_mean, 0.0)
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var
    output = normed * gammas
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def layernorm_kernel_backward(output_ptr, dy_ptr, mean_centered_ptr, output_row_stride, dy_row_stride, mean_centered_row_stride, n_cols, eps, **meta):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']
    dy_row_start_ptr = dy_ptr + row_idx * dy_row_stride
    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    dy_ptrs = dy_row_start_ptr + col_offsets
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    dy = tl.load(dy_ptrs, mask=mask, other=0.0)
    mean_centered = tl.load(mean_centered_ptrs, mask=mask, other=0.0)
    row_var = tl.sum(mean_centered * mean_centered, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(row_var + eps)
    normed = mean_centered * inv_var
    output = 1.0 / n_cols * inv_var * (n_cols * dy - tl.sum(dy, axis=0) - normed * tl.sum(dy * normed, axis=0))
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def layernorm_gamma_kernel_backward(dgamma_ptr, norm_ptr, dy_ptr, norm_stride, dy_stride, dgamma_row_stride, n_rows, n_cols, **meta):
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    BLOCK_SIZE = meta['BLOCK_SIZE']
    ROW_BLOCK_SIZE = meta['BLOCK_SIZE_ROW']
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, ROW_BLOCK_SIZE)
    col_range = col_idx * BLOCK_SIZE + col_offsets
    row_range = row_idx * ROW_BLOCK_SIZE + row_offsets
    col_mask = col_range < n_cols
    mask = (row_range < n_rows)[:, None] & col_mask[None, :]
    dy_ptr += row_range[:, None] * dy_stride + col_range[None, :]
    norm_ptr += row_range[:, None] * norm_stride + col_range[None, :]
    dy = tl.load(dy_ptr, mask=mask, other=0.0)
    norm = tl.load(norm_ptr, mask=mask, other=0.0)
    dgamma = tl.sum(dy * norm, axis=0)
    dgamma_ptr += row_idx * dgamma_row_stride + col_range
    tl.store(dgamma_ptr, dgamma, mask=col_mask)


@triton.jit
def softmax_kernel_forward(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, causal, **meta):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    if causal:
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
def softmax_kernel_backward(output_ptr, input_ptr, grad_ptr, grad_row_stride, input_row_stride, output_row_stride, n_cols, **meta):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta['BLOCK_SIZE']
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

