import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
benchmarks_visualizer = _module
scripts = _module
benchmark_cross_entropy = _module
benchmark_embedding = _module
benchmark_fused_linear_cross_entropy = _module
benchmark_fused_linear_jsd = _module
benchmark_geglu = _module
benchmark_jsd = _module
benchmark_kl_div = _module
benchmark_layer_norm = _module
benchmark_rms_norm = _module
benchmark_rope = _module
benchmark_swiglu = _module
utils = _module
callback = _module
training = _module
medusa_util = _module
train = _module
env_report = _module
ops = _module
cross_entropy = _module
embedding = _module
mm_int8int2 = _module
fused_linear_cross_entropy = _module
fused_linear_jsd = _module
geglu = _module
jsd = _module
kl_div = _module
layer_norm = _module
rms_norm = _module
rope = _module
swiglu = _module
utils = _module
transformers = _module
auto_model = _module
functional = _module
model = _module
gemma = _module
llama = _module
mistral = _module
mixtral = _module
mllama = _module
phi3 = _module
qwen2 = _module
qwen2_vl = _module
monkey_patch = _module
trainer_integration = _module
triton = _module
monkey_patch = _module
test = _module
conftest = _module
convergence = _module
test_mini_models = _module
test_mini_models_multimodal = _module
test_mini_models_no_logits = _module
generate_tokenized_dataset = _module
test_auto_model = _module
test_cross_entropy = _module
test_embedding = _module
test_fused_linear_cross_entropy = _module
test_fused_linear_jsd = _module
test_geglu = _module
test_jsd = _module
test_kl_div = _module
test_layer_norm = _module
test_mm_int8int2 = _module
test_monkey_patch = _module
test_rms_norm = _module
test_rope = _module
test_swiglu = _module
test_trainer_integration = _module
test_transformers = _module
test_triton_monkey_patch = _module

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


from torch.nn import CrossEntropyLoss


from torch.nn import Embedding


import torch.nn as nn


import triton.language as tl


from typing import Optional


from typing import Literal


import math


import functools


from typing import Callable


import random


from triton.runtime.cache import FileCacheManager


@triton.jit
def liger_cross_entropy_kernel(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride, n_cols, n_non_ignore, ignore_index, label_smoothing: 'tl.constexpr', reduction: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Parameters:
    X_ptr: Pointer to input tensor.
    X_stride (int): The stride of the input tensor.
    Y_ptr: Pointer to target tensor.
    Y_stride (int): The stride of the target tensor.
    loss_ptr: Pointer to tensor to store the loss.
    loss_stride (int): The stride of the loss tensor.
    n_cols (int): The number of columns in the input tensor.
    n_non_ignore (int): The number of non-ignored elements in the batch.
    ignore_index (int): The index to ignore in the target.
    label_smoothing (float): The amount of smoothing when computing the loss, where 0.0 means no smoothing.
    reduction (str): The string for the reduction to apply
    BLOCK_SIZE (int): The block size for Triton operations.
    """
    program_id = tl.program_id(0)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride
    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    loss_ptr += program_id * loss_stride
    m = float('-inf')
    d = 0.0
    ori_X_y = tl.load(X_ptr + y)
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float('-inf'))
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float('-inf'))
        if reduction == 'mean':
            X_block = (tl.exp(X_block - m) / d - eps) / n_non_ignore
        else:
            X_block = tl.exp(X_block - m) / d - eps
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)
    tl.debug_barrier()
    loss = -(ori_X_y - m - tl.log(d))
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    if reduction == 'mean':
        loss = loss / n_non_ignore
    X_y = tl.load(X_ptr + y)
    if reduction == 'mean':
        X_y += -(1 - label_smoothing) / n_non_ignore
    else:
        X_y += -(1 - label_smoothing)
    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


@triton.jit
def embedding_forward_kernel(embeddings_ptr, indices_ptr, output_ptr, n_elements, embedding_dim: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim
    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(embeddings_ptr + embedding_offsets, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(output_ptr + output_offsets, embeddings, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def embedding_backward_kernel(grad_output_ptr, grad_weight_ptr, indices_ptr, n_elements, embedding_dim: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim
    grad_output = tl.load(grad_output_ptr + offsets_m[:, None] * embedding_dim + offsets_n[None, :], mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    tl.atomic_add(grad_weight_ptr + grad_weight_offsets, grad_output, mask=mask_m[:, None] & mask_n[None, :])


def get_autotune_config():
    return [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4)]


@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K: 'tl.constexpr', stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'):
    tl.static_assert(K % (4 * BLOCK_SIZE_K) == 0, 'K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K')
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    """
        This part of the code generates pointers to the specific blocks of matrices A and B that the current thread block will process.

        As described in the PyTorch documentation, a stride refers to the step size needed to move from one element to the next along a given dimension:

        For matrix A: stride_am = A.stride(0) = K (stride along the rows), and stride_ak = A.stride(1) = 1 (stride along the columns).
        For matrix B: stride_bk = B.stride(0) = N (stride along the rows), and stride_bn = B.stride(1) = 1 (stride along the columns).
        Now, let's break down the pointer generation:

        offs_am[:, None] creates a column of shape [BLOCK_SIZE_M, 1], which represents the row indices of matrix A that this block is processing. It is multiplied by K (the number of columns in matrix A) since A is stored in row-major order. So, the element at position (i, j) in A is located at index i*K + j in memory.
        offs_k[None, BLOCK_SIZE_K] creates a row vector representing the column indices of the block, i.e., a range from 0 to BLOCK_SIZE_K. This is used to compute the positions of the columns within the block.
        When combined, the result has the shape [BLOCK_SIZE_M, BLOCK_SIZE_K], where each entry (i, j) points to the element in matrix A at position (i, j) for the current block.

        The same logic is applied to matrix B, but the resulting shape is [BLOCK_SIZE_K, BLOCK_SIZE_N], representing the block of matrix B that the thread block will work on.
    """
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    """
        We split the loop into two layers. The outer loop runs 4 times, and each iteration focuses on a specific portion of matrix A.

        For example, when i = 0, we’re only concerned with the blocks of matrix A that cover the range from 0 to K // (4 * BLOCK_SIZE_K).
        Since matrix B is packed, its first dimension is effectively divided by 4. So, while we process the first segment of matrix A,
        we still iterate over the entire first dimension of matrix B.

        In each of the 4 iterations of the outer loop, we go through the full blocks of matrix B, but what changes is the data we extract.
        Matrix B elements contain 4 weights, all packed into an int8 format, and during each iteration of the outer loop,
        we extract a different weight by using bitwise shifting operations. This way, we access a unique weight on each pass.
    """
    for i in range(4):
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K)):
            k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
            mask = 3 << 2 * i
            b = (b_uint8 & mask) >> 2 * i
            tensor_full = tl.full((1,), 1, dtype=tl.int8)
            accumulator += tl.dot(a, b - tensor_full, out_dtype=tl.int32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel(dc, a, b, stride, n_cols: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    db_row = dc_row * geglu_a
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    da_row = dc_row * b_row * (term1 + term2)
    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)


@triton.jit
def _jsd_kernel(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride, dX_ptr, dX_stride, label_ptr, beta, n_non_ignore, ignore_index: 'tl.constexpr', n_cols, BLOCK_SIZE: 'tl.constexpr', HAS_LABEL: 'tl.constexpr'):
    pid = tl.program_id(0)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride
    label_ptr += pid
    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(dX_ptr + offsets, 0.0, mask=offsets < n_cols)
            return
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=float('-inf'))
        Y = tl.load(Y_ptr + offsets, mask=mask, other=float('-inf'))
        Q = tl.exp(X)
        P = tl.exp(Y)
        M = beta * P + (1 - beta) * Q
        log_M = tl.log(M)
        loss = beta * P * Y + (1 - beta) * Q * X - M * log_M
        loss = loss / n_non_ignore
        tl.store(loss_ptr + offsets, loss, mask=mask)
        dX = (1 - beta) * Q * (X - log_M) / n_non_ignore
        tl.store(dX_ptr + offsets, dX, mask=mask)


_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)


_REDUCTION_MODE_NONE = tl.constexpr(0)


@triton.jit
def _kldiv_kernel_forward(y_ptr, y_stride, gt_ptr, gt_stride, loss_ptr, loss_stride, n_cols, eps, BLOCK_SIZE: 'tl.constexpr', log_target: 'tl.constexpr'=False, reduction: 'tl.constexpr'=_REDUCTION_MODE_BATCHMEAN):
    pid = tl.program_id(0)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += pid * loss_stride
    base_offsets = tl.arange(0, BLOCK_SIZE)
    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss_sum += tl.sum(loss, axis=0)
    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)


@triton.jit
def _kldiv_kernel_backward(target_ptr, target_stride, new_grads_ptr, new_grads_stride, n_cols, BLOCK_SIZE: 'tl.constexpr', log_target: 'tl.constexpr'=False):
    pid = tl.program_id(0)
    target_ptr += pid * target_stride
    new_grads_ptr += pid * new_grads_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)
        tl.store(new_grads_ptr + offsets, res, mask=mask)


@triton.jit
def _layer_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride, W_ptr, W_row_stride, B_ptr, B_row_stride, Mean_ptr, Mean_row_stride, RSTD_ptr, RSTD_row_stride, n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)
    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / n_cols
    rstd = rsqrt(var + eps)
    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)
    Y_row = (X_row - mean) * rstd * W_row + B_row
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _layer_norm_backward_kernel(X_ptr, W_ptr, Mean_ptr, RSTD_ptr, DX_ptr, DW_ptr, DB_ptr, DY_ptr, stride_x, stride_dx, stride_dw, stride_db, stride_dy, n_rows, n_cols, rows_per_program: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', dtype: 'tl.constexpr'):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy
    for _ in range(row_start, row_end):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)
        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx, mask=mask)
        dw_row += dy * x_hat
        db_row += dy
        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy
    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row, mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row, mask=mask)


_CASTING_MODE_GEMMA = tl.constexpr(1)


_CASTING_MODE_LLAMA = tl.constexpr(0)


_CASTING_MODE_NONE = tl.constexpr(-1)


@triton.jit
def _rms_norm_forward_kernel(Y_ptr, Y_row_stride, X_ptr, X_row_stride, W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride, n_cols, eps, offset, casting_mode: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row
    if casting_mode == _CASTING_MODE_GEMMA:
        W_row = W_row
        X_row = X_row
    if casting_mode == _CASTING_MODE_NONE:
        eps = eps
        offset = offset
    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = rsqrt(mean_square + eps)
    tl.store(RSTD_ptr, rstd)
    X_row = X_row * rstd
    if casting_mode == _CASTING_MODE_LLAMA:
        X_row = X_row
    Y_row = X_row * (offset + W_row)
    if casting_mode == _CASTING_MODE_GEMMA:
        Y_row = Y_row
    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(dY_ptr, dY_row_stride, X_ptr, X_row_stride, X_dtype: 'tl.constexpr', W_ptr, W_row_stride, RSTD_ptr, RSTD_row_stride, dW_ptr, dW_row_stride, n_rows, n_cols, offset, rows_per_program: 'tl.constexpr', casting_mode: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    dx = (1 / RMS) * [dy * (w + offset - (1 / N) * (1 / RMS^2) * ((dy * (w + offset)) dot x) * x]. * means element-wise multiplication, whileas dot means dot product
    dw = sum(dy * (x / RMS)). summation over BxT dimension
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    dY_ptr += row_start * dY_row_stride
    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
    W_row = W_row + offset
    for _ in range(row_start, row_end):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(RSTD_ptr)
        X_row = X_row
        if casting_mode == _CASTING_MODE_LLAMA:
            m = dY_row * W_row
        elif casting_mode == _CASTING_MODE_GEMMA:
            dY_row = dY_row
            m = dY_row * W_row
        else:
            m = dY_row * W_row
        dX_row = rstd_row * m
        dX_row += rstd_row * (-(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row)
        if casting_mode == _CASTING_MODE_LLAMA:
            dW_row += dY_row * (X_row * rstd_row)
        else:
            dW_row += dY_row * (X_row * rstd_row)
        tl.store(dY_ptr + col_offsets, dX_row, mask=mask)
        dY_ptr += dY_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride
    tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


@triton.jit
def _triton_rope(q_ptr, q_row_stride, k_ptr, k_row_stride, cos, cos_row_stride, sin, sin_row_stride, sl, bs: 'tl.constexpr', n_qh: 'tl.constexpr', n_kh: 'tl.constexpr', hd: 'tl.constexpr', pad_n_qh: 'tl.constexpr', pad_n_kh: 'tl.constexpr', pad_hd: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', BACKWARD_PASS: 'tl.constexpr'=False):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride
    cos_row_idx = pid % sl
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0)
    second_half_q_offsets = first_half_q_offsets + hd // 2
    second_half_k_offsets = first_half_k_offsets + hd // 2
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0)
    if not BACKWARD_PASS:
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(a_ptr, b_ptr, c_ptr, stride, n_cols: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


@triton.jit
def _swiglu_backward_kernel(dc_ptr, a_ptr, b_ptr, stride, n_cols: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)


@triton.jit
def element_mul_kernel(X_ptr, X_stride, grad_output_ptr, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    """
    This function multiplies each element of the tensor pointed by X_ptr with the value pointed by grad_output_ptr.
    The multiplication is performed in-place on the tensor pointed by X_ptr.

    Parameters:
    X_ptr: Pointer to the input tensor.
    X_stride (int): The stride of the input tensor.
    grad_output_ptr: Pointer to the gradient output value.
    n_cols (int): The number of columns in the input tensor.
    BLOCK_SIZE (int): The block size for Triton operations.
    """
    program_id = tl.program_id(0)
    X_ptr += program_id * X_stride
    grad_output = tl.load(grad_output_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)

