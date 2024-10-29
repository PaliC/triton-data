import sys
_module = sys.modules[__name__]
del sys
setup = _module
test_cutlass_gemm = _module
test = _module
fp8_gemm_bench = _module
fp8_rowwise_tma_persistent = _module
perf_test_moe = _module
profile_moe = _module
test_moe_gemm = _module
v0_moe_fused = _module
v1_moe_fused = _module
v2_moe_fused = _module
stay_attention = _module
scaled_fp8_gemm = _module
splitk_gemm_fp8 = _module
tma_gemm = _module
a100_qlinear = _module
benchmark = _module
h100_qlinear = _module
test_dequant_moe_gemm = _module
w4a16_fused_dequant_gemm = _module
small_benchmark_cuda_graphs = _module
splitk_dequant_gemm = _module
attention_triton = _module
softmax = _module
fused_rms_norm = _module
kernels = _module
fused_softmax = _module
vector_add = _module
test_softmax = _module
test_utils = _module

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


from typing import Callable


from typing import Tuple


import torch


import triton


from torch._tensor import Tensor


import logging


from typing import List


from typing import Optional


import triton.language as tl


from triton import Config


from triton.ops.matmul_perf_model import early_config_prune


from triton.ops.matmul_perf_model import estimate_matmul_time


from triton.runtime.jit import reinterpret as tl_reinterpret


from triton.runtime.jit import TensorWrapper


import time


from typing import Any


from typing import Dict


import functools


import numpy as np


from triton import language as tl


import torch.nn as nn


from torch import autograd


import math


@triton.jit
def _kernel_matmul_fp8_row_tma_persistent(A_ptr, B_ptr, C_ptr, M, N, K, A_scale, B_scale, stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn, dot_out_dtype: 'tl.constexpr', allow_tf32: 'tl.constexpr', fp8_fast_accum: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr', AB_DTYPE: 'tl.constexpr', NUM_SMS: 'tl.constexpr') ->None:
    """Matmul kernel of [M, K] @ [N, K] with row-wise scales

    performs swizzled matmul in [BLOCK_M, BLOCK_K] with [BLOCK_K, BLOCK_N] tiles.

    Args:
        A (TensorWrapper): [M, K] input tensor.
        B (TensorWrapper): [N, K] input tensor.
        C (TensorWrapper): [M, N] output tensor.
        M (int): M dimension of input tensor.
        N (int): N dimension of input tensor.
        K (int): K dimension of input tensor.
        A_scale (TensorWrapper): [M] reciprocal scale tensor per row. A * A_scale = original A
        B_scale (TensorWrapper): [N] reciprocal scale tensor per row. B * B_scale = original B
        stride_am (int): Stride of M dimension of A.
        stride_ak (int): Stride of K dimension of A.
        stride_bn (int): Stride of N dimension of B.
        stride_bk (int): Stride of K dimension of B.
        stride_cm (int): Stride of M dimension of C.
        stride_cn (int): Stride of N dimension of C.
        dot_out_dtype (torch.dtype): Output type of tensor core.
        allow_tf32 (bool): Whether to use TF32 for tensor core.
        fp8_fast_accum (bool): Whether to use fast accumulation for tensor core.
        BLOCK_M (int): Block size for M dimension.
        BLOCK_N (int): Block size for N dimension.
        BLOCK_K (int): Block size for K dimension.
        GROUP_M (int): Number of groups for M dimension swizzle.
        SPLIT_K (int): Number of SM's to launch per row.
        EVEN_K (bool): Whether K is evenly divisible by BLOCK_K * SPLIT_K.
        AB_DTYPE (bool): Wether to cast A and B to C.dtype before tensor core.
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1
    tile_id = start_pid - NUM_SMS
    ki = -1
    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0
    num_pid_in_group = GROUP_M * num_pid_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    dtype_fp8 = tl.float8e4nv
    scale_dtype = tl.float32
    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + tile_id % group_size_m
            pid_n = tile_id % num_pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            offs_am = tl.multiple_of(offs_am, BLOCK_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_N)
        offs_k = ki * BLOCK_K
        a = tl._experimental_descriptor_load(A_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], dtype_fp8)
        b = tl._experimental_descriptor_load(B_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], dtype_fp8)
        acc = tl.dot(a, b.T, acc, out_dtype=dot_out_dtype, allow_tf32=allow_tf32)
        if ki == k_tiles - 1:
            rm = pid_m * BLOCK_M
            rn = pid_n * BLOCK_N
            a_scale = tl._experimental_descriptor_load(A_scale, [rm], [BLOCK_M], scale_dtype)
            b_scale = tl._experimental_descriptor_load(B_scale, [rn], [BLOCK_N], scale_dtype)
            scale = a_scale[:, None] * b_scale[None, :]
            acc *= scale
            acc = acc
            tl._experimental_descriptor_store(C_ptr, acc, [rm, rn])
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)


@triton.jit()
def col_major(pid, m, n, num_tokens_post_padded, block_m: 'tl.constexpr', block_n: 'tl.constexpr'):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    pid_m = pid % grid_n
    pid_n = pid // grid_m
    return pid_m, pid_n


@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, stride_weight, stride_token_id, block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k: 'tl.constexpr', MUL_ROUTED_WEIGHT: 'tl.constexpr', top_k: 'tl.constexpr', compute_type: 'tl.constexpr'):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`. The sorting of `sorted_token_ids`
    by expert index and padding ensures divisibility by block_m, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    pid = tl.program_id(axis=0)
    pid_m, pid_n = col_major(pid, EM, N, block_m, block_n)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * block_m >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * block_m + tl.arange(0, block_m)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % N
    offs_k = tl.arange(0, block_k)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, block_k)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * block_k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * block_k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_weight, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit()
def grouped_launch(pid, m, n, block_m: 'tl.constexpr', block_n: 'tl.constexpr', group_m: 'tl.constexpr'):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    width = group_m * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + pid % group_size
    pid_n = pid % width // group_size
    return pid_m, pid_n


@triton.jit()
def column_major(pid, m, n, block_m: 'tl.constexpr', block_n: 'tl.constexpr'):
    grid_m = tl.cdiv(m, block_m)
    pid_m = pid % grid_m
    pid_n = pid // grid_m
    return pid_m, pid_n


@triton.jit
def scaled_gemm_splitk(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, scale_a, scale_b, m, n, k, block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k: 'tl.constexpr', split_k: 'tl.constexpr', group_m: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)
    pid_m, pid_n = column_major(pid, m, n, block_m, block_n)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        k_remaining = k - k_ * (block_k * split_k)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    acc = scale_a * scale_b * acc
    acc
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)


@triton.jit
def gemm_split_k_kernel(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, m, n, k, block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k: 'tl.constexpr', split_k: 'tl.constexpr', group_m: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)
    pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        k_remaining = k - k_ * (block_k * split_k)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    acc
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)


@triton.jit
def gemm_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr, prob_m, prob_n, prob_k, block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(prob_m, block_m)
    num_pid_k = tl.cdiv(prob_k, block_k)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * block_m
    offs_bn = pid_n * block_n
    offs_k = 0
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for kk in range(0, num_pid_k):
        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [block_m, block_k], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [block_n, block_k], tl.float8e4nv)
        accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
        offs_k += block_k
    accumulator = accumulator
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


@triton.jit
def print_tensor_dim(tensor, str_name):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.static_print(str_name, ' ', tensor.shape, ' ', tensor.dtype)


@triton.jit
def print_value(value):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        tl.device_print(str(value))


@triton.jit
def print_line(str_line):
    if tl.program_id(0) == 0 and tl.program_id(1) == 0:
        None


@triton.jit
def paged_attention_v1(scratchpad_key_ptr, scratchpad_value_ptr, output_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, context_lens_ptr, scale, num_seqs, num_heads, cache_block_stride, MAX_SEQ_LEN: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', HEAD_SIZE: 'tl.constexpr', MAX_NUM_BLOCKS_PER_SEQ: 'tl.constexpr'):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    query_offset = seq_idx * num_seqs + head_idx * HEAD_SIZE
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ
    context_len = tl.load(context_lens_ptr + seq_idx)
    for tok_idx in range(0, context_len):
        logical_block_idx = tok_idx // BLOCK_SIZE
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + logical_block_idx)
        start_of_block_offset = physical_block_idx * cache_block_stride + head_idx * HEAD_SIZE * BLOCK_SIZE
        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = start_of_block_offset + BLOCK_SIZE * tl.arange(0, HEAD_SIZE) + tok_idx_within_block
        tok_key = tl.load(key_cache_ptr + tok_offsets)
        tok_value = tl.load(value_cache_ptr + tok_offsets)
        scratchpad_offset = seq_idx * (MAX_SEQ_LEN * num_heads * HEAD_SIZE) + tok_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
        tl.store(scratchpad_key_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_key)
        tl.store(scratchpad_value_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_value)
    tl.debug_barrier()
    start_seq_offset = MAX_SEQ_LEN * num_heads * HEAD_SIZE * seq_idx
    start_tok_offset = start_seq_offset + tl.arange(0, MAX_SEQ_LEN) * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    mask = tl.arange(0, MAX_SEQ_LEN)[:, None] < context_len
    kv_offs = start_tok_offset[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    print_tensor_dim(kv_offs, 'kv_offs_v1')
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(keys, 'keys_v1')
    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(values, 'values_v1')
    scores = tl.sum(scale * keys * query_head[None, :], axis=1)
    mask = tl.full([MAX_SEQ_LEN], -float('inf'), dtype=tl.float32)
    cond = tl.arange(0, MAX_SEQ_LEN) < context_len
    scores_masked = tl.where(cond, scores, mask)
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0) + float(1e-06)
    logits = numerator / denominator
    print_tensor_dim(logits, 'logits_v1')
    weighted_values = tl.sum(values * logits[:, None], axis=0)
    print_tensor_dim(weighted_values, 'weighted_values_v1')
    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), weighted_values)


@triton.jit
def paged_attention_v2(scratchpad_key_ptr, scratchpad_value_ptr, partition_buf_ptr, output_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr, context_lens_ptr, scale, num_seqs, num_heads, cache_block_stride, num_partitions, PARTITION_SIZE: 'tl.constexpr', MAX_SEQ_LEN: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', HEAD_SIZE: 'tl.constexpr', MAX_NUM_BLOCKS_PER_SEQ: 'tl.constexpr'):
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    query_offset = seq_idx * num_seqs + head_idx * HEAD_SIZE
    query_head = tl.load(query_ptr + query_offset + tl.arange(0, HEAD_SIZE))
    print_tensor_dim(query_head, 'query_head')
    block_table_offset = seq_idx * MAX_NUM_BLOCKS_PER_SEQ
    context_len = tl.load(context_lens_ptr + seq_idx)
    assert context_len <= MAX_SEQ_LEN
    token_start_idx = partition_idx * PARTITION_SIZE
    token_end_idx = min((partition_idx + 1) * PARTITION_SIZE, context_len)
    for tok_idx in range(token_start_idx, token_end_idx):
        logical_block_offset = tok_idx // BLOCK_SIZE
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + logical_block_offset)
        start_of_block_offset = physical_block_idx * cache_block_stride + head_idx * HEAD_SIZE * BLOCK_SIZE
        tok_idx_within_block = tok_idx % BLOCK_SIZE
        tok_offsets = start_of_block_offset + BLOCK_SIZE * tl.arange(0, HEAD_SIZE) + tok_idx_within_block
        tok_key = tl.load(key_cache_ptr + tok_offsets)
        tok_value = tl.load(value_cache_ptr + tok_offsets)
        scratchpad_offset = seq_idx * (MAX_SEQ_LEN * num_heads * HEAD_SIZE) + tok_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
        print_tensor_dim(scratchpad_key_ptr, 'scratchpad_key_ptr')
        mask = tl.full([HEAD_SIZE], 1, dtype=tl.float32) > 0
        tl.store(scratchpad_key_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_key, mask)
        tl.store(scratchpad_value_ptr + scratchpad_offset + tl.arange(0, HEAD_SIZE), tok_value, mask)
    tl.debug_barrier()
    start_seq_offset = MAX_SEQ_LEN * num_heads * HEAD_SIZE * seq_idx
    start_tok_offsets = start_seq_offset + tl.arange(0, PARTITION_SIZE) * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    mask = tl.arange(0, PARTITION_SIZE)[:, None] < context_len
    kv_offs = start_tok_offsets[:, None] + tl.arange(0, HEAD_SIZE)[None, :]
    print_tensor_dim(kv_offs, 'kv_offs_v2')
    keys = tl.load(scratchpad_key_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(keys, 'keys_v2')
    scores = tl.sum(scale * keys * query_head[None, :], axis=1)
    print_tensor_dim(keys, 'scores_v2')
    partition_buf_offset = start_seq_offset + head_idx * HEAD_SIZE + partition_idx * PARTITION_SIZE
    print_tensor_dim(partition_buf_offset, 'partition_buf_offset_v2')
    tl.store(partition_buf_ptr + partition_buf_offset + tl.arange(0, PARTITION_SIZE), scores)
    mask = tl.full([PARTITION_SIZE], -float('inf'), dtype=tl.float32)
    cond = tl.arange(0, PARTITION_SIZE) < context_len
    scores_masked = tl.where(cond, scores, mask)
    scores_minus_max = scores_masked - tl.max(scores_masked, axis=0)
    numerator = tl.exp(scores_minus_max)
    denominator = tl.sum(numerator, axis=0) + float(1e-06)
    logits = numerator / denominator
    print_tensor_dim(logits, 'logits_v2')
    values = tl.load(scratchpad_value_ptr + kv_offs, mask=mask, other=0.0)
    print_tensor_dim(values, 'values_v2')
    weighted_values += tl.sum(values * logits[:, None], axis=0)
    print_tensor_dim(weighted_values, 'weighed_values_v2')
    output_offset = seq_idx * (num_heads * HEAD_SIZE) + head_idx * HEAD_SIZE
    tl.store(output_ptr + output_offset + tl.arange(0, HEAD_SIZE), weighted_values)


@triton.jit
def _softmax_kernel_fwd(output_ptr, output_row_stride, input_ptr, input_row_stride, n_cols, block_size: 'tl.constexpr'):
    row_index = tl.program_id(0)
    input_row_ptr = input_ptr + row_index * input_row_stride
    col_offsets = tl.arange(0, block_size)
    input_ptrs = input_row_ptr + col_offsets
    rw_mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=rw_mask, other=float('-inf'))
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denom = tl.sum(numerator, axis=0)
    sm_out = numerator / denom
    out_row_ptr = output_ptr + row_index * output_row_stride
    out_row_ptrs = out_row_ptr + col_offsets
    tl.store(out_row_ptrs, sm_out, mask=rw_mask)


@triton.jit
def _softmax_kernel_bwd(output_ptr, stride_output_row, grad_ptr, stride_grad_row, input_ptr, stride_input_row, n_cols, block_size: 'tl.constexpr'):
    row_index = tl.program_id(0)
    input_row_ptr = input_ptr + row_index * stride_input_row
    grad_row_ptr = grad_ptr + row_index * stride_grad_row
    col_offsets = tl.arange(0, block_size)
    rw_mask = col_offsets < n_cols
    input_row_ptrs = input_row_ptr + col_offsets
    grad_row_ptrs = grad_row_ptr + col_offsets
    probs_row = tl.load(input_row_ptrs, mask=rw_mask, other=0)
    grads_row = tl.load(grad_row_ptrs, mask=rw_mask, other=0)
    dx = probs_row * grads_row
    dsm_out = dx - probs_row * tl.sum(dx, axis=0)
    output_row_ptr = output_ptr + row_index * stride_output_row
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, dsm_out, mask=rw_mask)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N'])
@triton.jit
def _rms_norm_fwd_kernel(X, stride_x, Y, stride_y, W, Rstd, eps, M, N, block_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, block_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N'])
@triton.jit
def _rms_norm_bwd_kernel_sm(X, stride_x, W, DY, stride_dy, DX, stride_dx, Rstd, DW, eps, M, N, rows_per_program, block_N: 'tl.constexpr'):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, block_N)
    mask = cols < N
    w = tl.load(W + cols, mask=mask, other=0.0)
    dw = tl.zeros((block_N,), dtype=tl.float32)
    row_end = min(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0)
        rstd = tl.load(Rstd + row)
        x_hat = x * rstd
        wdy = w * dy
        dw += dy * x_hat
        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = (wdy - x_hat * c1) * rstd
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)


@triton.jit
def kernel_vector_addition(a_ptr, b_ptr, out_ptr, num_elems: 'tl.constexpr', block_size: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    block_start = pid * block_size
    thread_offsets = block_start + tl.arange(0, block_size)
    mask = thread_offsets < num_elems
    a_pointers = tl.load(a_ptr + thread_offsets, mask=mask)
    b_pointers = tl.load(b_ptr + thread_offsets, mask=mask)
    res = a_pointers + b_pointers
    tl.store(out_ptr + thread_offsets, res, mask=mask)

