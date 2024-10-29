import sys
_module = sys.modules[__name__]
del sys
build_fastfold_wheel = _module
perf = _module
demo = _module
fastfold = _module
common = _module
protein = _module
residue_constants = _module
config = _module
data = _module
data_modules = _module
data_pipeline = _module
data_transforms = _module
data_transforms_multimer = _module
errors = _module
feature_pipeline = _module
feature_processing_multimer = _module
input_pipeline = _module
input_pipeline_multimer = _module
mmcif_parsing = _module
msa_identifiers = _module
msa_pairing = _module
parsers = _module
templates = _module
tools = _module
hhblits = _module
hhsearch = _module
hmmbuild = _module
hmmsearch = _module
jackhmmer = _module
kalign = _module
utils = _module
distributed = _module
comm = _module
comm_async = _module
core = _module
habana = _module
fastnn = _module
custom_op = _module
fusedsoftmax = _module
hpu_fusedsoftmax_test = _module
setup = _module
setup2 = _module
initializer = _module
kernel = _module
msa = _module
ops = _module
triangle = _module
inject_habana = _module
model = _module
embedders = _module
embedders_multimer = _module
evoformer = _module
attention_core = _module
cuda_native = _module
layer_norm = _module
softmax = _module
jit = _module
fused_ops = _module
options = _module
layer_norm = _module
softmax = _module
triton = _module
attention_core = _module
layer_norm = _module
softmax = _module
template = _module
hub = _module
alphafold = _module
loss = _module
lr_scheduler = _module
nn = _module
dropout = _module
heads = _module
outer_product_mean = _module
pair_transition = _module
primitives = _module
structure_module = _module
triangular_attention = _module
triangular_multiplicative_update = _module
relax = _module
amber_minimize = _module
cleanup = _module
all_atom_multimer = _module
checkpointing = _module
feats = _module
geometry = _module
quat_rigid = _module
rigid_matrix_vector = _module
rotation_matrix = _module
test_utils = _module
vector = _module
import_weights = _module
inject_fastnn = _module
rigid_utils = _module
superimposition = _module
tensor_utils = _module
validation_utils = _module
workflow = _module
factory = _module
hhfilter = _module
task_factory = _module
fastfold_data_workflow = _module
fastfold_multimer_data_workflow = _module
workflow_run = _module
hpuhelper = _module
inference = _module
inference_test = _module
train = _module
tests = _module
test_attention_core = _module
test_basic_ops = _module
test_evoformer = _module
test_evoformer_stack = _module
test_extramsa_stack = _module
test_layernorm = _module
test_msa_att_col = _module
test_msa_att_row = _module
test_msa_global_att_col = _module
test_out_product_mean = _module
test_softmax = _module
test_template_embedder = _module
test_inference = _module
test_train = _module

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


import math


import logging


import torch


import numbers


from torch.nn.parameter import Parameter


from functools import reduce


import triton


import triton.language as tl


@triton.jit
def _attention_core(Q, K, V, mask, bias, sm_scale, TMP, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, BATCH, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', use_mask: 'tl.constexpr', use_bias: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off_hz * stride_qh + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    if use_bias:
        batch_2 = Z // BATCH
        off_hz_bias = off_hz // (batch_2 * H) * H + off_hz % H
        offs_base_bias = off_hz_bias * (N_CTX * N_CTX) + offs_m[:, None] * N_CTX + offs_n[None, :]
    if use_mask:
        off_hz_mask = off_hz // H
        offs_base_mask = off_hz_mask * N_CTX
    t_ptrs = TMP + off_hz * N_CTX + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q_load_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        load_mask = (start_n + offs_n)[:, None] < N_CTX
        k = tl.load(k_ptrs + start_n * stride_kn, mask=load_mask, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= N_CTX, float('-1e20'), qk)
        qk = tl.where((start_n + offs_n)[None, :] >= N_CTX, float('-1e20'), qk)
        if use_bias:
            bias_load_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            bias_load_mask = tl.where(offs_m[:, None] >= N_CTX, 1.0, bias_load_mask)
            bias_load_mask = tl.where((start_n + offs_n)[None, :] >= N_CTX, 1.0, bias_load_mask)
            bias_data = tl.load(bias + offs_base_bias + start_n, mask=bias_load_mask == 0.0, other=0.0)
            qk += bias_data
        if use_mask:
            mask_data = tl.load(mask + offs_base_mask + offs_n + start_n, mask=start_n + offs_n < N_CTX, other=0.0)
            qk = tl.where(mask_data[None, :] == 0.0, float('-1e20'), qk)
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale, mask=offs_m < N_CTX)
        acc_scale = tl.load(TMP + off_hz * N_CTX + start_m * BLOCK_M + tl.arange(0, BLOCK_M), mask=start_m * BLOCK_M + tl.arange(0, BLOCK_M) < N_CTX, other=float(0.0))
        acc = acc * acc_scale[:, None]
        load_mask = (start_n + offs_n)[:, None] < N_CTX
        v = tl.load(v_ptrs + start_n * stride_vn, mask=load_mask, other=0.0)
        p = p
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    out_store_mask = offs_m[:, None] < N_CTX
    tl.store(out_ptrs, acc, mask=out_store_mask)


@triton.jit
def _layer_norm_fwd_fused(Out, A, Weight, Bias, Mean, Rstd, stride, N, eps, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    Out += row * stride
    A += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(A + cols, mask=cols < N, other=0.0)
        a = tl.where(cols < N, a - mean, 0.0)
        _var += a * a
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        weight = tl.load(Weight + cols, mask=mask)
        bias = tl.load(Bias + cols, mask=mask)
        a = tl.load(A + cols, mask=mask, other=0.0)
        a_hat = (a - mean) * rstd
        out = a_hat * weight + bias
        tl.store(Out + cols, out, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(_DA, _DOut, _A, Weight, Mean, Rstd, stride, NumRows, NumCols, eps, BLOCK_SIZE_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    row = pid
    A = _A + row * stride
    DOut = _DOut + row * stride
    DA = _DA + row * stride
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    _mean1 = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    _mean2 = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    for off in range(0, NumCols, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < NumCols
        a = tl.load(A + cols, mask=mask, other=0)
        dout = tl.load(DOut + cols, mask=mask, other=0)
        weight = tl.load(Weight + cols, mask=mask, other=0)
        a_hat = (a - mean) * rstd
        wdout = weight * dout
        _mean1 += a_hat * wdout
        _mean2 += wdout
    mean1 = tl.sum(_mean1, axis=0) / NumCols
    mean2 = 0.0
    mean2 = tl.sum(_mean2, axis=0) / NumCols
    for off in range(0, NumCols, BLOCK_SIZE_N):
        cols = off + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < NumCols
        a = tl.load(A + cols, mask=mask, other=0)
        dout = tl.load(DOut + cols, mask=mask, other=0)
        weight = tl.load(Weight + cols, mask=mask, other=0)
        a_hat = (a - mean) * rstd
        wdout = weight * dout
        da = (wdout - (a_hat * mean1 + mean2)) * rstd
        tl.store(DA + cols, da, mask=mask)


@triton.jit
def _layer_norm_bwd_dwdb(A, DOut, Mean, Var, DW, DB, M, N, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    UNROLL: 'tl.constexpr' = 4
    for i in range(0, M, BLOCK_SIZE_M * UNROLL):
        for j in range(UNROLL):
            rows = i + j * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            a = tl.load(A + offs, mask=mask, other=0.0)
            dout = tl.load(DOut + offs, mask=mask, other=0.0)
            mean = tl.load(Mean + rows, mask=rows < M, other=0.0)
            rstd = tl.load(Var + rows, mask=rows < M, other=0.0)
            a_hat = (a - mean[:, None]) * rstd[:, None]
            dw += dout * a_hat
            db += dout
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(DW + cols, sum_dw, mask=cols < N)
    tl.store(DB + cols, sum_db, mask=cols < N)


@triton.jit
def _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs, col_offsets, n_cols, use_mask: 'tl.constexpr', use_bias: 'tl.constexpr'):
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    if use_bias:
        bias = tl.load(bias_ptrs, mask=col_offsets < n_cols, other=float('-inf'))
        row += bias
    if use_mask:
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=float('-inf'))
        row = tl.where(mask == 0, float('-1e20'), row)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


@triton.jit
def _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs, col_offsets, n_cols, is_bf16: 'tl.constexpr'):
    output_row = tl.load(output_ptrs, mask=col_offsets < n_cols, other=float(0))
    d_output_row = tl.load(d_output_ptrs, mask=col_offsets < n_cols, other=float(0))
    if is_bf16:
        output_row = output_row
        d_output_row = d_output_row
    row_sum = tl.sum(output_row * d_output_row, axis=0)
    d_softmax_output = (d_output_row - row_sum) * output_row
    tl.store(d_input_ptrs, d_softmax_output, mask=col_offsets < n_cols)


@triton.jit
def softmax_mask_bias_kernel(output_ptr, input_ptr, mask_ptr, bias_ptr, input_row_stride, output_row_stride, n_cols, n_heads, BLOCK_SIZE: 'tl.constexpr', use_mask: 'tl.constexpr', use_bias: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride
    input_ptrs = input_row_ptr + col_offsets
    output_ptrs = output_row_ptr + col_offsets
    mask_ptrs = input_ptrs
    if use_mask:
        mask_row_ptr = mask_ptr + row_idx // (n_heads * n_cols) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets
    bias_ptrs = input_ptrs
    if use_bias:
        bias_row_ptr = bias_ptr + row_idx % (n_heads * n_cols) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets
    _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs, col_offsets, n_cols, use_mask, use_bias)


@triton.jit
def softmax_mask_bias_kernel_two_rows(output_ptr, input_ptr, mask_ptr, bias_ptr, input_row_stride, output_row_stride, n_cols, n_heads, BLOCK_SIZE: 'tl.constexpr', use_mask: 'tl.constexpr', use_bias: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_row_ptr = input_ptr + 2 * row_idx * input_row_stride
    output_row_ptr = output_ptr + 2 * row_idx * output_row_stride
    input_ptrs = input_row_ptr + col_offsets
    output_ptrs = output_row_ptr + col_offsets
    mask_ptrs = input_ptrs
    if use_mask:
        mask_row_ptr = mask_ptr + 2 * row_idx // (n_heads * n_cols) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets
    bias_ptrs = input_ptrs
    if use_bias:
        bias_row_ptr = bias_ptr + 2 * row_idx % (n_heads * n_cols) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets
    _softmax_core(input_ptrs, output_ptrs, mask_ptrs, bias_ptrs, col_offsets, n_cols, use_mask, use_bias)
    mask_ptrs = input_ptrs
    if use_mask:
        mask_row_ptr = mask_ptr + (2 * row_idx + 1) // (n_heads * n_cols) * n_cols
        mask_ptrs = mask_row_ptr + col_offsets
    bias_ptrs = input_ptrs
    if use_bias:
        bias_row_ptr = bias_ptr + (2 * row_idx + 1) % (n_heads * n_cols) * n_cols
        bias_ptrs = bias_row_ptr + col_offsets
    _softmax_core(input_ptrs + n_cols, output_ptrs + n_cols, mask_ptrs, bias_ptrs, col_offsets, n_cols, use_mask, use_bias)


@triton.jit
def softmax_grad_kernel(d_output_ptr, output_ptr, d_input_ptr, d_output_row_stride, output_row_stride, d_input_row_stride, n_cols, BLOCK_SIZE: 'tl.constexpr', is_bf16: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_row_ptr = output_ptr + row_idx * output_row_stride
    d_output_row_ptr = d_output_ptr + row_idx * d_output_row_stride
    d_input_row_ptr = d_input_ptr + row_idx * d_input_row_stride
    output_ptrs = output_row_ptr + col_offsets
    d_output_ptrs = d_output_row_ptr + col_offsets
    d_input_ptrs = d_input_row_ptr + col_offsets
    _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs, col_offsets, n_cols, is_bf16)


@triton.jit
def softmax_grad_kernel_two_rows(d_output_ptr, output_ptr, d_input_ptr, d_output_row_stride, output_row_stride, d_input_row_stride, n_cols, BLOCK_SIZE: 'tl.constexpr', is_bf16: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_row_ptr = output_ptr + 2 * row_idx * output_row_stride
    d_output_row_ptr = d_output_ptr + 2 * row_idx * d_output_row_stride
    d_input_row_ptr = d_input_ptr + 2 * row_idx * d_input_row_stride
    output_ptrs = output_row_ptr + col_offsets
    d_output_ptrs = d_output_row_ptr + col_offsets
    d_input_ptrs = d_input_row_ptr + col_offsets
    _softmax_grad_core(output_ptrs, d_output_ptrs, d_input_ptrs, col_offsets, n_cols, is_bf16)
    _softmax_grad_core(output_ptrs + n_cols, d_output_ptrs + n_cols, d_input_ptrs + n_cols, col_offsets, n_cols, is_bf16)

