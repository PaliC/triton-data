import sys
_module = sys.modules[__name__]
del sys
gpu_benchmark_diff = _module
run_benchmark_wrapper = _module
conf = _module
my_model = _module
cifar_MetaFormer = _module
cifar_ViT = _module
generate = _module
model = _module
mp_utils = _module
sample_utils = _module
stats = _module
tokenizer = _module
microGPT = _module
build_conda = _module
compute_wheel_version = _module
setup = _module
torch_stub_tests = _module
tests = _module
multiprocessing_utils = _module
test_attention_mask = _module
test_attention_patterns = _module
test_attention_utils = _module
test_attentions = _module
test_checkpoint = _module
test_compositional_attention = _module
test_core_attention = _module
test_custom_ops = _module
test_embedding = _module
test_favor = _module
test_feedforward = _module
test_global_attention = _module
test_hydra_helper = _module
test_indexing = _module
test_ipc = _module
test_mem_eff_attention = _module
test_multiprocessing_utils = _module
test_nystrom_attention = _module
test_ortho_attention = _module
test_pickling = _module
test_profiler = _module
test_residual = _module
test_reversible = _module
test_rmsnorm = _module
test_rope_padded = _module
test_rotary_embeddings = _module
test_seqpar = _module
test_sequence_parallel_fused_ops = _module
test_sparse_tensors = _module
test_sparsecs = _module
test_sparsity24 = _module
test_splitk_reference = _module
test_swiglu = _module
test_tiled_matmul = _module
test_triton_varargs = _module
test_unbind = _module
utils = _module
xformers = _module
_cpp_lib = _module
_deprecation_warning = _module
attn_bias_utils = _module
LRA = _module
batch_fetch_results = _module
batch_submit = _module
code = _module
dataset = _module
model_wrapper = _module
run_grid_search = _module
run_tasks = _module
run_with_submitit = _module
cifar10 = _module
listops = _module
pathfinder = _module
retrieval = _module
text = _module
benchmarks = _module
benchmark_attn_decoding = _module
benchmark_core = _module
benchmark_indexing = _module
benchmark_mem_eff_attention = _module
benchmark_merge_attentions = _module
benchmark_multi_head_dispatch = _module
benchmark_nystrom_utils = _module
benchmark_revnet = _module
benchmark_sddmm = _module
benchmark_sequence_parallel_fused = _module
benchmark_sp24 = _module
benchmark_swiglu = _module
benchmark_tiled_matmul = _module
utils = _module
checkpoint = _module
components = _module
activations = _module
attention = _module
_sputnik_sparse = _module
attention_mask = _module
attention_patterns = _module
base = _module
compositional = _module
core = _module
favor = _module
feature_maps = _module
softmax = _module
fourier_mix = _module
global_tokens = _module
lambda_layer = _module
linformer = _module
local = _module
nystrom = _module
ortho = _module
pooling = _module
random = _module
scaled_dot_product = _module
sparsity_config = _module
visual = _module
feedforward = _module
conv_mlp = _module
mixture_of_experts = _module
mlp = _module
input_projection = _module
multi_head_dispatch = _module
patch_embedding = _module
positional_embedding = _module
param = _module
rotary = _module
sine = _module
vocab = _module
residual = _module
reversible = _module
simplicial_embedding = _module
generate_kernels = _module
generate_instances = _module
factory = _module
block_configs = _module
block_factory = _module
hydra_helper = _module
model_factory = _module
weight_init = _module
helpers = _module
hierarchical_configs = _module
test_utils = _module
timm_sparse_attention = _module
info = _module
ops = _module
_triton = _module
k_index_select_cat = _module
k_scaled_index_add = _module
rmsnorm_kernels = _module
rope_padded_kernels = _module
tiled_matmul_kernels = _module
common = _module
differentiable_collectives = _module
fmha = _module
splitk_kernels = _module
attn_bias = _module
ck = _module
ck_decoder = _module
ck_splitk = _module
common = _module
cutlass = _module
dispatch = _module
flash = _module
flash3 = _module
torch_attention_compat = _module
triton_splitk = _module
indexing = _module
ipc = _module
modpar_layers = _module
rmsnorm = _module
rope_padded = _module
seqpar = _module
sequence_parallel_fused_ops = _module
sp24 = _module
swiglu_op = _module
tiled_matmul = _module
unbind = _module
profiler = _module
api = _module
device_limits = _module
find_slowest = _module
profile_analyzer = _module
profiler_dcgm = _module
profiler_dcgm_impl = _module
sparse = _module
_csr_ops = _module
blocksparse_tensor = _module
csr_tensor = _module
test = _module
triton = _module
vararg_kernel = _module
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


import functools


import torch


import logging


import math


import random


from functools import partial


from typing import Any


from typing import List


from typing import Optional


from typing import Sequence


from typing import Tuple


from typing import Type


from typing import TypeVar


from typing import Union


import torch.nn.functional as F


from scipy.stats import binomtest


from torch.utils.checkpoint import checkpoint


from typing import Dict


import torch.nn as nn


import triton


import itertools


from torch.utils import benchmark


from triton.ops.matmul import matmul as triton_matmul


import copy


from collections import defaultdict


from collections import namedtuple


from typing import Generator


from typing import Iterator


from typing import Set


import matplotlib.pyplot as plt


import numpy as np


import pandas as pd


from typing import TYPE_CHECKING


import triton.language as tl


from triton.ops.matmul_perf_model import early_config_prune


from triton.ops.matmul_perf_model import estimate_matmul_time


from typing import Callable


from typing import Iterable


from typing import Mapping


from typing import cast


@triton.jit
def index_select_cat_fwd_kernel(output_ptr, source_ptr, index_ptr, num_indices, num_cols, stride0, stride1, BLOCK_SIZE_INDEX: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    rows = tl.load(index_ptr + indices, mask=indices < num_indices)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)
    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)


@triton.jit
def index_select_cat_bwd_kernel(grad_source_ptr, index_ptr, grad_output_ptr, num_rows, num_indices, num_cols, stride0, stride1, BLOCK_SIZE_INDEX: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    grad_output_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    grad_output_offsets = grad_output_ptr + grad_output_indices[:, None] * stride0 + cols[None, :] * stride1
    grad_output_mask = (grad_output_indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    grad_output = tl.load(grad_output_offsets, mask=grad_output_mask)
    grad_source_indices = tl.load(index_ptr + grad_output_indices, mask=grad_output_indices < num_indices)
    grad_source_offsets = grad_source_ptr + grad_source_indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(grad_source_offsets, grad_output, mask=grad_output_mask)


@triton.jit
def scaled_index_add_fwd_kernel(input_ptr, index_ptr, source_ptr, scaling_ptr, alpha, num_inp_indices, num_src_indices, num_rows, num_cols, stride0, stride1, stride2, BLOCK_SIZE_INDEX: 'tl.constexpr', BLOCK_SIZE_ROW: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr', HAS_SCALING: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)
    rows = pid1 * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    cols = pid2 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    source_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    source_offsets = source_ptr + source_indices[:, None, None] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :] * stride2
    source_mask = (source_indices[:, None, None] < num_src_indices) & (rows[None, :, None] < num_rows) & (cols[None, None, :] < num_cols)
    source = tl.load(source_offsets, mask=source_mask)
    input_indices = tl.load(index_ptr + source_indices, mask=source_indices < num_src_indices)
    input_offsets = input_ptr + input_indices[:, None, None] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :] * stride2
    x = tl.load(input_offsets, mask=source_mask)
    if HAS_SCALING:
        scaling = tl.load(scaling_ptr + cols[None, None, :] * stride2, mask=cols[None, None, :] < num_cols)
        tl.store(input_offsets, x + alpha * scaling * source, mask=source_mask)
    else:
        tl.store(input_offsets, x + alpha * source, mask=source_mask)


@triton.jit
def scaled_index_add_bwd_kernel(grad_output_ptr, grad_source_ptr, grad_scaling_ptr, source_ptr, scaling_ptr, index_ptr, alpha, num_inp_indices, num_src_indices, num_rows, num_cols, stride0, stride1, stride2, BLOCK_SIZE_INDEX: 'tl.constexpr', BLOCK_SIZE_ROW: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr', HAS_SCALING: 'tl.constexpr'):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)
    rows = pid1 * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    cols = pid2 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    source_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    source_offsets = source_ptr + source_indices[:, None, None] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :] * stride2
    source_mask = (source_indices[:, None, None] < num_src_indices) & (rows[None, :, None] < num_rows) & (cols[None, None, :] < num_cols)
    source = tl.load(source_offsets, mask=source_mask)
    grad_output_indices = tl.load(index_ptr + source_indices, mask=source_indices < num_src_indices)
    grad_output_offsets = grad_output_ptr + grad_output_indices * stride0 + rows[None, :, None] * stride1 + cols[None, None, :] * stride2
    grad_output = tl.load(grad_output_offsets, mask=source_mask)
    grad_source_offsets = grad_source_ptr + source_indices[:, None, None] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :] * stride2
    if HAS_SCALING:
        scaling = tl.load(scaling_ptr + cols[None, None, :] * stride2, mask=cols[None, None, :] < num_cols)
        tl.store(grad_source_offsets, alpha * grad_output * scaling, mask=source_mask)
        grad_scaling_offsets = grad_scaling_ptr + source_indices[:, None, None] * stride0 + rows[None, :, None] * stride1 + cols[None, None, :] * stride2
        tl.store(grad_scaling_offsets, alpha * grad_output * source, mask=source_mask)
    else:
        tl.store(grad_source_offsets, alpha * grad_output, mask=source_mask)


@triton.jit
def _rms_norm_kernel(x_ptr, h1_ptr, w_ptr, eps, stride, N_COLS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', INCLUDE_WEIGHT: 'tl.constexpr'):
    row = tl.program_id(0)
    x_ptr += row * stride
    h1_ptr += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(x_ptr + cols, mask=cols < N_COLS, other=0.0, eviction_policy='evict_last')
        _mean += a * a
    rstd = rsqrt(tl.sum(_mean, axis=0) / N_COLS + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(x_ptr + cols, mask=mask, other=0.0, eviction_policy='evict_first')
        if INCLUDE_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(h1_ptr + cols, a * rstd * w, mask=mask)
        else:
            tl.store(h1_ptr + cols, a * rstd, mask=mask)


@triton.jit
def _rms_norm_add_kernel(x_ptr, y_ptr, h1_ptr, w_ptr, eps, stride, N_COLS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', INCLUDE_WEIGHT: 'tl.constexpr'):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    h1_ptr += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        ax = tl.load(x_ptr + cols, mask=mask, other=0.0, eviction_policy='evict_last')
        ay = tl.load(y_ptr + cols, mask=mask, other=0.0, eviction_policy='evict_first')
        a = ax + ay
        tl.store(x_ptr + cols, a, mask=mask)
        _mean += a * a
    rstd = rsqrt(tl.sum(_mean, axis=0) / N_COLS + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(x_ptr + cols, mask=mask, other=0.0, eviction_policy='evict_first')
        if INCLUDE_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(h1_ptr + cols, a * rstd * w, mask=mask)
        else:
            tl.store(h1_ptr + cols, a * rstd, mask=mask)


@triton.jit
def _rope_padded_kernel(xq, xk, xv, out_q, cache_k, cache_v, seqstartq, seqstartk, seqlenk, theta, linear_scale, use_dynamic_scaling: 'tl.constexpr', dynamic_old_context_len: 'tl.constexpr', dynamic_scale_factor: 'tl.constexpr', dynamic_low_freq_factor: 'tl.constexpr', dynamic_high_freq_factor: 'tl.constexpr', first_seqpos, seqpos, k_start: 'tl.constexpr', v_start: 'tl.constexpr', n_groups, dim: 'tl.constexpr', stride_xqM, stride_xqG, stride_xqH, stride_xkM, stride_xkG, stride_xkH, stride_xvM, stride_xvG, stride_xvH, stride_cachekM, stride_cachekG, stride_cachekH, stride_cachevM, stride_cachevG, stride_cachevH, stride_seqstartq, stride_seqstartk, stride_seqlenk, stride_outqM, stride_outqG, stride_outqH, stride_seqpos, internal_dtype: 'tl.constexpr', const_batch_strides: 'tl.constexpr', cache_padding_length, seqlenk_shift: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', adjacents: 'tl.constexpr'):
    """
    Each letter in this diagram is a whole row of length dim.

     INPUT      xq        xk       xv

        head_dim ─►

      batch   qqqqqq      kk       vv
        │     qqqqqq      kk       vv
        ▼     qqqqqq      kk       vv

    head_idx:  (goes across all heads of all 3 inputs)
              ▲     ▲     ▲ ▲      ▲ ▲
              │     │     │ │      │ │
                          │        │
              0  k_start  │v_start │n_total_heads
                          │        │
                          │        │
                      k_start    v_start

    Output is to out_q (same shape as xq), an xk-shaped part
    of cache_k and an xv-shaped part of cache_v
    """
    query_pos_in_batch_elt = tl.program_id(0)
    batch_elt = tl.program_id(1)
    group_head_idx = tl.program_id(2)
    group_idx = group_head_idx % n_groups
    head_idx = group_head_idx // n_groups
    if internal_dtype == 'f32':
        theta = theta
    elif internal_dtype == 'f64':
        theta = theta
    if const_batch_strides:
        query_pos = query_pos_in_batch_elt + tl.num_programs(1) * batch_elt
        end_query_pos = tl.num_programs(1) * (batch_elt + 1)
    else:
        query_pos = query_pos_in_batch_elt + tl.load(seqstartq + batch_elt * stride_seqstartq)
        end_query_pos = tl.load(seqstartq + (batch_elt + 1) * stride_seqstartq)
        if query_pos >= end_query_pos:
            return
    is_q = head_idx < k_start
    is_v = head_idx >= v_start
    xq += query_pos * stride_xqM + head_idx * stride_xqH + group_idx * stride_xqG
    out_q += query_pos * stride_outqM + head_idx * stride_outqH + group_idx * stride_outqG
    if const_batch_strides:
        cache_start = cache_padding_length * batch_elt
    else:
        cache_start = tl.load(seqstartk + batch_elt * stride_seqstartk)
    end_of_batch_elt_cache = cache_start + tl.load(seqlenk + batch_elt * stride_seqlenk) + seqlenk_shift
    cache_pos = end_of_batch_elt_cache - (end_query_pos - query_pos)
    if seqpos is not None:
        seq_pos = tl.load(seqpos + query_pos * stride_seqpos)
    else:
        seq_pos = cache_pos - cache_start
        if first_seqpos is not None:
            seq_pos += tl.load(first_seqpos + batch_elt * stride_seqpos)
    cache_k += (head_idx - k_start) * stride_cachekH + cache_pos * stride_cachekM + group_idx * stride_cachekG
    xk += query_pos * stride_xkM + (head_idx - k_start) * stride_xkH + group_idx * stride_xkG
    in_qk = tl.where(is_q, xq, xk)
    out_qk = tl.where(is_q, out_q, cache_k)
    cache_v += (head_idx - v_start) * stride_cachevH + cache_pos * stride_cachevM + group_idx * stride_cachevG
    xv += query_pos * stride_xvM + (head_idx - v_start) * stride_xvH + group_idx * stride_xvG
    out = tl.where(is_v, cache_v, out_qk)
    x_in = tl.where(is_v, xv, in_qk)
    for offset in range(0, dim // 2, BLOCK_SIZE // 2):
        c = tl.arange(0, BLOCK_SIZE // 2)
        powers = (offset + c) * 2.0
        if adjacents:
            cols_re = (offset + c) * 2
            cols_im = cols_re + 1
        else:
            cols_re = offset + c
            cols_im = cols_re + dim // 2
        mask = cols_im < dim
        re_x = tl.load(x_in + cols_re, mask=mask)
        im_x = tl.load(x_in + cols_im, mask=mask)
        freqs = pow(theta, powers / -dim)
        if use_dynamic_scaling:
            lo_freq_wavelen = dynamic_old_context_len / dynamic_low_freq_factor
            hi_freq_wavelen = dynamic_old_context_len / dynamic_high_freq_factor
            wavelens = 6.28318530718 / freqs
            is_low_freq = wavelens > lo_freq_wavelen
            freqs = tl.where(is_low_freq, freqs / dynamic_scale_factor, freqs)
            is_mid_freq = hi_freq_wavelen <= wavelens and wavelens <= lo_freq_wavelen
            smooth = (dynamic_old_context_len / wavelens - dynamic_low_freq_factor) / (dynamic_high_freq_factor - dynamic_low_freq_factor)
            freqs = tl.where(is_mid_freq, (1 - smooth) * freqs / dynamic_scale_factor + smooth * freqs, freqs)
        freqs = seq_pos * freqs / linear_scale
        sines = tl.sin(freqs)
        cosines = tl.cos(freqs)
        re_out = re_x * cosines - im_x * sines
        im_out = im_x * cosines + re_x * sines
        re_out_ = tl.where(is_v, re_x, re_out)
        im_out_ = tl.where(is_v, im_x, im_out)
        if internal_dtype == 'f64':
            if re_x.dtype == tl.bfloat16:
                re_out_ = re_out_
                im_out_ = im_out_
        tl.store(out + cols_re, re_out_, mask=mask)
        tl.store(out + cols_im, im_out_, mask=mask)


@triton.jit
def cast_uint32_to_half2(scale_shift):
    """Extract two float16 packed into one int32"""
    scale = scale_shift & 65535
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16)
    shift = shift.to(tl.uint16)
    return scale, shift


@triton.jit
def dequantize(x_, scale, shift, PACKED_PER_VAL: 'tl.constexpr'):
    """PACKED_PER_VAL is the number of values packed into each element x_.
    For example, for int4 quantization and x_ of type int32, PACKED_PER_VAL is 8.
    """
    BLOCK_N: 'tl.constexpr' = x_.shape[0]
    BLOCK_DMODEL_PACKED: 'tl.constexpr' = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * (32 // PACKED_PER_VAL)
    quant_offset = x_[:, :, None, :] >> offsets
    quant_offset = tl.reshape(quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL))
    if PACKED_PER_VAL == 4:
        fp8_type = tl.float8e4b8 if torch.version.hip is not None else tl.float8e4nv
        dequant = quant_offset.to(tl.uint8).to(fp8_type, bitcast=True) * scale + shift
    else:
        quant_offset = (quant_offset & 15).to(tl.uint16)
        quant_offset = quant_offset * 32768.0
        scale_512 = scale * 512
        dequant = quant_offset * scale_512 + shift
    return dequant


@triton.jit
def load_dequantize_k_v_group(K_block_ptr, V_block_ptr, K_scale_shift_block_ptr, V_scale_shift_block_ptr, BOUNDS_CHECKS_N: 'tl.constexpr', PACKED_PER_VAL: 'tl.constexpr', PACKED_D_PER_GROUP: 'tl.constexpr', FP8_QUANTIZED: 'tl.constexpr', dtype: 'tl.constexpr', group_id: 'tl.constexpr'):
    """Load K/V for a given block. In case of int4/fp8-quantized K/V, dequantize them after loading.
    If quantization is group-wise, use group_id to advance the pointers to the current group.
    """
    K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))
    k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())
    if FP8_QUANTIZED:
        v_scale_shift = tl.load(V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL)
        k_scale_shift = tl.load(K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        k_t = dequantize(tl.trans(k), tl.trans(k_scale), tl.trans(k_shift), PACKED_PER_VAL)
        k = tl.trans(k_t)
    elif PACKED_PER_VAL > 1:
        K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
        V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))
        k_scale_shift = tl.load(K_scale_shift_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
        v_scale_shift = tl.load(V_scale_shift_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())
        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL)
        k_t = dequantize(tl.trans(k), tl.trans(k_scale), tl.trans(k_shift), PACKED_PER_VAL)
        k = tl.trans(k_t)
    return k, v


@triton.jit
def _fwd_kernel_splitK(Q, K, V, sm_scale, Out_splitK, LSE_splitk, block_tables, Seq_len, Seq_starts_k, Seq_starts_q, Seq_starts_q_multiplier, additive_bias, K_fp8_scale_shift, V_fp8_scale_shift, stride_qz, stride_qm, stride_qg, stride_qh, stride_qk, stride_kz, stride_kn, stride_kg, stride_kh, stride_kk, stride_vz, stride_vn, stride_vg, stride_vh, stride_vk, stride_osk_z, stride_osk_g, stride_osk_h, stride_osk_s, stride_osk_m, stride_osk_k, stride_lsek_z, stride_lsek_g, stride_lsek_h, stride_lsek_s, stride_lsek_m, stride_blocktablesz, stride_blocktablesl, stride_bias_b, stride_bias_g, stride_bias_h, stride_bias_qm, stride_bias_km, stride_k_fp8_scale_shift_z: 'tl.constexpr', stride_k_fp8_scale_shift_n: 'tl.constexpr', stride_k_fp8_scale_shift_g: 'tl.constexpr', stride_k_fp8_scale_shift_h: 'tl.constexpr', stride_v_fp8_scale_shift_z: 'tl.constexpr', stride_v_fp8_scale_shift_n: 'tl.constexpr', stride_v_fp8_scale_shift_g: 'tl.constexpr', stride_v_fp8_scale_shift_h: 'tl.constexpr', kv_cache_blocks_per_row: 'tl.constexpr', Z: 'tl.constexpr', N_CTX_Q: 'tl.constexpr', N_CTX_K: 'tl.constexpr', BLOCK_N_PER_SPLIT: 'tl.constexpr', H: 'tl.constexpr', G: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', USE_SEQ_LEN: 'tl.constexpr', PACKED_PER_VAL: 'tl.constexpr', N_GROUPS: 'tl.constexpr', BOUNDS_CHECKS_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_SPLITK: 'tl.constexpr', SPLIT_K_EARLY_EXIT: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', NUM_QUERIES_CAUSAL: 'tl.constexpr', USE_PAGED_ATTENTION: 'tl.constexpr', PAGE_SIZE: 'tl.constexpr', WRITE_LSE: 'tl.constexpr', HAS_ADDITIVE_BIAS: 'tl.constexpr'):
    """This kernel can accept non-quantized or int4-quantized keys/values.
    PACKED_PER_VAL determines the quantization type:
        - PACKED_PER_VAL == 1 means no quantization
        - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
    For the quantized case K/V should be int32 tensors.
    Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
    Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
    So K[B, H, M, :] has a form
    [   quant_coef0, quant_coef1, ...|
        group0_quant_value0, group0_quant_value1,... |
        group1_quant_value0, group1_quant_value1,...]
    where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

    Note: this kernel needs to be processed by xformers.triton.vararg_kernel.unroll_varargs
    before compilation. That will unroll variables marked with "VAR_ARGS_ARRAY" into lists.
    See how FwOp.apply does it below.

    Set IS_SPLITK=False to indicate the MHA result should be written directly.
    No metadata will be written.
    """
    internal_dtype = tl.float64 if Out_splitK.dtype.element_ty is tl.float64 else tl.float32
    tl.static_assert(PACKED_PER_VAL == 1 and tl.constexpr(K.dtype.element_ty != tl.int32) or (PACKED_PER_VAL == 4 or PACKED_PER_VAL == 8) and tl.constexpr(K.dtype.element_ty == tl.int32), f'Only int4 and fp8 quantization is supported, K/V should have dtype int32 in the quantized case: PACKED_PER_VAL={PACKED_PER_VAL!r} tl.constexpr(K.dtype)={tl.constexpr(K.dtype)!r} tl.constexpr(K.dtype.element_ty)={tl.constexpr(K.dtype.element_ty)!r}')
    tl.static_assert(((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8, 'Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.')
    tl.static_assert(N_GROUPS == 1 or K_fp8_scale_shift is None, f'Only row-wise fp8 quantization is supported, but got N_GROUPS={N_GROUPS!r} > 1.')
    FP8_QUANTIZED: 'tl.constexpr' = K_fp8_scale_shift is not None
    INT4_QUANTIZED: 'tl.constexpr' = PACKED_PER_VAL > 1 and not FP8_QUANTIZED
    PACKED_D_PER_GROUP: 'tl.constexpr' = BLOCK_DMODEL // PACKED_PER_VAL // N_GROUPS
    D_PER_GROUP: 'tl.constexpr' = BLOCK_DMODEL // N_GROUPS
    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_hg = off_zhg % (H * G)
    off_h = off_hg // G
    off_g = off_hg % G
    splitk_idx = tl.program_id(2)
    if USE_SEQ_LEN:
        kv_len = tl.load(Seq_len + off_z)
        if SPLIT_K_EARLY_EXIT and kv_len == 0:
            return
    else:
        kv_len = N_CTX_K
    if Seq_starts_k is None:
        start_kv_idx = 0
    else:
        start_kv_idx = tl.load(Seq_starts_k + off_z)
    if Seq_starts_q is None:
        q_len = N_CTX_Q
        queries_use_batch_dim = 1
        off_m = 0
    else:
        queries_use_batch_dim = 0
        off_m = tl.load(Seq_starts_q + off_z) * Seq_starts_q_multiplier
        q_len = tl.load(Seq_starts_q + off_z + 1) * Seq_starts_q_multiplier - off_m
        if q_len == 0:
            return
    k_base = K + off_h * stride_kh + off_g * stride_kg
    v_base = V + off_h * stride_vh + off_g * stride_vg
    if FP8_QUANTIZED:
        k_fp8_scale_shift_base = K_fp8_scale_shift + off_h * stride_k_fp8_scale_shift_h + off_g * stride_k_fp8_scale_shift_g
        v_fp8_scale_shift_base = V_fp8_scale_shift + off_h * stride_v_fp8_scale_shift_h + off_g * stride_v_fp8_scale_shift_g
    else:
        k_fp8_scale_shift_base = None
        v_fp8_scale_shift_base = None
    chunk_hi = (splitk_idx + 1) * BLOCK_N_PER_SPLIT
    chunk_lo = splitk_idx * BLOCK_N_PER_SPLIT
    ignore_in_first_block = 0
    if PAGE_SIZE > 0:
        BLOCKS_IN_PAGE: 'tl.constexpr' = PAGE_SIZE // BLOCK_N
        is_last_chunk = splitk_idx == tl.num_programs(2) - 1
        shift = BLOCK_N - 1 if is_last_chunk else 0
        lo = tl.maximum(chunk_lo, start_kv_idx) // BLOCK_N * BLOCK_N
        ignore_in_first_block = tl.maximum(0, start_kv_idx - lo)
        hi = (chunk_hi + shift) // BLOCK_N * BLOCK_N
        hi = tl.minimum(hi, kv_len + start_kv_idx)
        block_table = block_tables + stride_blocktablesz * off_z
        logical_block_idx = lo // BLOCK_N
    else:
        lo = chunk_lo
        hi = tl.minimum(chunk_hi, kv_len)
        if Seq_starts_k is not None:
            k_base += start_kv_idx * stride_kn
            v_base += start_kv_idx * stride_vn
        else:
            k_base += off_z * stride_kz
            v_base += off_z * stride_vz
        K_block_ptr = tl.make_block_ptr(base=k_base + stride_kk * INT4_QUANTIZED * N_GROUPS, shape=(PACKED_D_PER_GROUP, hi), strides=(stride_kk, stride_kn), offsets=(0, lo), block_shape=(PACKED_D_PER_GROUP, BLOCK_N), order=(0, 1))
        V_block_ptr = tl.make_block_ptr(base=v_base + stride_vk * INT4_QUANTIZED * N_GROUPS, shape=(hi, PACKED_D_PER_GROUP), strides=(stride_vn, stride_vk), offsets=(lo, 0), block_shape=(BLOCK_N, PACKED_D_PER_GROUP), order=(1, 0))
        if INT4_QUANTIZED:
            K_scale_shift_block_ptr = tl.make_block_ptr(base=k_base, shape=(1, hi), strides=(stride_kk, stride_kn), offsets=(0, lo), block_shape=(1, BLOCK_N), order=(0, 1))
            V_scale_shift_block_ptr = tl.make_block_ptr(base=v_base, shape=(hi, 1), strides=(stride_vn, stride_vk), offsets=(lo, 0), block_shape=(BLOCK_N, 1), order=(1, 0))
        elif FP8_QUANTIZED:
            if Seq_starts_k is not None:
                k_fp8_scale_shift_base += start_kv_idx * stride_k_fp8_scale_shift_n
                v_fp8_scale_shift_base += start_kv_idx * stride_v_fp8_scale_shift_n
            else:
                k_fp8_scale_shift_base += off_z * stride_k_fp8_scale_shift_z
                v_fp8_scale_shift_base += off_z * stride_v_fp8_scale_shift_z
            K_scale_shift_block_ptr = tl.make_block_ptr(base=k_fp8_scale_shift_base, shape=(1, hi), strides=(1, stride_k_fp8_scale_shift_n), offsets=(0, lo), block_shape=(1, BLOCK_N), order=(0, 1))
            V_scale_shift_block_ptr = tl.make_block_ptr(base=v_fp8_scale_shift_base, shape=(hi, 1), strides=(stride_v_fp8_scale_shift_n, 1), offsets=(lo, 0), block_shape=(BLOCK_N, 1), order=(1, 0))
        else:
            K_scale_shift_block_ptr = None
            V_scale_shift_block_ptr = None
        if HAS_ADDITIVE_BIAS:
            additive_bias_block_ptr = tl.make_block_ptr(base=additive_bias + off_z * stride_bias_b + off_g * stride_bias_g + off_h * stride_bias_h, shape=(N_CTX_Q, hi), strides=(stride_bias_qm, stride_bias_km), offsets=(start_m * BLOCK_M, lo), block_shape=(BLOCK_M, BLOCK_N), order=(0, 1))
    if SPLIT_K_EARLY_EXIT and lo >= hi:
        return
    Q_block_ptr = tl.make_block_ptr(base=Q + off_m * stride_qm + off_h * stride_qh + off_z * stride_qz * queries_use_batch_dim + off_g * stride_qg, shape=(q_len, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, D_PER_GROUP), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc: "'VAR_ARGS_ARRAY'"
    for i in range(len(acc)):
        acc[i] = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=internal_dtype)
    qk_scale = sm_scale * 1.44269504
    q: "'VAR_ARGS_ARRAY'"
    for i in range(len(acc)):
        q[i] = tl.load(tl.advance(Q_block_ptr, (0, i * D_PER_GROUP)), boundary_check=(0,))
    if IS_CAUSAL:
        q_offset = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        diag_idx = q_offset[:, None] % NUM_QUERIES_CAUSAL - tl.arange(0, BLOCK_N)[None, :]
        diag_idx_shifted = tl.constexpr(diag_idx - NUM_QUERIES_CAUSAL + kv_len)
    for start_n in range(lo, hi, BLOCK_N):
        if PAGE_SIZE > 0:
            block_offset_in_page = logical_block_idx % BLOCKS_IN_PAGE
            logical_page_idx = logical_block_idx // BLOCKS_IN_PAGE
            physical_page_idx = tl.load(block_table + stride_blocktablesl * logical_page_idx)
            offset = physical_page_idx * PAGE_SIZE + block_offset_in_page * BLOCK_N
            current_block_size = min(hi - start_n, BLOCK_N)
            K_block_ptr = tl.make_block_ptr(base=k_base + stride_kk * INT4_QUANTIZED * N_GROUPS, shape=(PACKED_D_PER_GROUP, offset + current_block_size), strides=(stride_kk, stride_kn), offsets=(0, offset), block_shape=(PACKED_D_PER_GROUP, BLOCK_N), order=(0, 1))
            V_block_ptr = tl.make_block_ptr(base=v_base + stride_vk * INT4_QUANTIZED * N_GROUPS, shape=(offset + current_block_size, PACKED_D_PER_GROUP), strides=(stride_vn, stride_vk), offsets=(offset, 0), block_shape=(BLOCK_N, PACKED_D_PER_GROUP), order=(1, 0))
            if INT4_QUANTIZED:
                K_scale_shift_block_ptr = tl.make_block_ptr(base=k_base, shape=(1, offset + current_block_size), strides=(stride_kk, stride_kn), offsets=(0, offset), block_shape=(1, BLOCK_N), order=(0, 1))
                V_scale_shift_block_ptr = tl.make_block_ptr(base=v_base, shape=(offset + current_block_size, 1), strides=(stride_vn, stride_vk), offsets=(offset, 0), block_shape=(BLOCK_N, 1), order=(1, 0))
            elif FP8_QUANTIZED:
                K_scale_shift_block_ptr = tl.make_block_ptr(base=k_fp8_scale_shift_base, shape=(1, offset + current_block_size), strides=(1, stride_k_fp8_scale_shift_n), offsets=(0, offset), block_shape=(1, BLOCK_N), order=(0, 1))
                V_scale_shift_block_ptr = tl.make_block_ptr(base=v_fp8_scale_shift_base, shape=(offset + current_block_size, 1), strides=(stride_v_fp8_scale_shift_n, 1), offsets=(offset, 0), block_shape=(BLOCK_N, 1), order=(1, 0))
            else:
                K_scale_shift_block_ptr = None
                V_scale_shift_block_ptr = None
            logical_block_idx += 1
        k: "'VAR_ARGS_ARRAY'"
        v: "'VAR_ARGS_ARRAY'"
        for i in range(len(acc)):
            k[i], v[i] = load_dequantize_k_v_group(K_block_ptr, V_block_ptr, K_scale_shift_block_ptr, V_scale_shift_block_ptr, BOUNDS_CHECKS_N, PACKED_PER_VAL, PACKED_D_PER_GROUP, FP8_QUANTIZED, Q.dtype.element_ty, i)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for i in range(len(acc)):
            qk += tl.dot(q[i], k[i])
        qk *= qk_scale
        if start_n == lo and ignore_in_first_block > 0:
            qk = tl.where(tl.arange(0, BLOCK_N) < ignore_in_first_block, float('-inf'), qk)
        if HAS_ADDITIVE_BIAS:
            loaded_bias = tl.load(additive_bias_block_ptr, boundary_check=(0, 1) if BOUNDS_CHECKS_N else (0,))
            qk += loaded_bias * 1.44269504
            additive_bias_block_ptr = tl.advance(additive_bias_block_ptr, (0, BLOCK_N))
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float('-inf'))
        if IS_CAUSAL:
            qk = tl.where(diag_idx_shifted >= start_n, qk, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if HAS_ADDITIVE_BIAS or IS_CAUSAL:
            alpha = tl.where(m_i_new == float('-inf'), 0, alpha)
            p = tl.where(m_i_new[:, None] == float('-inf'), 0, p)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p
        for i in range(len(acc)):
            acc[i] *= alpha[:, None]
            acc[i] += tl.dot(p, v[i])
        if not PAGE_SIZE:
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            if PACKED_PER_VAL > 1:
                K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (0, BLOCK_N))
                V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (BLOCK_N, 0))
    O_block_ptr = tl.make_block_ptr(base=Out_splitK + off_z * stride_osk_z * queries_use_batch_dim + off_m * stride_osk_m + off_g * stride_osk_g + off_h * stride_osk_h + splitk_idx * stride_osk_s, shape=(q_len, D_PER_GROUP), strides=(stride_osk_m, 1), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, D_PER_GROUP), order=(1, 0))
    for i in range(len(acc)):
        attn_out = tl.where(l_i[:, None] == 0, 0.0, acc[i] / l_i[:, None])
        tl.store(tl.advance(O_block_ptr, (0, i * D_PER_GROUP)), attn_out, boundary_check=(0,))
    if WRITE_LSE:
        LSE_splitk_ptr = LSE_splitk + off_z * stride_lsek_z * queries_use_batch_dim + off_m * stride_lsek_m + off_g * stride_lsek_g + off_h * stride_lsek_h + splitk_idx * stride_lsek_s + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_lsek_m
        mask = start_m * BLOCK_M + tl.arange(0, BLOCK_M) < q_len
        lse_dtype = LSE_splitk.dtype.element_ty
        tl.store(LSE_splitk_ptr, (tl.math.log2(l_i) + m_i) / 1.44269504, mask=mask)


@triton.jit
def _splitK_reduce(Out_splitK, LSE_splitK, Out, LSE, split_k: 'tl.constexpr', splitK_pow2: 'tl.constexpr', stride_osk_z: 'tl.constexpr', stride_osk_g: 'tl.constexpr', stride_osk_h: 'tl.constexpr', stride_osk_s: 'tl.constexpr', stride_osk_m: 'tl.constexpr', stride_osk_k: 'tl.constexpr', stride_lsek_z: 'tl.constexpr', stride_lsek_g: 'tl.constexpr', stride_lsek_h: 'tl.constexpr', stride_lsek_s: 'tl.constexpr', stride_lsek_m: 'tl.constexpr', stride_oz: 'tl.constexpr', stride_og: 'tl.constexpr', stride_oh: 'tl.constexpr', stride_om: 'tl.constexpr', stride_ok: 'tl.constexpr', stride_lse_z: 'tl.constexpr', stride_lse_g: 'tl.constexpr', stride_lse_h: 'tl.constexpr', stride_lse_m: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', H: 'tl.constexpr', G: 'tl.constexpr', WRITE_LSE: 'tl.constexpr'):
    off_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = off_zhg // G % H
    off_g = off_zhg % G
    Out_splitK_ptr = Out_splitK + stride_osk_z * off_z + stride_osk_g * off_g + stride_osk_h * off_h + stride_osk_m * off_m + tl.arange(0, BLOCK_SIZE)[None, :] + stride_osk_s * tl.arange(0, splitK_pow2)[:, None]
    LSE_splitK_ptr0 = LSE_splitK + stride_lsek_z * off_z + stride_lsek_g * off_g + stride_lsek_h * off_h + stride_lsek_m * off_m + stride_lsek_s * tl.arange(0, splitK_pow2)
    if splitK_pow2 > split_k:
        mask_1d = tl.arange(0, splitK_pow2) < split_k
        mask_2d = mask_1d[:, None]
        lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float('-inf'))
        lse_max = tl.max(lse_splitk)
        out_splitk = tl.load(Out_splitK_ptr, mask=mask_2d, other=0)
        lse_splitk = tl.load(LSE_splitK_ptr0, mask=mask_1d, other=float('-inf'))
    else:
        lse_splitk = tl.load(LSE_splitK_ptr0)
        lse_max = tl.max(lse_splitk)
        out_splitk = tl.load(Out_splitK_ptr)
        lse_splitk = tl.load(LSE_splitK_ptr0)
    sumexp_normalized_splitk = tl.math.exp2((lse_splitk - lse_max) * 1.44269504)
    sumexp_normalized = tl.sum(sumexp_normalized_splitk, axis=0)
    numerator_normalized = tl.sum(out_splitk * sumexp_normalized_splitk[:, None], axis=0)
    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float('-inf'), 0.0, acc)
    Out_ptr = Out + stride_oz * off_z + stride_oh * off_h + stride_og * off_g + stride_om * off_m + tl.arange(0, BLOCK_SIZE)
    if acc.dtype is tl.float64 and Out.dtype.element_ty is not tl.float64:
        acc = acc
    tl.store(Out_ptr, acc)
    if WRITE_LSE:
        l_ptrs = LSE + off_z * stride_lse_z + off_g * stride_lse_g + off_h * stride_lse_h + off_m * stride_lse_m
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        to_store = tl.where(lse_max == float('-inf'), lse_max, to_store)
        tl.store(l_ptrs, to_store)


@triton.jit
def _splitK_reduce_varargs(Out_splitK: "'VAR_ARGS_ARRAY'", LSE_splitK: "'VAR_ARGS_ARRAY'", Out, LSE, stride_osk_z: "'VAR_ARGS_ARRAY'", stride_osk_g: "'VAR_ARGS_ARRAY'", stride_osk_h: "'VAR_ARGS_ARRAY'", stride_osk_m: "'VAR_ARGS_ARRAY'", stride_osk_k: "'VAR_ARGS_ARRAY'", stride_lsek_z: "'VAR_ARGS_ARRAY'", stride_lsek_g: "'VAR_ARGS_ARRAY'", stride_lsek_h: "'VAR_ARGS_ARRAY'", stride_lsek_m: "'VAR_ARGS_ARRAY'", stride_oz, stride_og, stride_oh, stride_om, stride_ok, stride_lse_z, stride_lse_g, stride_lse_h, stride_lse_m, BLOCK_SIZE: 'tl.constexpr', H: 'tl.constexpr', G: 'tl.constexpr', WRITE_LSE: 'tl.constexpr'):
    """
    This version of reduce kernel takes attention and LSE of chunks as lists of tensors,
    as opposed to _splitK_reduce, which takes each as a stacked tensor.
    """
    off_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = off_zhg // G % H
    off_g = off_zhg % G
    out_splitk_offset: "'VAR_ARGS_ARRAY'"
    for i in range(len(Out_splitK)):
        out_splitk_offset[i] = stride_osk_z[i] * off_z + stride_osk_g[i] * off_g + stride_osk_h[i] * off_h + stride_osk_m[i] * off_m + tl.arange(0, BLOCK_SIZE)
    lse_splitk_offset: "'VAR_ARGS_ARRAY'"
    for i in range(len(Out_splitK)):
        lse_splitk_offset[i] = stride_lsek_z[i] * off_z + stride_lsek_g[i] * off_g + stride_lsek_h[i] * off_h + stride_lsek_m[i] * off_m
    lse_max = float('-inf')
    for split_k_idx in range(len(Out_splitK)):
        LSE_splitK_ptr = LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]
        lse_splitk = tl.load(LSE_splitK_ptr)
        lse_max = tl.maximum(lse_max, lse_splitk)
    sumexp_normalized = 0.0
    numerator_normalized = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for split_k_idx in range(len(Out_splitK)):
        out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[split_k_idx])
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])
        sumexp_normalized_splitk = tl.math.exp2((lse_splitk - lse_max) * 1.44269504)
        sumexp_normalized += sumexp_normalized_splitk
        numerator_normalized += out_splitk * sumexp_normalized_splitk
    acc = numerator_normalized / sumexp_normalized
    acc = tl.where(lse_max == float('-inf'), 0.0, acc)
    Out_ptr = Out + stride_oz * off_z + stride_oh * off_h + stride_og * off_g + stride_om * off_m + tl.arange(0, BLOCK_SIZE)
    if acc.dtype is tl.float64 and Out.dtype.element_ty is not tl.float64:
        acc = acc
    tl.store(Out_ptr, acc)
    if WRITE_LSE:
        l_ptrs = LSE + off_z * stride_lse_z + off_g * stride_lse_g + off_h * stride_lse_h + off_m * stride_lse_m
        to_store = lse_max + tl.math.log2(sumexp_normalized) / 1.44269504
        to_store = tl.where(lse_max == float('-inf'), lse_max, to_store)
        tl.store(l_ptrs, to_store)


@triton.jit
def _splitK_reduce_varargs_backward(Out_splitK: "'VAR_ARGS_ARRAY'", LSE_splitK: "'VAR_ARGS_ARRAY'", Dout_splitK: "'VAR_ARGS_ARRAY'", DLSE_splitK: "'VAR_ARGS_ARRAY'", Out, LSE, DOut, DLSE, stride_osk_z: "'VAR_ARGS_ARRAY'", stride_osk_g: "'VAR_ARGS_ARRAY'", stride_osk_h: "'VAR_ARGS_ARRAY'", stride_osk_m: "'VAR_ARGS_ARRAY'", stride_osk_k: "'VAR_ARGS_ARRAY'", stride_lsek_z: "'VAR_ARGS_ARRAY'", stride_lsek_g: "'VAR_ARGS_ARRAY'", stride_lsek_h: "'VAR_ARGS_ARRAY'", stride_lsek_m: "'VAR_ARGS_ARRAY'", stride_oz, stride_og, stride_oh, stride_om, stride_ok, stride_lse_z, stride_lse_g, stride_lse_h, stride_lse_m, stride_doz, stride_dog, stride_doh, stride_dom, stride_dok, stride_dlse_z, stride_dlse_g, stride_dlse_h, stride_dlse_m, BLOCK_SIZE: 'tl.constexpr', H: 'tl.constexpr', G: 'tl.constexpr'):
    """
    Backward for _splitK_reduce_varargs. Similar to forward, it takes
    attention and LSE of chunks as lists of tensors,
    and outputs the corresponding gradients in the same format.
    """
    off_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = off_zhg // G % H
    off_g = off_zhg % G
    out_splitk_offset: "'VAR_ARGS_ARRAY'"
    for i in range(len(Out_splitK)):
        out_splitk_offset[i] = stride_osk_z[i] * off_z + stride_osk_g[i] * off_g + stride_osk_h[i] * off_h + stride_osk_m[i] * off_m + tl.arange(0, BLOCK_SIZE)
    lse_splitk_offset: "'VAR_ARGS_ARRAY'"
    for i in range(len(Out_splitK)):
        lse_splitk_offset[i] = stride_lsek_z[i] * off_z + stride_lsek_g[i] * off_g + stride_lsek_h[i] * off_h + stride_lsek_m[i] * off_m
    lse_max = float('-inf')
    for split_k_idx in range(len(Out_splitK)):
        LSE_splitK_ptr = LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]
        lse_splitk = tl.load(LSE_splitK_ptr)
        lse_max = tl.maximum(lse_max, lse_splitk)
    offset_out = stride_oz * off_z + stride_oh * off_h + stride_og * off_g + stride_om * off_m + tl.arange(0, BLOCK_SIZE)
    offset_dout = stride_doz * off_z + stride_doh * off_h + stride_dog * off_g + stride_dom * off_m + tl.arange(0, BLOCK_SIZE)
    out = tl.load(Out + offset_out)
    dattn = tl.load(DOut + offset_dout)
    offset_lse = stride_lse_z * off_z + stride_lse_h * off_h + stride_lse_g * off_g + stride_lse_m * off_m
    offset_dlse = stride_dlse_z * off_z + stride_dlse_h * off_h + stride_dlse_g * off_g + stride_dlse_m * off_m
    lse = tl.load(LSE + offset_lse)
    dlse = tl.load(DLSE + offset_dlse)
    for split_k_idx in range(len(Out_splitK)):
        out_splitk = tl.load(Out_splitK[split_k_idx] + out_splitk_offset[split_k_idx])
        lse_splitk = tl.load(LSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx])
        dout_splitk_ptr = Dout_splitK[split_k_idx] + out_splitk_offset[split_k_idx]
        dlse_splitk_ptr = DLSE_splitK[split_k_idx] + lse_splitk_offset[split_k_idx]
        dattn_dattn_i = tl.exp(lse_splitk - lse_max) / tl.exp(lse - lse_max)
        dX_dattn_i = dattn_dattn_i * dattn
        tl.store(dout_splitk_ptr, dX_dattn_i)
        dattn_dlse_i = (out_splitk - out) * dattn_dattn_i
        dlse_dlse_i = dattn_dattn_i
        dX_dlse_i = dlse_dlse_i * dlse + tl.sum(dattn_dlse_i * dattn)
        tl.store(dlse_splitk_ptr, dX_dlse_i)

