import sys
_module = sys.modules[__name__]
del sys
run_hf = _module
run_hf_streaming = _module
run_vllm = _module
benchmark_e2e = _module
benchmark_e2e_vllm = _module
args = _module
compute_scores = _module
eval_utils = _module
run_infinitebench = _module
needle_summary = _module
needle_test = _module
needle_tools = _module
needle_viz = _module
run_ppl = _module
prepare = _module
common_words_extraction = _module
constants = _module
freq_words_extraction = _module
download_paulgraham_essay = _module
niah = _module
qa = _module
variable_tracking = _module
template = _module
tokenizer = _module
evaluate = _module
call_api = _module
client_wrappers = _module
model_wrappers = _module
serve_trt = _module
serve_vllm = _module
minference = _module
configs = _module
model2path = _module
minference_configuration = _module
models_patch = _module
modules = _module
inf_llm = _module
minference_forward = _module
snap_kv = _module
ops = _module
block_sparse_flash_attention = _module
flash_attn_triton = _module
pit_sparse_flash_attention = _module
pit_sparse_flash_attention_v2 = _module
streaming_kernel = _module
patch = _module
utils = _module
version = _module
setup = _module
test_e2e = _module

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


import logging


import numpy as np


import torch


import inspect


import warnings


import triton


import triton.language as tl


import math


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


@triton.jit
def _triton_block_sparse_attn_fwd_kernel(Q, K, V, seqlens, sm_scale, block_index, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok, Z, H, N_CTX, NUM_ROWS, MAX_BLOCKS_PRE_ROW, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', dtype: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    qo_offset = off_hz // H * stride_qz + off_hz % H * stride_qh
    kv_offset = off_hz // H * stride_kz + off_hz % H * stride_kh
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = q * qk_scale
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)
    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc, mask=m_mask)


@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit
def _fwd_kernel(Q, K, V, Bias, Out, Lse, TMP, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_ob, stride_oh, stride_om, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + (offs_m[:, None] * stride_bm + offs_n[None, :])
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float('-inf'))
        if BIAS_TYPE != 'none':
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=start_n + offs_n < seqlen_k, other=0.0)
                bias = bias[None, :]
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0)
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        elif EVEN_HEADDIM:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        p = p
        acc_o += tl.dot(p, v)
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
    else:
        tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))


@triton.autotune(configs=[triton.Config({}, num_stages=1, num_warps=4), triton.Config({}, num_stages=1, num_warps=8), triton.Config({}, num_stages=2, num_warps=4), triton.Config({}, num_stages=2, num_warps=8), triton.Config({}, num_stages=3, num_warps=4), triton.Config({}, num_stages=3, num_warps=8), triton.Config({}, num_stages=4, num_warps=4), triton.Config({}, num_stages=4, num_warps=8), triton.Config({}, num_stages=5, num_warps=4), triton.Config({}, num_stages=5, num_warps=8)], key=['N_CTX'])
@triton.jit
def triton_sparse_fwd_kernel(Q, K, V, seqlens, sm_scale, col_count, col_index, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok, Z, H, N_CTX, NUM_ROWS, MAX_COLS_PRE_ROW, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', dtype: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    qo_offset = off_hz // H * stride_qz + off_hz % H * stride_qh
    kv_offset = off_hz // H * stride_kz + off_hz % H * stride_kh
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    num_cols = tl.load(col_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = col_index + (off_hz * NUM_ROWS + start_m) * MAX_COLS_PRE_ROW
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = q * qk_scale
    m_mask = offs_m[:, None] < seqlen
    split = tl.maximum(num_cols - BLOCK_N, 0) & ~(BLOCK_N - 1)
    for start_n in range(0, split, BLOCK_N):
        cols = tl.load(cols_ptr + start_n + offs_n)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    for start_n in range(split, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=N_CTX - 1)
        causal_mask = cols[None, :] <= offs_m[:, None]
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & causal_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc, mask=m_mask)


@triton.jit
def triton_dense_fwd_kernel(Q, K, V, seqlens, sm_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok, Z, H, N_CTX, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', dtype: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return
    qo_offset = off_hz // H * stride_qz + off_hz % H * stride_qh
    kv_offset = off_hz // H * stride_kz + off_hz % H * stride_kh
    Q_block_ptr = tl.make_block_ptr(base=Q + qo_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + kv_offset, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + kv_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk), offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = q * qk_scale
    lo = 0
    hi = (start_m + 1) * BLOCK_M
    m_mask = offs_m[:, None] < seqlen
    for start_n in range(lo, hi, BLOCK_N):
        n_mask = start_n + offs_n[None, :] <= offs_m[:, None]
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    O_block_ptr = tl.make_block_ptr(base=Out + qo_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_ok), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(O_block_ptr, acc, mask=m_mask)


@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(Q, K, V, seqlens, sm_scale, block_count, block_offset, column_count, column_index, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok, Z, H, N_CTX, NUM_ROWS, NNZ_S, NNZ_V, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', dtype: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    qo_offset = off_hz // H * stride_qz + off_hz % H * stride_qh
    kv_offset = off_hz // H * stride_kz + off_hz % H * stride_kh
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = q * qk_scale
    m_mask = offs_m[:, None] < seqlen
    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    for start_n in range(0, num_cols, BLOCK_N):
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc, mask=m_mask)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, N_CTX, sliding_window_offset, sliding_window_size, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', SLIDING_WINDOW: 'tl.constexpr', IS_EVEN_M: 'tl.constexpr', IS_EVEN_N: 'tl.constexpr', COMPLEMENT_SLIDING_WINDOW: 'tl.constexpr'):
    if SLIDING_WINDOW and not COMPLEMENT_SLIDING_WINDOW:
        if COMPLEMENT_SLIDING_WINDOW:
            lo = 0
            hi = ((start_m + 1) * BLOCK_M + sliding_window_offset - sliding_window_size + BLOCK_N - 1) // BLOCK_N * BLOCK_N
        else:
            lo = (start_m * BLOCK_M + sliding_window_offset - sliding_window_size + 1) // BLOCK_N * BLOCK_N
            hi = ((start_m + 1) * BLOCK_M - 1 + sliding_window_offset + BLOCK_N) // BLOCK_N * BLOCK_N
            if lo < 0:
                lo = 0
            if hi > N_CTX:
                hi = N_CTX
            lo = tl.multiple_of(lo, BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, lo))
            V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    else:
        lo, hi = 0, N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if IS_EVEN_N:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale
        if SLIDING_WINDOW:
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[None, :] + start_m * BLOCK_M - start_n + sliding_window_offset
            if COMPLEMENT_SLIDING_WINDOW:
                mask = dist >= sliding_window_size
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)
            qk = tl.where(mask, qk, float('-inf'))
        if not IS_EVEN_N:
            qk = tl.where((tl.arange(0, BLOCK_N) + start_n < N_CTX)[None, :], qk, float('-inf'))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)
        if not IS_EVEN_N:
            p = tl.where((tl.arange(0, BLOCK_N) + start_n < N_CTX)[None, :], p, 0)
        l_ij = tl.sum(p, 1)
        tmp = m_i - m_ij
        alpha_mask = tmp != tmp
        alpha = tl.math.exp2(tmp)
        alpha = tl.where(alpha_mask, 1.0, alpha)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        if IS_EVEN_N:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')
        acc += tl.dot(p, v)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.heuristics({'IS_EVEN_M': lambda args: args['N_CTX'] % args['BLOCK_M'] == 0, 'IS_EVEN_N': lambda args: args['NKV_CTX'] % args['BLOCK_N'] == 0})
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, L, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, H_KV, N_CTX, ROUND_CTX, NKV_CTX, sliding_window_offset, sliding_window_size, IS_EVEN_M: 'tl.constexpr', IS_EVEN_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', END: 'tl.constexpr', INIT: 'tl.constexpr', SLIDING_WINDOW: 'tl.constexpr', COMPLEMENT_SLIDING_WINDOW: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H // H_KV)
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_hkv * stride_kh
    v_offset = off_z * stride_vz + off_hkv * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V + v_offset, shape=(NKV_CTX, BLOCK_DMODEL), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(BLOCK_DMODEL, NKV_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(ROUND_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_ptrs = M + off_hz * ROUND_CTX + offs_m
    l_ptrs = L + off_hz * ROUND_CTX + offs_m
    if INIT:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    else:
        m_i = tl.load(m_ptrs)
        l_i = tl.load(l_ptrs)
        acc = tl.load(O_block_ptr)
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634
    if IS_EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, NKV_CTX, sliding_window_offset, sliding_window_size, BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N, COMPLEMENT_SLIDING_WINDOW)
    if END:
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
    else:
        tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc)


@triton.heuristics({'IS_EVEN_M': lambda args: args['N_CTX'] % args['BLOCK_M'] == 0, 'IS_EVEN_N': lambda args: args['NKV_CTX'] % args['BLOCK_N'] == 0})
@triton.jit
def _score_kernel(Q, K, M, sm_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_oz, stride_oh, stride_on, Z, H, H_KV, N_CTX, ROUND_CTX, NKV_CTX, sliding_window_offset, sliding_window_size, SLIDING_WINDOW: 'tl.constexpr', COMPLEMENT_SLIDING_WINDOW: 'tl.constexpr', IS_EVEN_M: 'tl.constexpr', IS_EVEN_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H // H_KV)
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_hkv * stride_kh
    m_ptrs = M + off_hz * ROUND_CTX + tl.arange(0, BLOCK_M)
    o = tl.zeros([BLOCK_M], dtype=tl.float32)
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + k_offset, shape=(BLOCK_DMODEL, NKV_CTX), strides=(stride_kk, stride_kn), offsets=(0, start_n * BLOCK_N), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    if IS_EVEN_N:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
    lo = 0
    hi = ROUND_CTX
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        if IS_EVEN_M:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
        m = tl.load(m_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale
        if SLIDING_WINDOW:
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[None, :] + start_m - start_n * BLOCK_N + sliding_window_offset
            if COMPLEMENT_SLIDING_WINDOW:
                mask = dist >= sliding_window_size
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)
        qk = qk - m[:, None]
        p = tl.math.exp2(qk)
        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)
        if not IS_EVEN_N:
            p = tl.where((tl.arange(0, BLOCK_M) + start_m < N_CTX)[:, None], p, 0)
        o += tl.sum(p, axis=0)
        Q_block_ptr = tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))
        m_ptrs = m_ptrs + BLOCK_M
    o_offset = off_z * stride_oz + off_h * stride_oh
    o_range = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
    o_ptrs = Out + o_offset + o_range
    tl.store(o_ptrs, o, mask=o_range < NKV_CTX)

