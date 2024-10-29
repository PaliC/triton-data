import sys
_module = sys.modules[__name__]
del sys
benchmark_alibi = _module
benchmark_causal = _module
benchmark_flash_attention = _module
benchmark_gemm = _module
generate_kernels = _module
setup = _module
flash_attn = _module
bert_padding = _module
flash_attn_interface = _module
flash_attn_triton = _module
flash_attn_triton_og = _module
flash_blocksparse_attention = _module
flash_blocksparse_attn_interface = _module
fused_softmax = _module
layers = _module
patch_embed = _module
rotary = _module
losses = _module
cross_entropy = _module
models = _module
baichuan = _module
bert = _module
bigcode = _module
btlm = _module
falcon = _module
gpt = _module
gpt_neox = _module
gptj = _module
llama = _module
opt = _module
vit = _module
modules = _module
block = _module
embedding = _module
mha = _module
mlp = _module
ops = _module
activations = _module
fused_dense = _module
layer_norm = _module
rms_norm = _module
triton = _module
cross_entropy = _module
k_activations = _module
layer_norm = _module
linear = _module
mlp = _module
rotary = _module
utils = _module
benchmark = _module
distributed = _module
generation = _module
pretrained = _module
hopper = _module
benchmark_attn = _module
benchmark_flash_attention_fp8 = _module
benchmark_split_kv = _module
test_attn_kvcache = _module
test_flash_attn = _module
test_kvcache = _module
test_rotary = _module
test_cross_entropy = _module
test_cross_entropy_parallel = _module
test_baichuan = _module
test_bert = _module
test_bigcode = _module
test_btlm = _module
test_falcon = _module
test_gpt = _module
test_gpt_generation_parallel = _module
test_gpt_neox = _module
test_gpt_parallel = _module
test_gptj = _module
test_llama = _module
test_opt = _module
test_vit = _module
test_block_parallel = _module
test_embedding_parallel = _module
test_mha_parallel = _module
test_mlp_parallel = _module
test_dropout_layer_norm = _module
test_fused_dense = _module
test_fused_dense_parallel = _module
test_layer_norm = _module
test_flash_attn_ck = _module
test_rotary = _module
test_util = _module
run = _module
callbacks = _module
causality_monitor = _module
ema = _module
flop_count = _module
gpu_affinity = _module
loss_scale_monitor = _module
model_checkpoint = _module
norm_monitor = _module
params_log = _module
speed_monitor = _module
wandb_callbacks = _module
detokenizer = _module
lm_dataset = _module
fault_tolerant_sampler = _module
imagenet = _module
language_modeling_hf = _module
timm_mixup = _module
ddp_comm_hooks = _module
eval = _module
accuracy = _module
num_tokens = _module
perplexity = _module
seq_common = _module
param_grouping = _module
timm_lr_scheduler = _module
seq = _module
train = _module
checkpoint = _module
ddp_zero1 = _module
ddp_zero2 = _module
flops = _module
test_language_modeling_hf = _module

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


from functools import partial


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


import time


import torch.utils.benchmark as benchmark


from triton.testing import do_bench


import triton


import triton.language as tl


from typing import Optional


from typing import Tuple


from typing import Union


import logging


import re


from collections import OrderedDict


from collections.abc import Sequence


from typing import Any


from typing import Mapping


from collections import namedtuple


from typing import Dict


from typing import List


from copy import deepcopy


from torch.nn.init import trunc_normal_


from torchvision.ops import StochasticDepth


from torch import Tensor


from enum import Enum


from torch.cuda.amp import custom_fwd


from torch.cuda.amp import custom_bwd


from triton.ops.matmul_perf_model import early_config_prune


from triton.ops.matmul_perf_model import estimate_matmul_time


import random


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, TMP, L, M, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
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
    t_ptrs = TMP + off_hz * N_CTX + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(q_ptrs)
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k, trans_b=True)
        qk *= sm_scale
        qk += tl.where(offs_m[:, None] >= start_n + offs_n[None, :], 0, float('-inf'))
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
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + start_n * stride_vk)
        p = p
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


@triton.jit
def _bwd_preprocess_do_o_dot(Out, DO, Delta, stride_ob, stride_oh, stride_om, stride_dob, stride_doh, stride_dom, nheads, seqlen_q, seqlen_q_rounded, headdim, BLOCK_M: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M: 'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr'):
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    elif EVEN_HEADDIM:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
    else:
        tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
        tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD: 'tl.constexpr', BIAS_TYPE: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    begin_m = 0 if not IS_CAUSAL else start_n * BLOCK_N // BLOCK_M * BLOCK_M
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == 'vector':
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == 'matrix':
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)
        return
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    elif EVEN_HEADDIM:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
    else:
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        elif EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.dot(q, k, trans_b=True)
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float('-inf'))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk, float('-inf'))
        if BIAS_TYPE != 'none':
            tl.debug_barrier()
            if BIAS_TYPE == 'vector':
                if EVEN_N:
                    bias = tl.load(b_ptrs)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0)
                bias = bias[None, :]
            elif BIAS_TYPE == 'matrix':
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs)
                else:
                    bias = tl.load(b_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0)
            qk = qk * softmax_scale + bias
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == 'none':
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(p, do, trans_a=True)
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        dp = tl.dot(do, v, trans_b=True)
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        Di = tl.load(D + offs_m_curr)
        ds = p * (dp - Di[:, None]) * softmax_scale
        dk += tl.dot(ds, q, trans_a=True)
        if not EVEN_M & EVEN_HEADDIM:
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy='evict_last')
            elif EVEN_HEADDIM:
                dq = tl.load(dq_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q, eviction_policy='evict_last')
            else:
                dq = tl.load(dq_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0, eviction_policy='evict_last')
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), eviction_policy='evict_last')
        else:
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq)
            elif EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
            else:
                tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == 'matrix':
            b_ptrs += BLOCK_M * stride_bm
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM)


@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, M, D, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, Z, H, N_CTX, num_block, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_qz + off_h * stride_qh
    V += off_z * stride_qz + off_h * stride_qh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_qz + off_h * stride_qh
    DV += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        do_ptrs = DO + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            q = tl.load(q_ptrs)
            qk = tl.dot(q, k, trans_b=True)
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], qk, float('-inf'))
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            do = tl.load(do_ptrs)
            dv += tl.dot(p, do, trans_a=True)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, v, trans_b=True)
            ds = p * dp * sm_scale
            dk += tl.dot(ds, q, trans_a=True)
            dq = tl.load(dq_ptrs, eviction_policy='evict_last')
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq, eviction_policy='evict_last')
            dq_ptrs += BLOCK_M * stride_qm
            q_ptrs += BLOCK_M * stride_qm
            do_ptrs += BLOCK_M * stride_qm
        dv_ptrs = DV + (offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


@triton.jit
def _bwd_preprocess(Out, DO, L, NewDO, Delta, BLOCK_M: 'tl.constexpr', D_HEAD: 'tl.constexpr'):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :])
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :])
    denom = tl.load(L + off_m)
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.heuristics({'HAS_SMOOTHING': lambda args: args['smoothing'] > 0.0})
@triton.jit
def cross_entropy_fwd_kernel(loss_ptr, lse_ptr, z_loss_ptr, logits_ptr, labels_ptr, smoothing, logit_scale, lse_square_scale, ignore_index, total_classes, class_start_idx, n_cols, logits_row_stride, BLOCK_SIZE: 'tl.constexpr', HAS_SMOOTHING: 'tl.constexpr', SPLIT: 'tl.constexpr', PRECOMPUTED_LSE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    sum_logits = 0.0
    if not PRECOMPUTED_LSE:
        m_i = -float('inf')
        l_i = 0.0
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            logits = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float('inf')) * logit_scale
            if HAS_SMOOTHING:
                sum_logits += tl.sum(tl.where(cols < n_cols, logits, 0.0))
            m_i_new = tl.maximum(m_i, tl.max(logits))
            l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(tl.exp(logits - m_i_new))
            m_i = m_i_new
        lse = tl.log(l_i) + m_i
        tl.store(lse_ptr + row_idx, lse)
    else:
        lse = tl.load(lse_ptr + row_idx)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= 0 and label_idx < n_cols:
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (lse if not SPLIT else 0.0) - smoothing * sum_logits / total_classes - (1 - smoothing) * logits_label
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        elif HAS_SMOOTHING:
            loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
        else:
            loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + row_idx, z_loss)


@triton.heuristics({'HAS_SMOOTHING': lambda args: args['smoothing'] > 0.0})
@triton.jit
def cross_entropy_bwd_kernel(dlogits_ptr, dloss_ptr, logits_ptr, lse_ptr, labels_ptr, smoothing, logit_scale, lse_square_scale, ignore_index, total_classes, class_start_idx, n_cols, logits_row_stride, dlogits_row_stride, dloss_row_stride, BLOCK_SIZE: 'tl.constexpr', HAS_SMOOTHING: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != ignore_index:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float('inf')) * logit_scale
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx -= class_start_idx
    if HAS_SMOOTHING:
        smooth_positive = 1.0 - smoothing
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - smooth_positive, probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, dloss * logit_scale * probs, mask=col_offsets < n_cols)


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def cosh(x):
    exp_x = tl.exp(x)
    return (exp_x + 1.0 / exp_x) * 0.5


@triton.jit
def relu(x):
    """
    ReLU_ activation function

    .. _ReLU: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    """
    zero = 0.0
    return tl.where(x >= 0, x, zero)


@triton.jit
def relu_grad(x):
    zero = 0.0
    one = 1.0
    return tl.where(x >= 0, one, zero)


@triton.jit
def squared_relu(x):
    """
    Squared ReLU activation, as proposed in the Primer_ paper.

    .. _Primer: https://arxiv.org/abs/2109.08668
    """
    x_ = relu(x)
    return x_ * x_


@triton.jit
def squared_relu_grad(x):
    return tl.where(x >= 0, 2.0 * x, 0.0)


@triton.jit
def leaky_relu(x):
    """
    LeakyReLU_ activation

    .. _LeakyReLU: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    """
    scale = 0.01 + 0.0
    scale = scale
    return tl.where(x >= 0, x, scale * x)


@triton.jit
def leaky_relu_grad(x):
    min_grad = 0.01
    max_grad = 1
    min_grad = min_grad
    max_grad = max_grad
    return tl.where(x >= 0, max_grad, min_grad)


_sqrt1_2 = math.sqrt(1.0 / 2)


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))


_gaussian_pdf_normalization = 1.0 / math.sqrt(2 * math.pi)


@triton.jit
def gelu_grad(x):
    cdf = 0.5 * (1.0 + tl.libdevice.erf(x * _sqrt1_2))
    pdf = tl.exp(-0.5 * x * x) * _gaussian_pdf_normalization
    return cdf + x * pdf


_sqrt2pi = math.sqrt(2.0 / math.pi)


@triton.jit
def gelu_approx(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def gelu_approx_grad(x):
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM', 'HAS_BIAS'])
@triton.heuristics({'HAS_X1': lambda args: args['X1'] is not None})
@triton.heuristics({'HAS_W1': lambda args: args['W1'] is not None})
@triton.heuristics({'HAS_B1': lambda args: args['B1'] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(X, Y, W, B, RESIDUAL, X1, W1, B1, Y1, RESIDUAL_OUT, ROWSCALE, SEEDS, DROPOUT_MASK, Mean, Rstd, stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, stride_x1_row, stride_y1_row, M, N, eps, dropout_p, IS_RMS_NORM: 'tl.constexpr', BLOCK_N: 'tl.constexpr', HAS_RESIDUAL: 'tl.constexpr', STORE_RESIDUAL_OUT: 'tl.constexpr', HAS_BIAS: 'tl.constexpr', HAS_DROPOUT: 'tl.constexpr', STORE_DROPOUT_MASK: 'tl.constexpr', HAS_ROWSCALE: 'tl.constexpr', HAS_X1: 'tl.constexpr', HAS_W1: 'tl.constexpr', HAS_B1: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row)
        x *= rowscale
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row)
            x1 *= rowscale
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + M + row), cols, n_rounds=7) > dropout_p
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK + (M + row) * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask)
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL', 'IS_RMS_NORM', 'HAS_BIAS', 'HAS_DROPOUT'])
@triton.heuristics({'HAS_ROWSCALE': lambda args: args['ROWSCALE'] is not None})
@triton.heuristics({'HAS_DY1': lambda args: args['DY1'] is not None})
@triton.heuristics({'HAS_DX1': lambda args: args['DX1'] is not None})
@triton.heuristics({'HAS_B1': lambda args: args['DB1'] is not None})
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['Y'] is not None})
@triton.jit
def _layer_norm_bwd_kernel(X, W, B, Y, DY, DX, DW, DB, DRESIDUAL, W1, DY1, DX1, DW1, DB1, DRESIDUAL_IN, ROWSCALE, SEEDS, Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row, stride_dx_row, stride_dres_row, stride_dy1_row, stride_dx1_row, stride_dres_in_row, M, N, eps, dropout_p, rows_per_program, IS_RMS_NORM: 'tl.constexpr', BLOCK_N: 'tl.constexpr', HAS_DRESIDUAL: 'tl.constexpr', STORE_DRESIDUAL: 'tl.constexpr', HAS_BIAS: 'tl.constexpr', HAS_DROPOUT: 'tl.constexpr', HAS_ROWSCALE: 'tl.constexpr', HAS_DY1: 'tl.constexpr', HAS_DX1: 'tl.constexpr', HAS_B1: 'tl.constexpr', RECOMPUTE_OUTPUT: 'tl.constexpr'):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if HAS_DY1:
        DY1 += row_start * stride_dy1_row
    if HAS_DX1:
        DX1 += row_start * stride_dx1_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0)
    if HAS_DY1:
        w1 = tl.load(W1 + cols, mask=mask)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_DY1:
        dw1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_B1:
            db1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0)
        dy = tl.load(DY + cols, mask=mask, other=0)
        if HAS_DY1:
            dy1 = tl.load(DY1 + cols, mask=mask, other=0)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_DY1:
            wdy += w1 * dy1
            dw1 += dy1 * xhat
            if HAS_B1:
                db1 += dy1
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        if HAS_DX1:
            if HAS_DROPOUT:
                keep_mask = tl.rand(tl.load(SEEDS + M + row), cols, n_rounds=7) > dropout_p
                dx1 = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
            else:
                dx1 = dx
            tl.store(DX1 + cols, dx1, mask=mask)
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + row), cols, n_rounds=7) > dropout_p
            dx = tl.where(keep_mask, dx / (1.0 - dropout_p), 0.0)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + row)
            dx *= rowscale
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
        if HAS_DY1:
            DY1 += stride_dy1_row
        if HAS_DX1:
            DX1 += stride_dx1_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)
    if HAS_DY1:
        tl.store(DW1 + row_block_id * N + cols, dw1, mask=mask)
        if HAS_B1:
            tl.store(DB1 + row_block_id * N + cols, db1, mask=mask)


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1}, num_stages=num_stages, num_warps=num_warps))
    return configs


@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2)] + get_configs_io_bound(), key=['CACHE_KEY_M', 'CACHE_KEY_N', 'CACHE_KEY_K'], prune_configs_by={'early_config_prune': early_config_prune, 'perf_model': estimate_matmul_time, 'top_k': 10})
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0})
@triton.jit
def kernel_fwd(C, ACT_INPUT, A, B, bias, M, N, K, CACHE_KEY_M, CACHE_KEY_N, CACHE_KEY_K, stride_cm, stride_am, stride_ak, stride_bn, stride_bk, BLOCK_M: 'tl.constexpr', GROUP_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', SPLIT_K: 'tl.constexpr', EVEN_K: 'tl.constexpr', A_ROWMAJOR: 'tl.constexpr', B_COLMAJOR: 'tl.constexpr', BIAS: 'tl.constexpr', SAVE_ACT_INPUT: 'tl.constexpr', ACTIVATION: 'tl.constexpr'):
    """
    Kernel for computing Out = activation(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Bias has shape (N,)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)
    'ActInputs' optionally saves the A x W + C intermediate for backward computations
    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    if A_ROWMAJOR:
        A = A + (ram[:, None] * stride_am + rk[None, :])
    else:
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    if B_COLMAJOR:
        B = B + (rk[:, None] + rbn[None, :] * stride_bn)
    else:
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        if A_ROWMAJOR:
            A += BLOCK_K
        else:
            A += BLOCK_K * stride_ak
        if B_COLMAJOR:
            B += BLOCK_K
        else:
            B += BLOCK_K * stride_bk
    if BIAS:
        bias = tl.load(bias + rn, mask=rn < N, other=0.0)
        acc += bias[None, :]
    if SAVE_ACT_INPUT:
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        tl.store(act_in_ptrs, acc)
    if ACTIVATION == 'gelu':
        acc = gelu(acc)
    elif ACTIVATION == 'gelu_approx':
        acc = gelu_approx(acc)
    elif ACTIVATION == 'squared_relu':
        acc = squared_relu(acc)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc)


@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2)] + get_configs_io_bound(), key=['CACHE_KEY_M', 'CACHE_KEY_N', 'CACHE_KEY_K'], prune_configs_by={'early_config_prune': early_config_prune, 'perf_model': estimate_matmul_time, 'top_k': 10})
@triton.heuristics({'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0})
@triton.jit
def kernel_bwd(C, ACT_INPUT, A, B, M, N, K, CACHE_KEY_M, CACHE_KEY_N, CACHE_KEY_K, stride_cm, stride_am, stride_ak, stride_bk, stride_bn, BLOCK_M: 'tl.constexpr', GROUP_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', SPLIT_K: 'tl.constexpr', EVEN_K: 'tl.constexpr', ACTIVATION: 'tl.constexpr'):
    """
    Kernel for computing Out = activation(A x W + C)
    - Input has shape (M, K)
    - Weight has shape (K, N)
    - Output has shape (M, N)
    - ActInputs (optional) has shape (M, N)
    'ActInputs' optionally saves the A x W + C intermediate for backward computations
    This kernel will consolidate over K
    """
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    if ACTIVATION != 'id':
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        act_input = tl.load(act_in_ptrs)
    if ACTIVATION == 'gelu':
        acc *= gelu_grad(act_input)
    elif ACTIVATION == 'gelu_approx':
        acc *= gelu_approx_grad(act_input)
    elif ACTIVATION == 'squared_relu':
        acc *= squared_relu_grad(act_input)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen, rotary_dim, seqlen_ro, stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim, stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim, BLOCK_K: 'tl.constexpr', IS_SEQLEN_OFFSETS_TENSOR: 'tl.constexpr', IS_VARLEN: 'tl.constexpr', INTERLEAVED: 'tl.constexpr', CONJUGATE: 'tl.constexpr', BLOCK_M: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2
    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads
    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)
    if not INTERLEAVED:
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        x0 = tl.load(X, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        x1 = tl.load(X + rotary_dim_half * stride_x_headdim, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half), other=0.0)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim)
        tl.store(OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
        tl.store(OUT + rotary_dim_half * stride_out_headdim, o1, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    else:
        rk_swap = rk + (rk + 1) % 2 * 2 - 1
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(COS, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=1.0)
        sin = tl.load(SIN, mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half), other=0.0)
        x0 = tl.load(X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0)
        x1 = tl.load(X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))

