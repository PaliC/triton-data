import sys
_module = sys.modules[__name__]
del sys
falcontune = _module
backend = _module
base = _module
cuda = _module
autograd = _module
quantlinear = _module
triton = _module
autograd = _module
custom_autotune = _module
flash_attn_triton = _module
quantlinear = _module
triton_utils = _module
data = _module
finetune = _module
generate = _module
model = _module
falcon = _module
config = _module
model = _module
gradient_checkpointing = _module
lora = _module
utils = _module
run = _module
setup = _module
setup_cuda = _module

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


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


import math


import time


from typing import Dict


import triton


import triton.language as tl


import warnings


from typing import Optional


from typing import Tuple


from typing import Union


import torch.utils.checkpoint


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import LayerNorm


from torch.nn import MSELoss


from torch.nn import functional as F


import re


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
        qk += tl.dot(q, k, trans_b=True)
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


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(configs=[triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'SEQUENCE_PARALLEL': False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'SEQUENCE_PARALLEL': True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ'))], key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM'])
@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit
def _bwd_kernel(Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_dob, stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm, stride_dkb, stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', SEQUENCE_PARALLEL: 'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != 'none':
        Bias += off_b * stride_bb + off_h * stride_bh
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD=False, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD=True, BIAS_TYPE=BIAS_TYPE, IS_CAUSAL=IS_CAUSAL, BLOCK_HEADDIM=BLOCK_HEADDIM, EVEN_M=EVEN_M, EVEN_N=EVEN_N, EVEN_HEADDIM=EVEN_HEADDIM, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

