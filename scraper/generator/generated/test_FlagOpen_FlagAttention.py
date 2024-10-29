import sys
_module = sys.modules[__name__]
del sys
flash_benchmark = _module
flash_decoding_benchmark = _module
paged_benchmark = _module
piecewise_benchmark = _module
flash_attention_example = _module
flash_attention_with_aux_outputs = _module
paged_example = _module
piecewise_example = _module
use_cutom_config_func = _module
flag_attn = _module
dropout = _module
flash = _module
paged = _module
piecewise = _module
split_kv = _module
testing = _module
dropout = _module
total = _module
test_dropout = _module
test_flash_attention = _module
test_paged_attention = _module
test_piecewise_attention = _module

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


import triton


import triton.language as tl


@triton.jit
def _fwd_kernel(Q1, K1, Q2, K2, V, sm_scale, L, O, stride_q1z, stride_q1h, stride_q1m, stride_q1k, stride_k1z, stride_k1h, stride_k1n, stride_k1k, stride_q2z, stride_q2h, stride_q2m, stride_q2k, stride_k2z, stride_k2h, stride_k2n, stride_k2k, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_ok, Z, H, M, N, P_SEQ, w: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', LARGER_M: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    input_dtype = Q1.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q1_ptrs = Q1 + (offs_m[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
    q2_ptrs = Q2 + (offs_m[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
    k1_ptrs = K1 + (offs_n_init[:, None] * stride_k1n + offs_k[None, :] * stride_k1k)
    k2_ptrs = K2 + (offs_n_init[:, None] * stride_k2n + offs_k[None, :] * stride_k2k)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        q1 = tl.load(q1_ptrs)
        q2 = tl.load(q2_ptrs)
    else:
        mask_m = offs_m < M
        q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
        q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
    I = tl.where(offs_k[:, None] == offs_k, tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype), tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
    q1 = tl.dot(q1, I)
    q2 = tl.dot(q2, I)
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        piecewise_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :] + w
        if DIVISIBLE_N:
            k1 = tl.load(k1_ptrs)
            k2 = tl.load(k2_ptrs)
            v = tl.load(v_ptrs)
        else:
            mask_n = offs_n < N
            k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
            k2 = tl.load(k2_ptrs, mask=mask_n[:, None])
            v = tl.load(v_ptrs, mask=mask_n[:, None])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(piecewise_mask, tl.dot(q2, tl.trans(k2)), tl.dot(q1, tl.trans(k1)))
        if not DIVISIBLE_N:
            s = tl.where(mask_n, s, float('-inf'))
        if IS_CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn
    if IS_CAUSAL and LARGER_M:
        is_empty_line = offs_m + P_SEQ < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l_i = tl.where(is_empty_line, float('-inf'), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l_i = m_i * sm_scale + tl.log(l_i)
    if DIVISIBLE_M:
        tl.store(l_ptrs, l_i)
        tl.store(o_ptrs, acc)
    else:
        tl.store(l_ptrs, l_i, mask=mask_m)
        tl.store(o_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def _bwd_preprocess(Out, DO, Delta, stride_oz, stride_oh, stride_om, stride_ok, stride_doz, stride_doh, stride_dom, stride_dok, stride_dz, stride_dh, stride_dm, M, BLOCK_M: 'tl.constexpr', D_HEAD: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr'):
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok
    if DIVISIBLE_M:
        o = tl.load(o_ptrs)
        do = tl.load(do_ptrs)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
    delta = tl.sum(o * do, axis=1)
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kv_kernel(Q1, K1, Q2, K2, V, sm_scale, DO, DK1, DK2, DV, L, D, stride_q1z, stride_q1h, stride_q1m, stride_q1k, stride_k1z, stride_k1h, stride_k1n, stride_k1k, stride_q2z, stride_q2h, stride_q2m, stride_q2k, stride_k2z, stride_k2h, stride_k2n, stride_k2k, stride_vz, stride_vh, stride_vn, stride_vk, stride_doz, stride_doh, stride_dom, stride_dok, stride_dk1z, stride_dk1h, stride_dk1n, stride_dk1k, stride_dk2z, stride_dk2h, stride_dk2n, stride_dk2k, stride_dvz, stride_dvh, stride_dvn, stride_dvk, Z, H, M, N, P_SEQ, w: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', CAUSAL: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    input_dtype = Q1.dtype.element_ty
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M
    DK1 += off_z * stride_dk1z + off_h * stride_dk1h
    DK2 += off_z * stride_dk2z + off_h * stride_dk2h
    DV += off_z * stride_dvz + off_h * stride_dvh
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = lo // BLOCK_M * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q1_ptrs = Q1 + (offs_m_init[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
    q2_ptrs = Q2 + (offs_m_init[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
    k1_ptrs = K1 + (offs_k[:, None] * stride_k1k + offs_n[None, :] * stride_k1n)
    k2_ptrs = K2 + (offs_k[:, None] * stride_k2k + offs_n[None, :] * stride_k2n)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok)
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
    dk1_ptrs = DK1 + (offs_n[:, None] * stride_dk1n + offs_k[None, :] * stride_dk1k)
    dk2_ptrs = DK2 + (offs_n[:, None] * stride_dk2n + offs_k[None, :] * stride_dk2k)
    if DIVISIBLE_N:
        k1 = tl.load(k1_ptrs)
        k2 = tl.load(k2_ptrs)
        v = tl.load(v_ptrs)
    else:
        mask_n = offs_n < N
        k1 = tl.load(k1_ptrs, mask=mask_n[None, :])
        k2 = tl.load(k2_ptrs, mask=mask_n[None, :])
        v = tl.load(v_ptrs, mask=mask_n[:, None])
    dk1 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        if DIVISIBLE_M:
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
            do = tl.load(do_ptrs)
            delta = tl.load(D + offs_m)
            l = tl.load(L + offs_m)
        else:
            mask_m = offs_m < M
            q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
            q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
            do = tl.load(do_ptrs, mask=mask_m[:, None])
            delta = tl.load(D + offs_m, mask=mask_m)
            l = tl.load(L + offs_m, mask=mask_m)
        piecewise_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :] + w
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(piecewise_mask, tl.dot(q2, k2), tl.dot(q1, k1))
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        if not DIVISIBLE_M:
            valid_mask = mask_m[:, None]
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            p = tl.where(causal_mask, p, 0.0)
        dv += tl.dot(tl.trans(p), do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_M:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds2 = tl.where(piecewise_mask, ds, 0.0)
        ds1 = tl.where(piecewise_mask, 0.0, ds)
        dk1 += tl.dot(tl.trans(ds1), q1)
        dk2 += tl.dot(tl.trans(ds2), q2)
        q1_ptrs += BLOCK_M * stride_q1m
        q2_ptrs += BLOCK_M * stride_q2m
        do_ptrs += BLOCK_M * stride_dom
    dk1 *= sm_scale
    dk2 *= sm_scale
    if DIVISIBLE_N:
        tl.store(dk1_ptrs, dk1)
        tl.store(dk2_ptrs, dk2)
        tl.store(dv_ptrs, dv)
    else:
        tl.store(dk1_ptrs, dk1, mask=mask_n[:, None])
        tl.store(dk2_ptrs, dk2, mask=mask_n[:, None])
        tl.store(dv_ptrs, dv, mask=mask_n[:, None])


@triton.jit
def _bwd_q_kernel(Q1, K1, Q2, K2, V, sm_scale, DO, DQ1, DQ2, L, D, stride_q1z, stride_q1h, stride_q1m, stride_q1k, stride_k1z, stride_k1h, stride_k1n, stride_k1k, stride_q2z, stride_q2h, stride_q2m, stride_q2k, stride_k2z, stride_k2h, stride_k2n, stride_k2k, stride_vz, stride_vh, stride_vn, stride_vk, stride_doz, stride_doh, stride_dom, stride_dok, stride_dq1z, stride_dq1h, stride_dq1m, stride_dq1k, stride_dq2z, stride_dq2h, stride_dq2m, stride_dq2k, Z, H, M, N, P_SEQ, w: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', CAUSAL: 'tl.constexpr', LARGER_M: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    input_dtype = Q1.dtype.element_ty
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K1 += off_z * stride_k1z + off_h * stride_k1h
    K2 += off_z * stride_k2z + off_h * stride_k2h
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    D += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M
    DQ1 += off_z * stride_dq1z + off_h * stride_dq1h
    DQ2 += off_z * stride_dq2z + off_h * stride_dq2h
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q1_ptrs = Q1 + (offs_m[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
    q2_ptrs = Q2 + (offs_m[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
    k1_ptrs = K1 + (offs_n_init[:, None] * stride_k1n + offs_k[None, :] * stride_k1k)
    k2_ptrs = K2 + (offs_n_init[:, None] * stride_k2n + offs_k[None, :] * stride_k2k)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    dq1_ptrs = DQ1 + (offs_m[:, None] * stride_dq1m + offs_k[None, :] * stride_dq1k)
    dq2_ptrs = DQ2 + (offs_m[:, None] * stride_dq2m + offs_k[None, :] * stride_dq2k)
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok)
    d_ptrs = D + offs_m
    l_ptrs = L + offs_m
    if DIVISIBLE_M:
        q1 = tl.load(q1_ptrs)
        q2 = tl.load(q2_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
    else:
        mask_m = offs_m < M
        q1 = tl.load(q1_ptrs, mask=mask_m[:, None])
        q2 = tl.load(q2_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)
    dq1 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dq2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k1 = tl.load(k1_ptrs)
            k2 = tl.load(k2_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k1 = tl.load(k1_ptrs, mask=mask_n[:, None])
            k2 = tl.load(k2_ptrs, mask=mask_n[:, None])
        piecewise_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :] + w
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.where(piecewise_mask, tl.dot(q2, tl.trans(k2)), tl.dot(q1, tl.trans(k1)))
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        if not DIVISIBLE_N:
            ds = tl.where(mask_n, ds, 0.0)
        if CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            ds = tl.where(causal_mask, ds, 0.0)
        ds2 = tl.where(piecewise_mask, ds, 0.0)
        ds1 = tl.where(piecewise_mask, 0.0, ds)
        dq1 += tl.dot(ds1, k1)
        dq2 += tl.dot(ds2, k2)
        k1_ptrs += BLOCK_N * stride_k1n
        k2_ptrs += BLOCK_N * stride_k2n
        v_ptrs += BLOCK_N * stride_vn
    dq1 *= sm_scale
    dq2 *= sm_scale
    if DIVISIBLE_M:
        tl.store(dq1_ptrs, dq1)
        tl.store(dq2_ptrs, dq2)
    else:
        tl.store(dq1_ptrs, dq1, mask=mask_m[:, None])
        tl.store(dq2_ptrs, dq2, mask=mask_m[:, None])


def get_num_stages(PARTITION_SIZE, KV_BLOCK_SIZE):
    if PARTITION_SIZE == 0:
        return 1
    elif torch.cuda.get_device_capability() == (8, 0):
        if KV_BLOCK_SIZE < 256:
            return 3
        else:
            return 2
    elif torch.cuda.get_device_capability() == (8, 6):
        if KV_BLOCK_SIZE < 256:
            return 2
        else:
            return 1
    else:
        return 1


def get_num_warps(QUERY_GROUP_SIZE, HEAD_SIZE, KV_BLOCK_SIZE):
    if QUERY_GROUP_SIZE == 1:
        if HEAD_SIZE >= 128 and KV_BLOCK_SIZE >= 32:
            return 16
        else:
            return 8
    else:
        return 4


@triton.heuristics({'num_warps': lambda args: get_num_warps(args['QUERY_GROUP_SIZE'], args['HEAD_SIZE'], args['KV_BLOCK_SIZE']), 'num_stages': lambda args: get_num_stages(args['QUERY_GROUP_SIZE'], args['KV_BLOCK_SIZE'])})
@triton.jit
def _paged_attn_kernel(m_i_ptr, l_i_ptr, out_ptr, q_ptr, k_cache_ptr, v_cache_ptr, context_lens_ptr, block_tables_ptr, attn_scale, stride_bt0, stride_bt1, stride_q0, stride_q1, stride_q2, stride_kv0, stride_kv1, stride_kv2, stride_kv3, stride_o0, stride_o1, stride_o2, stride_o3, stride_o4, HEAD_SIZE: 'tl.constexpr', QUERY_GROUP_SIZE: 'tl.constexpr', PADDED_QUERY_GROUP_SIZE: 'tl.constexpr', NUM_KV_HEADS: 'tl.constexpr', KV_BLOCK_SIZE: 'tl.constexpr', PARTITION_SIZE: 'tl.constexpr'):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx)
    if USE_PARTITIONING:
        context_start_idx = part_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
        num_blocks = tl.cdiv(context_end_idx - context_start_idx, KV_BLOCK_SIZE)
    else:
        num_blocks = tl.cdiv(context_len, KV_BLOCK_SIZE)
    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    head_offset = tl.arange(0, HEAD_SIZE)
    padding_group_offset = tl.arange(0, PADDED_QUERY_GROUP_SIZE)
    kv_offset = kv_head_idx * stride_kv1 + block_offset[:, None] * stride_kv2 + head_offset[None, :] * stride_kv3
    q_offset = seq_idx * stride_q0 + (kv_head_idx * QUERY_GROUP_SIZE + padding_group_offset[:, None]) * stride_q1 + head_offset[None, :] * stride_q2
    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    q = tl.load(q_ptr + q_offset, mask=group_mask, other=0.0)
    m_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)
    num_prev_blocks = part_idx * (PARTITION_SIZE // KV_BLOCK_SIZE)
    for i in range(num_blocks):
        block_idx = num_prev_blocks + i
        block_number = tl.load(block_tables_ptr + seq_idx * stride_bt0 + block_idx * stride_bt1)
        kv_block_offset = block_number * stride_kv0 + kv_offset
        mask_offset = block_idx * KV_BLOCK_SIZE + block_offset
        kv_mask = mask_offset[:, None] < context_len
        k = tl.load(k_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)
        if PADDED_QUERY_GROUP_SIZE == 1:
            qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
        else:
            qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk *= attn_scale
        qk = tl.where(mask_offset < context_len, qk, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2((qk - m_i_new[:, None]) * log2e)
        alpha = tl.math.exp2((m_i - m_i_new) * log2e)
        acc *= alpha[:, None]
        v = tl.load(v_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)
        if PADDED_QUERY_GROUP_SIZE == 1:
            acc += tl.sum(p.T[:, :, None] * v[:, None, :], axis=0)
        else:
            p = p
            acc += tl.dot(p, v, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    acc = acc / l_i[:, None]
    if USE_PARTITIONING:
        part_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE + part_idx * QUERY_GROUP_SIZE + padding_group_offset
        mask = padding_group_offset < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + part_offset, m_i, mask=mask)
        tl.store(l_i_ptr + part_offset, l_i, mask=mask)
    out_offset = seq_idx * stride_o0
    if USE_PARTITIONING:
        out_offset += kv_head_idx * stride_o1
    else:
        out_offset += kv_head_idx * QUERY_GROUP_SIZE * stride_o1
    out_offset += part_idx * stride_o2 + padding_group_offset[:, None] * stride_o3 + head_offset[None, :] * stride_o4
    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE
    tl.store(out_ptr + out_offset, acc, mask=group_mask)


@triton.jit
def _paged_attn_v2_reduce_kernel(out_ptr, m_i_ptr, l_i_ptr, tmp_out_ptr, context_lens_ptr, max_num_partitions, stride_o0, stride_o1, stride_o2, HEAD_SIZE: 'tl.constexpr', QUERY_GROUP_SIZE: 'tl.constexpr', NUM_KV_HEADS: 'tl.constexpr', PARTITION_SIZE: 'tl.constexpr', NUM_PARTITIONS: 'tl.constexpr'):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    context_len = tl.load(context_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(context_len, PARTITION_SIZE)
    group_head_offset = tl.arange(0, QUERY_GROUP_SIZE)[:, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, :]
    if num_partitions == 1:
        tmp_out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE + group_head_offset
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset)
        out_offset = seq_idx * stride_o0 + kv_head_idx * QUERY_GROUP_SIZE * stride_o1 + group_head_offset * stride_o2
        tl.store(out_ptr + out_offset, tmp_out)
        return
    ml_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE + tl.arange(0, NUM_PARTITIONS)[:, None] * QUERY_GROUP_SIZE + tl.arange(0, QUERY_GROUP_SIZE)[None, :]
    mask = tl.arange(0, NUM_PARTITIONS)[:, None] < num_partitions
    m_i = tl.load(m_i_ptr + ml_offset, mask=mask, other=float('-inf'))
    m = tl.max(m_i, axis=0)
    l_i = tl.load(l_i_ptr + ml_offset, mask=mask, other=0.0)
    l_i *= tl.exp(m_i - m[None, :])
    l = tl.sum(l_i, axis=0)
    r = l_i / l[None, :]
    r = tl.reshape(r, (NUM_PARTITIONS, QUERY_GROUP_SIZE, 1))
    tmp_out_offset = (seq_idx * NUM_KV_HEADS + kv_head_idx) * max_num_partitions * QUERY_GROUP_SIZE * HEAD_SIZE + tl.arange(0, NUM_PARTITIONS)[:, None, None] * QUERY_GROUP_SIZE * HEAD_SIZE + tl.arange(0, QUERY_GROUP_SIZE)[None, :, None] * HEAD_SIZE + tl.arange(0, HEAD_SIZE)[None, None, :]
    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None], other=0.0)
    out = tl.sum(tmp_out * r, axis=0)
    out_offset = seq_idx * stride_o0 + kv_head_idx * QUERY_GROUP_SIZE * stride_o1 + group_head_offset * stride_o2
    tl.store(out_ptr + out_offset, out)


@triton.jit
def _fwd_split_kv_kernel(Q, K, V, sm_scale, L, O, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_os, stride_om, stride_ok, Z, H, M, N, P_SEQ, N_SPLIT_SIZE, S, num_groups, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', LARGER_M: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    input_dtype = Q.dtype.element_ty
    start_m = tl.program_id(0)
    n_split_id = tl.program_id(1)
    off_zh = tl.program_id(2)
    off_h = off_zh % H
    off_z = off_zh // H
    off_hk = off_h // num_groups
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh + n_split_id * stride_os
    L += ((off_z * H + off_h) * S + n_split_id) * M
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    l_ptrs = L + offs_m
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k, tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype), tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I)
    N_LEFT = n_split_id * N_SPLIT_SIZE
    N_RIGHT = tl.minimum(N_LEFT + N_SPLIT_SIZE, N)
    if IS_CAUSAL:
        hi = tl.minimum(N_RIGHT, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(N_LEFT, hi)
    else:
        hi = N_RIGHT
    offs_n_init = N_LEFT + offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    for start_n in range(N_LEFT, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier='.cg')
            v = tl.load(v_ptrs, cache_modifier='.cg')
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier='.cg')
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier='.cg')
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)
        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float('-inf'))
        if IS_CAUSAL:
            causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, float('-inf'))
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    if IS_CAUSAL and LARGER_M:
        is_empty_line = offs_m + P_SEQ < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float('-inf'), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * sm_scale + tl.log(l_i)
    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier='.cg')
        tl.store(o_ptrs, acc, cache_modifier='.cg')
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier='.cg')
        tl.store(o_ptrs, acc, mask=mask_m[:, None], cache_modifier='.cg')


@triton.jit
def _fwd_combine_kv_splits(multiple_o, multiple_l, final_o, final_l, stride_mul_oz, stride_mul_oh, stride_mul_os, stride_mul_om, stride_mul_ok, stride_fin_oz, stride_fin_oh, stride_fin_om, stride_fin_ok, Z, H, M, S, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr'):
    start_m = tl.program_id(0)
    offs_h = tl.program_id(1)
    offs_z = tl.program_id(2)
    multiple_o += offs_z * stride_mul_oz + offs_h * stride_mul_oh
    multiple_l += (offs_z * H + offs_h) * S * M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not DIVISIBLE_M:
        mask_m = offs_m < M
    m = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    acc = tl.full([BLOCK_M], value=float(0.0), dtype=tl.float32)
    l_ptrs = multiple_l + offs_m
    for _ in range(0, S):
        if DIVISIBLE_M:
            l = tl.load(l_ptrs)
        else:
            l = tl.load(l_ptrs, mask=mask_m)
        m_new = tl.maximum(m, l)
        acc = acc * tl.exp(m - m_new) + tl.exp(l - m_new)
        m = m_new
        l_ptrs += M
    l_acc = m + tl.log(acc)
    o_acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_ptrs = multiple_l + offs_m
    offs_k = tl.arange(0, BLOCK_DMODEL)
    o_ptrs = multiple_o + offs_m[:, None] * stride_mul_om + offs_k[None, :] * stride_mul_ok
    for _ in range(0, S):
        l = tl.load(l_ptrs, mask=offs_m < M)
        rescale = tl.exp(l - l_acc)
        if DIVISIBLE_M:
            o = tl.load(o_ptrs)
        else:
            o = tl.load(o_ptrs, mask=mask_m[:, None])
        o_acc += o * rescale[:, None]
        l_ptrs += M
        o_ptrs += stride_mul_os
    final_o += offs_z * stride_fin_oz + offs_h * stride_fin_oh
    final_l += (offs_z * H + offs_h) * M
    a_ptrs = final_o + offs_m[:, None] * stride_fin_om + offs_k * stride_fin_ok
    b_ptrs = final_l + offs_m
    if DIVISIBLE_M:
        tl.store(a_ptrs, o_acc)
        tl.store(b_ptrs, l_acc)
    else:
        tl.store(a_ptrs, o_acc, mask=mask_m[:, None])
        tl.store(b_ptrs, l_acc, mask=mask_m)


@triton.jit
def recompute_mask_kernel(mask, B, H, M, N, dropout_p, seed, offset):
    row, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = b * H * M * N + h * M * N + row * N
    BLOCK: 'tl.constexpr' = 1024
    offs_base += tl.arange(0, BLOCK)
    for start_n in range(0, N, BLOCK):
        offs = start_n + offs_base
        rng_offs = offset + offs
        pmask = tl.rand(seed, rng_offs, n_rounds=6) > dropout_p
        row_mask = start_n + tl.arange(0, BLOCK) < N
        tl.store(mask + offs, pmask, mask=row_mask)


@triton.jit
def _total_attention_kernel(Q, K, L, TA, sm_scale, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, Z, H, M, N, P_SEQ, num_groups, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', CAUSAL: 'tl.constexpr', DIVISIBLE_M: 'tl.constexpr', DIVISIBLE_N: 'tl.constexpr'):
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: 'tl.constexpr' = 1.4426950408889634
    qk_scale = sm_scale * log2e
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    L += (off_z * H + off_h) * M
    TA += (off_z * H + off_h) * N
    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = lo // BLOCK_M * BLOCK_M
    else:
        lo = 0
    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    ta_ptrs = TA + offs_n
    if DIVISIBLE_N:
        k = tl.load(k_ptrs)
    else:
        mask_n = offs_n < N
        k = tl.load(k_ptrs, mask=mask_n[:, None])
    tot_attn = tl.zeros([BLOCK_N], dtype=tl.float32)
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = P_SEQ + offs_m[:, None] >= offs_n[None, :]
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[:, None]
            q = tl.load(q_ptrs, mask=mask_m[:, None])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)
        if not DIVISIBLE_M:
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)
        tot_attn += tl.sum(p, 0)
        q_ptrs += BLOCK_M * stride_qm
    if DIVISIBLE_N:
        tl.store(ta_ptrs, tot_attn)
    else:
        tl.store(ta_ptrs, tot_attn, mask=mask_n)

