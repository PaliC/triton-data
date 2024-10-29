import sys
_module = sys.modules[__name__]
del sys
download = _module
eval = _module
infinitebench_eval = _module
merge = _module
metrics = _module
pred = _module
inf_llm = _module
attention = _module
context_manager = _module
dot_production_attention = _module
base = _module
torch_impl = _module
triton_impl = _module
infinite_lm = _module
origin = _module
rope = _module
stream_llm = _module
utils = _module
chat = _module
greedy_search = _module
patch = _module
patch_mc = _module
setup = _module

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


from typing import Tuple


import math


import torch


import triton


import triton.language as tl


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

