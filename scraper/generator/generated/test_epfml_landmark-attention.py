import sys
_module = sys.modules[__name__]
del sys
llama_landmark_config = _module
llama_mem = _module
flash_landmark_attention = _module
test_flash_landmark_attention = _module
redpajama = _module
run_test = _module
train = _module
weight_diff = _module
config = _module
rotary = _module
data = _module
arxiv_math = _module
pg19 = _module
prepare = _module
utils = _module
distributed = _module
backend = _module
ddp = _module
single = _module
eval = _module
eval_cmd_generator = _module
main = _module
models = _module
base_new = _module
caches = _module
cache = _module
kv_cache = _module
kv_cache_train = _module
mem_cache = _module
landmark = _module
landmark_with_cmt = _module
positional_encoders = _module
encoder = _module
rotary_mem_jump = _module
rotary_utils = _module
base = _module
transformer_xl = _module

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


import triton


import torch


import triton.language as tl


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, Out, sqz, sqh, sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd, soz, soh, som, sod, L, M, Z, H, N_CTX_Q, N_CTX_KV, BLOCK: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', N_PREFIX_Q: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    BLOCK_M: 'tl.constexpr' = BLOCK
    BLOCK_N: 'tl.constexpr' = BLOCK
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_real = (start_m + N_PREFIX_Q) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd
    offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)
    for start_n in range(0, N_PREFIX_Q + start_m):
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
        qk += tl.dot(q_vals, k_vals, allow_tf32=False)
        qk *= sm_scale
        qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float('-inf'))
        landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float('-inf')), 1)
        normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, float('-inf'), qk)
        normal_m = tl.max(normal_qk, 1)
        normal_p = tl.exp(normal_qk - normal_m[:, None])
        normal_denom = tl.sum(normal_p, 1)
        m_curr = tl.maximum(landmark_qk, m_prev)
        m_curr_ = m_curr
        l_prev *= tl.exp(m_prev - m_curr_)
        landmark_p = tl.exp(landmark_qk - m_curr_)
        l_curr = landmark_p + l_prev
        l_rcp = 1.0 / l_curr
        landmark_p *= l_rcp
        acc *= (l_prev * l_rcp)[:, None]
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
        acc += tl.dot(landmark_p[:, None] * normal_p / normal_denom[:, None], v_vals, allow_tf32=False)
        l_prev = l_curr
        m_prev = m_curr
        offs_n += BLOCK_N
        offs_k += BLOCK_N * skn
        offs_v += BLOCK_N * svn
    k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX_KV, other=0)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
    qk += tl.dot(q_vals, k_vals, allow_tf32=False)
    qk *= sm_scale
    qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float('-inf'))
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    m_curr_ = m_curr
    l_prev *= tl.exp(m_prev - m_curr_)
    p = tl.exp(qk - m_curr_[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = p
    v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX_KV, other=0)
    acc += tl.dot(p, v_vals, allow_tf32=False)
    l_prev = l_curr
    m_prev = m_curr
    offs_L = off_hz * N_CTX_Q + offs_m
    offs_M = off_hz * N_CTX_Q + offs_m
    tl.store(L + offs_L, l_prev, mask=offs_m < N_CTX_Q)
    tl.store(M + offs_M, m_prev, mask=offs_m < N_CTX_Q)
    offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _bwd_preprocess(Out, soz, soh, som, sod, DO, L, slzh, slm, NewDO, Delta, N_CTX_Q, BLOCK_M: 'tl.constexpr', D_HEAD: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    off_o = off_hz * soh + off_m[:, None] * som + off_d[None, :] * sod
    off_l = off_hz * slzh + off_m * slm
    o = tl.load(Out + off_o)
    do = tl.load(DO + off_o)
    denom = tl.load(L + off_l)
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    tl.store(NewDO + off_o, do)
    tl.store(Delta + off_l, delta)


@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, M, D, sqz, sqh, sqm, sqd, skz, skh, skn, skd, svz, svh, svn, svd, Z, H, N_CTX_Q, N_CTX_KV, BLOCK: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', N_PREFIX_Q: 'tl.constexpr'):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    BLOCK_M: 'tl.constexpr' = BLOCK
    BLOCK_N: 'tl.constexpr' = BLOCK
    Q += off_z * sqz + off_h * sqh
    K += off_z * skz + off_h * skh
    V += off_z * svz + off_h * svh
    DO += off_z * sqz + off_h * sqh
    DQ += off_z * sqz + off_h * sqh
    DK += off_z * skz + off_h * skh
    DV += off_z * svz + off_h * svh
    offs_d = tl.arange(0, BLOCK_DMODEL)
    D_ptrs = D + off_hz * N_CTX_Q
    m_ptrs = M + off_hz * N_CTX_Q
    for start_n in range(0, N_CTX_KV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        if start_n < N_PREFIX_Q * BLOCK_M:
            start_q_index = 0
        elif N_CTX_Q <= start_n - N_PREFIX_Q * BLOCK_M:
            start_q_index = start_n - N_PREFIX_Q * BLOCK_M
        else:
            first_start_m = start_n - N_PREFIX_Q * BLOCK_M
            first_start_m = tl.multiple_of(first_start_m, BLOCK_M)
            offs_m = first_start_m + tl.arange(0, BLOCK_M)
            offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M
            offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float('-inf'))
            m = tl.load(m_ptrs + offs_m)
            m_ = m
            last_p = tl.exp(qk * sm_scale - m_[:, None])
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(last_p), do, allow_tf32=False)
            Di = tl.load(D_ptrs + offs_m)
            last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            ds = last_p * last_dp * sm_scale
            dk += tl.dot(tl.trans(ds), q, allow_tf32=False)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds, k, allow_tf32=False)
            tl.store(dq_ptrs, dq)
            start_q_index = first_start_m + BLOCK_M
        for start_m in range(start_q_index, N_CTX_Q, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)
            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            dq_ptrs = DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            landmark_qk = tl.max(tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float('-inf')), 1)
            normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, float('-inf'), qk)
            m = tl.load(m_ptrs + offs_m)
            m_ = m
            p = tl.exp(landmark_qk - m_)
            do = tl.load(do_ptrs)
            normal_m = tl.max(normal_qk, 1)
            normal_p = tl.exp(normal_qk - normal_m[:, None])
            normal_p_normalized = normal_p / tl.sum(normal_p, 1)[:, None]
            normal_kv = tl.dot(normal_p_normalized, v, allow_tf32=False)
            normal_D = tl.sum(do * normal_kv, 1)
            dv += tl.dot(tl.trans(p[:, None] * normal_p_normalized), do, allow_tf32=False)
            Di = tl.load(D_ptrs + offs_m)
            dp = tl.zeros([BLOCK_M], dtype=tl.float32) - Di
            dp += normal_D
            landmark_ds = p * dp
            normal_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - normal_D[:, None]
            normal_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            normal_ds = p[:, None] * normal_p_normalized * normal_dp
            ds = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, landmark_ds[:, None], normal_ds)
            ds *= sm_scale
            dk += tl.dot(tl.trans(ds), q, allow_tf32=False)
            dq = tl.load(dq_ptrs)
            dq += tl.dot(ds, k, allow_tf32=False)
            tl.store(dq_ptrs, dq)
        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)

