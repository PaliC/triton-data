import sys
_module = sys.modules[__name__]
del sys
original_cogvideo = _module
sageattn_cogvideo = _module
sageattention = _module
attn_qk_int8_per_block_h64 = _module
attn_qk_int8_per_block_h64_bf16 = _module
attn_qk_int8_per_block_h96 = _module
attn_qk_int8_per_block_h96_bf16 = _module
attn_qk_int8_per_block_h96_causal = _module
attn_qk_int8_per_block_h96_causal_bf16 = _module
attn_qk_int8_per_block_hd128 = _module
attn_qk_int8_per_block_hd128_bf16 = _module
attn_qk_int8_per_block_hd128_causal = _module
attn_qk_int8_per_block_hd128_causal_bf16 = _module
attn_qk_int8_per_block_hd64_causal = _module
attn_qk_int8_per_block_hd64_causal_bf16 = _module
core = _module
quant_per_block = _module
quant_per_block_hd96 = _module
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


import torch


import math


import triton


import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M: 'tl.constexpr', HEAD_DIM: 'tl.constexpr', BLOCK_N: 'tl.constexpr', STAGE: 'tl.constexpr', offs_m: 'tl.constexpr', offs_n: 'tl.constexpr', N_CTX: 'tl.constexpr'):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += HEAD_DIM * lo
        V_ptrs += HEAD_DIM * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < N_CTX - start_n
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k) * q_scale * k_scale
        if STAGE == 2:
            mask = offs_m[:, None] >= start_n + offs_n[None, :]
            qk = qk + tl.where(mask, 0, -1000000.0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=offs_n[:, None] < N_CTX - start_n)
        p = p
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, HEAD_DIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', STAGE: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z * stride_qz + off_h * stride_qh
    vk_offset = qvk_offset // stride_qm
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=offs_m[:, None] < N_CTX)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX)
    acc, l_i, _ = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, N_CTX)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc, mask=offs_m[:, None] < N_CTX)


@triton.jit
def q_kernel_per_block_int8(X, X_int8, BLK: 'tl.constexpr', Scale, L, C: 'tl.constexpr', scale_stride):
    off_b = tl.program_id(1)
    off_blk = tl.program_id(0)
    x_offset = off_b * L * C
    offs_m = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, 128)
    x_ptrs = X + x_offset + offs_m[:, None] * C + offs_k[None, :]
    x_int8_ptrs = X_int8 + x_offset + offs_m[:, None] * C + offs_k[None, :]
    scale_ptrs = Scale + off_b * scale_stride + off_blk
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < L) & (tl.arange(0, 128) < 96)[None, :])
    x *= C ** -0.5 * 1.44269504
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8
    tl.store(x_int8_ptrs, x_int8, mask=(offs_m[:, None] < L) & (tl.arange(0, 128) < 96)[None, :])
    tl.store(scale_ptrs, scale)


@triton.jit
def k_kernel_per_block_int8(X, X_int8, BLK: 'tl.constexpr', Scale, L, C: 'tl.constexpr', scale_stride):
    off_b = tl.program_id(1)
    off_blk = tl.program_id(0)
    x_offset = off_b * L * C
    offs_m = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, 128)
    x_ptrs = X + x_offset + offs_m[:, None] * C + offs_k[None, :]
    x_int8_ptrs = X_int8 + x_offset + offs_m[:, None] * C + offs_k[None, :]
    scale_ptrs = Scale + off_b * scale_stride + off_blk
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < L) & (tl.arange(0, 128) < 96)[None, :])
    scale = tl.max(tl.abs(x)) / 127.0
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8
    tl.store(x_int8_ptrs, x_int8, mask=(offs_m[:, None] < L) & (tl.arange(0, 128) < 96)[None, :])
    tl.store(scale_ptrs, scale)

