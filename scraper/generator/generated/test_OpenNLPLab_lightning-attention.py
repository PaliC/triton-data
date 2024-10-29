import sys
_module = sys.modules[__name__]
del sys
benchmark_lightning2 = _module
benchmark_srmsnorm = _module
example_lightning_attn = _module
lightning_attn = _module
ops = _module
lightning_attn_interface = _module
linear_attention = _module
srmsnorm = _module
triton = _module
lightning_attn2 = _module
lightning_attn2_no_decay = _module
srmsnorm = _module
utils = _module
setup = _module
test_lightning2 = _module
test_lightning2_no_decay = _module
test_srmsnorm = _module

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


import numpy as np


import torch


import triton


import math


import torch.nn.functional as F


import torch.nn as nn


import triton.language as tl


@triton.jit
def _fwd_kernel(Q, K, V, Out, b: 'tl.constexpr', h: 'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', BLOCK_MODEL: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    e_offset = off_e * BLOCK_MODEL
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    off_block = tl.arange(0, BLOCK)
    index = off_block[:, None] - off_block[None, :]
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        q = tl.load(Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0)
        k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * d, mask=off_block[None, :] < n, other=0.0)
        v = tl.load(V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0)
        qk = tl.dot(q, k_trans)
        qk = tl.where(index >= 0, qk, 0)
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv)
        o = o_intra + o_inter
        tl.store(O_block_ptr + off_block[:, None] * e, o, mask=off_block[:, None] < n)
        kv += tl.dot(k_trans, v)
        off_block += BLOCK


@triton.jit
def _bwd_intra_kernel(Q, K, V, DO, DQ, DK, DV, b: 'tl.constexpr', h: 'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', CBLOCK: 'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)
    Q_trans_block_ptr = Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    array = tl.arange(0, BLOCK)
    index = array[:, None] - array[None, :]
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0)
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0)
    dqk = tl.dot(do, v_trans)
    dqk = tl.where(index >= 0, dqk, 0)
    dq_intra = tl.dot(dqk, k)
    dk_intra_trans = tl.dot(q_trans, dqk)
    qk_trans = tl.dot(k, q_trans)
    qk_trans = tl.where(index <= 0, qk_trans, 0)
    dv_intra = tl.dot(qk_trans, do)
    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra
    tl.store(DQ_block_ptr, dq, mask=block_offset[:, None] < n)
    tl.store(DK_trans_block_ptr, dk_trans, mask=block_offset[None, :] < n)
    tl.store(DV_block_ptr, dv, mask=block_offset[:, None] < n)


@triton.jit
def _bwd_inter_kernel(Q, K, V, DO, DQ, DK, DV, b: 'tl.constexpr', h: 'tl.constexpr', n: 'tl.constexpr', d: 'tl.constexpr', e: 'tl.constexpr', BLOCK: 'tl.constexpr', NUM_BLOCK: 'tl.constexpr', CBLOCK: 'tl.constexpr', NUM_CBLOCK: 'tl.constexpr'):
    off_bh = tl.program_id(0)
    off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    DQ_block_ptr = DQ + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    K_block_ptr = K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    DO_block_ptr = DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)
    kv_trans = tl.zeros([e, d], dtype=tl.float32)
    for i in range(NUM_BLOCK):
        for j in range(NUM_CBLOCK):
            if i > 0:
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n, other=0.0)
                dq_inter = tl.dot(do, kv_trans)
                dq = dq_inter + tl.load(DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0)
                tl.store(DQ_block_ptr, dq, mask=off_block1[:, None] < n)
            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0)
            kv_trans_current += tl.dot(v_trans, k)
            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK
        kv_trans += kv_trans_current
    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)
    Q_trans_block_ptr = Q + qk_offset + m * d + tl.arange(0, CBLOCK)[None, :] * d + tl.arange(0, d)[:, None]
    K_block_ptr = K + qk_offset + m * d + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = V + v_offset + m * e + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    DK_trans_block_ptr = DK + qk_offset + m * d + tl.arange(0, CBLOCK)[None, :] * d + tl.arange(0, d)[:, None]
    DV_block_ptr = DV + v_offset + m * e + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + m * e + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    dkv = tl.zeros([d, e], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1, -1, -1):
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK
            if i < NUM_BLOCK - 1:
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n, other=0.0)
                v_trans = tl.load(V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0)
                dk_inter_trans = tl.dot(dkv, v_trans)
                dv_inter = tl.dot(k, dkv)
                dk_trans = dk_inter_trans + tl.load(DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0)
                dv = dv_inter + tl.load(DV_block_ptr, mask=off_block1[:, None] < n, other=0.0)
                tl.store(DK_trans_block_ptr, dk_trans, mask=off_block1[None, :] < n)
                tl.store(DV_block_ptr, dv, mask=off_block1[:, None] < n)
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK
            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0)
            q_trans = tl.load(Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0)
            dkv_current += tl.dot(q_trans, do)
        dkv += dkv_current


@triton.jit
def srms_norm_fw(X, Y, V, stride, N, eps, BLOCK_SIZE_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_zm = tl.where(mask, x, 0.0)
    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)
    y = x_zm * rstd
    tl.store(V + row, rstd)
    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def srms_norm_bwd_dx_fused(DX, DY, X, V, stride, N, BLOCK_SIZE_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    rstd = tl.load(V + row)
    xhat = x * rstd
    wdy = dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - xhat * mean1) * rstd
    mask = cols < N
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

