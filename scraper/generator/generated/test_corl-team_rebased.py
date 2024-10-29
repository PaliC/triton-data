import sys
_module = sys.modules[__name__]
del sys
utils = _module
flash_linear_attention = _module
fla = _module
layers = _module
based = _module
gla = _module
multiscale_retention = _module
rebased = _module
rebased_fast = _module
modules = _module
convolution = _module
rmsnorm = _module
rotary = _module
ops = _module
based = _module
gla = _module
retention = _module
triton = _module
abc = _module
chunk_fuse = _module
chunk_fuse = _module
parallel = _module
block_parallel = _module
inter_chunk_contribution = _module
chunk_scan_triton_full = _module
chunk_scan_triton_no_decay = _module
chunk_scan_triton_only_gk = _module
chunk_scan_triton_only_gv = _module
fn = _module
preprocess_cumsum_gk = _module
preprocess_cumsum_gv = _module
intra_chunk_contribution = _module
fn_only_gk = _module
fn_only_gv = _module
chunk = _module
chunk_fuse = _module
recurrent_fuse = _module
parallel = _module
parallel = _module
chunk = _module
chunk_fuse = _module
parallel = _module
recurrent_fuse = _module
rotary = _module
setup = _module
src = _module
config_pyr = _module
data = _module
associative_recall = _module
logger = _module
mixers = _module
attention = _module
base_conv = _module
based = _module
h3 = _module
dplr = _module
hippo = _module
fftconv = _module
krylov = _module
toeplitz = _module
vandermonde = _module
ss_kernel = _module
ss_kernel_diag = _module
ss_kernel_shift = _module
ssm_utils = _module
hybrid = _module
hyena = _module
listing = _module
mamba = _module
mamba_ssm = _module
selective_scan_interface = _module
layernorm = _module
selective_state_update = _module
mlp = _module
rebased = _module
rwkv = _module
selective = _module
model = _module
train = _module

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


import torch


import torch.nn as nn


import warnings


import torch.nn.functional as F


from torch.cuda.amp import custom_fwd


from torch.cuda.amp import custom_bwd


import triton


import triton.language as tl


from typing import Optional


from typing import Tuple


from typing import Union


import re


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch import Tensor


from torch.nn import LayerNorm


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM', 'HAS_BIAS'])
@triton.jit
def _layer_norm_fwd_1pass_kernel(X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean, Rstd, stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, N, eps, IS_RMS_NORM: 'tl.constexpr', BLOCK_N: 'tl.constexpr', HAS_RESIDUAL: 'tl.constexpr', STORE_RESIDUAL_OUT: 'tl.constexpr', HAS_BIAS: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
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


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N', 'HAS_DRESIDUAL', 'STORE_DRESIDUAL', 'IS_RMS_NORM', 'HAS_BIAS'])
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['Y'] is not None})
@triton.jit
def _layer_norm_bwd_kernel(X, W, B, Y, DY, DX, DW, DB, DRESIDUAL, DRESIDUAL_IN, Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row, stride_dx_row, stride_dres_row, stride_dres_in_row, M, N, eps, rows_per_program, IS_RMS_NORM: 'tl.constexpr', BLOCK_N: 'tl.constexpr', HAS_DRESIDUAL: 'tl.constexpr', STORE_DRESIDUAL: 'tl.constexpr', HAS_BIAS: 'tl.constexpr', RECOMPUTE_OUTPUT: 'tl.constexpr'):
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
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0)
        dy = tl.load(DY + cols, mask=mask, other=0)
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
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


@triton.jit
def chunk_abc_fwd_kernel_s(q, k, s, rk, ck, pk, s_qk_h, s_qk_t, s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BM: 'tl.constexpr', DK: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'):
    i_m, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_s = tl.make_block_ptr(s + (i_k * n_bh + i_bh) * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_pk = tl.make_block_ptr(pk + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_hk = tl.zeros([BK, BM], dtype=tl.float32)
    for _ in range(NT):
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_rk = tl.load(p_rk, boundary_check=(0,))
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1))
        b_inter = tl.dot(b_q, b_hk, allow_tf32=False) * b_rk[None, :]
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_q, b_k, allow_tf32=False), 0), b_ck, allow_tf32=False)
        b_s = (b_inter + b_intra) * b_pk
        b_hk = b_hk * b_rk[None, :] + tl.dot(b_k, b_ck, allow_tf32=False)
        tl.store(p_s, b_s, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_s = tl.advance(p_s, (BT, 0))
        p_rk = tl.advance(p_rk, (DM,))
        p_ck = tl.advance(p_ck, (BT, 0))
        p_pk = tl.advance(p_pk, (BT, 0))


@triton.jit
def chunk_abc_fwd_kernel_o(p, v, o, rv, cv, pv, s_qk_h, s_qk_t, s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BM: 'tl.constexpr', BV: 'tl.constexpr', DM: 'tl.constexpr', DV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_v, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_m * n_bh + i_bh) * s_qk_h, (T, DV), (s_qk_t, s_qk_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, 0), (BM, BT), (0, 1))
    p_pv = tl.make_block_ptr(pv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_hv = tl.zeros([BM, BV], dtype=tl.float32)
    for _ in range(NT):
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_rv = tl.load(p_rv, boundary_check=(0,))
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        b_pv = tl.load(p_pv, boundary_check=(0, 1))
        b_p = b_p * b_pv
        b_inter = tl.dot(b_p * b_rv[None, :], b_hv, allow_tf32=False)
        b_intra = tl.where(m_s, tl.dot(b_p, b_cv, allow_tf32=False), 0)
        b_intra = tl.dot(b_intra, b_v, allow_tf32=False)
        b_o = b_inter + b_intra
        b_hv = b_hv * b_rv[:, None] + tl.dot(b_cv, b_v, allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_p = tl.advance(p_p, (BT, 0))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_rv = tl.advance(p_rv, (DM,))
        p_cv = tl.advance(p_cv, (0, BT))
        p_pv = tl.advance(p_pv, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dp(v, rv, cv, pv, do, dp, s_qk_h, s_qk_t, s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BV: 'tl.constexpr', BM: 'tl.constexpr', DV: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'):
    i_m, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (DV, T), (s_qk_d, s_qk_t), (i_v * BV, 0), (BV, BT), (0, 1))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_pv = tl.make_block_ptr(pv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_dp = tl.make_block_ptr(dp + (i_v * n_bh + i_bh) * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_hv = tl.zeros([BV, BM], dtype=tl.float32)
    for _ in range(NT):
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_rv = tl.load(p_rv, boundary_check=(0,))
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        b_pv = tl.load(p_pv, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_inter = tl.dot(b_do, b_hv, allow_tf32=False) * b_rv[None, :]
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_do, b_v, allow_tf32=False), 0), b_cv, allow_tf32=False)
        b_dp = (b_inter + b_intra) * b_pv
        b_hv = b_hv * b_rv[None, :] + tl.dot(b_v, b_cv, allow_tf32=False)
        tl.store(p_dp, b_dp, boundary_check=(0, 1))
        p_v = tl.advance(p_v, (0, BT))
        p_rv = tl.advance(p_rv, (DM,))
        p_cv = tl.advance(p_cv, (BT, 0))
        p_pv = tl.advance(p_pv, (BT, 0))
        p_do = tl.advance(p_do, (BT, 0))
        p_dp = tl.advance(p_dp, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dq(k, rk, ck, dq, ds, s_qk_h, s_qk_t, s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BK: 'tl.constexpr', BM: 'tl.constexpr', DK: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, 0), (BM, BT), (0, 1))
    p_dq = tl.make_block_ptr(dq + (i_m * n_bh + i_bh) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_hk = tl.zeros([BM, BK], dtype=tl.float32)
    for _ in range(NT):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_rk = tl.load(p_rk, boundary_check=(0,))
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_inter = tl.dot(b_ds * b_rk[None, :], b_hk, allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_ds, b_ck, allow_tf32=False), 0), b_k, allow_tf32=False)
        b_dq = b_inter + b_intra
        b_hk = b_hk * b_rk[:, None] + tl.dot(b_ck, b_k, allow_tf32=False)
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
        p_k = tl.advance(p_k, (BT, 0))
        p_rk = tl.advance(p_rk, (DM,))
        p_ck = tl.advance(p_ck, (0, BT))
        p_dq = tl.advance(p_dq, (BT, 0))
        p_ds = tl.advance(p_ds, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dk(q, k, rk, ck, ds, dk, dsk, s_qk_h, s_qk_t, s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BK: 'tl.constexpr', BM: 'tl.constexpr', DK: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), ((NT - 1) * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, (NT - 1) * BT), (BK, BT), (0, 1))
    p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_ck = tl.make_block_ptr(ck + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, (NT - 1) * BT), (BM, BT), (0, 1))
    p_dk = tl.make_block_ptr(dk + (i_m * n_bh + i_bh) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), ((NT - 1) * BT, i_k * BK), (BT, BK), (1, 0))
    p_dsk = tl.make_block_ptr(dsk + (i_k * n_bh + i_bh) * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s, m_t = o_i[:, None] <= o_i[None, :], o_i[:, None] >= o_i[None, :]
    b_dhk = tl.zeros([BM, BK], dtype=tl.float32)
    for i in range(NT):
        p_rk = tl.make_block_ptr(rk + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), ((NT - i) % NT * DM + i_m * BM,), (BM,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_rk = tl.load(p_rk, boundary_check=(0,))
        b_ck = tl.load(p_ck, boundary_check=(0, 1))
        b_ds = tl.load(p_ds, boundary_check=(0, 1))
        b_inter = tl.dot(b_ck * b_rk[None, :], b_dhk, allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_ck, b_ds, allow_tf32=False), 0.0), b_q, allow_tf32=False)
        b_dk = b_inter + b_intra
        b_inter = tl.dot(b_dhk, b_k, allow_tf32=False) * b_rk[:, None]
        b_intra = tl.dot(b_ds, tl.where(m_t, tl.dot(b_q, b_k, allow_tf32=False), 0.0), allow_tf32=False)
        b_dsk = b_ck * tl.trans(b_inter + b_intra)
        b_dhk = b_dhk * b_rk[:, None] + tl.dot(b_ds, b_q, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dsk, b_dsk, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (-BT, 0))
        p_k = tl.advance(p_k, (0, -BT))
        p_ck = tl.advance(p_ck, (-BT, 0))
        p_ds = tl.advance(p_ds, (0, -BT))
        p_dk = tl.advance(p_dk, (-BT, 0))
        p_dsk = tl.advance(p_dsk, (-BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_dv(do, v, rv, cv, p, dv, dsv, s_qk_h, s_qk_t, s_qk_d, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BV: 'tl.constexpr', BM: 'tl.constexpr', DV: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'):
    i_v, i_m, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    p_do = tl.make_block_ptr(do + i_bh * s_qk_h, (T, DV), (s_qk_t, s_qk_d), ((NT - 1) * BT, i_v * BV), (BT, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_qk_h, (DV, T), (s_qk_d, s_qk_t), (i_v * BV, (NT - 1) * BT), (BV, BT), (0, 1))
    p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_cv = tl.make_block_ptr(cv + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (DM, T), (s_sk_m, s_sk_t), (i_m * BM, (NT - 1) * BT), (BM, BT), (0, 1))
    p_dv = tl.make_block_ptr(dv + (i_m * n_bh + i_bh) * s_qk_h, (T, DV), (s_qk_t, s_qk_d), ((NT - 1) * BT, i_v * BV), (BT, BV), (1, 0))
    p_dsv = tl.make_block_ptr(dsv + (i_v * n_bh + i_bh) * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_s, m_t = o_i[:, None] <= o_i[None, :], o_i[:, None] >= o_i[None, :]
    b_dhv = tl.zeros([BM, BV], dtype=tl.float32)
    for i in range(NT):
        p_rv = tl.make_block_ptr(rv + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), ((NT - i) % NT * DM + i_m * BM,), (BM,), (0,))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_rv = tl.load(p_rv, boundary_check=(0,))
        b_cv = tl.load(p_cv, boundary_check=(0, 1))
        b_p = tl.load(p_p, boundary_check=(0, 1))
        b_inter = tl.dot(b_cv * b_rv[None, :], b_dhv, allow_tf32=False)
        b_intra = tl.dot(tl.where(m_s, tl.dot(b_cv, b_p, allow_tf32=False), 0.0), b_do, allow_tf32=False)
        b_dv = b_inter + b_intra
        b_inter = tl.dot(b_dhv, b_v, allow_tf32=False) * b_rv[:, None]
        b_intra = tl.dot(b_p, tl.where(m_t, tl.dot(b_do, b_v, allow_tf32=False), 0.0), allow_tf32=False)
        b_dsv = b_cv * tl.trans(b_inter + b_intra)
        b_dhv = b_dhv * b_rv[:, None] + tl.dot(b_p, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
        tl.store(p_dsv, b_dsv, boundary_check=(0, 1))
        p_do = tl.advance(p_do, (-BT, 0))
        p_v = tl.advance(p_v, (0, -BT))
        p_cv = tl.advance(p_cv, (-BT, 0))
        p_p = tl.advance(p_p, (0, -BT))
        p_dv = tl.advance(p_dv, (-BT, 0))
        p_dsv = tl.advance(p_dsv, (-BT, 0))


@triton.jit
def chunk_abc_fwd_kernel_cum(s, r, c, p, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BM: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), (i_m * BM,), (BM,), (0,))
    p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), (0, i_m * BM), (BT, BM), (1, 0))
    b_mp = tl.zeros([BM], dtype=tl.float32)
    b_zp = tl.zeros([BM], dtype=tl.float32)
    for i in range(NT):
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_m = tl.max(b_s, 0)
        if i == 0:
            b_r = tl.exp(-b_m)
        else:
            b_m = tl.maximum(b_mp, b_m)
            b_r = tl.exp(b_mp - b_m)
        b_c = tl.exp(b_s - b_m[None, :])
        b_z = tl.cumsum(b_c, 0) + (b_zp * b_r)[None, :]
        b_p = tl.exp(-tl.log(b_z))
        b_mp = b_m
        b_zp = tl.max(b_z, 0)
        tl.store(p_r, b_r, boundary_check=(0,))
        tl.store(p_c, b_c, boundary_check=(0, 1))
        tl.store(p_p, b_p, boundary_check=(0, 1))
        p_s = tl.advance(p_s, (BT, 0))
        p_r = tl.advance(p_r, (DM,))
        p_c = tl.advance(p_c, (BT, 0))
        p_p = tl.advance(p_p, (BT, 0))


@triton.jit
def chunk_abc_bwd_kernel_rcum(s, r, c, o, s_sk_h, s_sk_t, s_sk_m, T, BT: 'tl.constexpr', BM: 'tl.constexpr', DM: 'tl.constexpr', NT: 'tl.constexpr'):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_c = tl.make_block_ptr(c + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_sk_h, (T, DM), (s_sk_t, s_sk_m), ((NT - 1) * BT, i_m * BM), (BT, BM), (1, 0))
    o_i = tl.arange(0, BT)
    m_t = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BM], dtype=tl.float32)
    for i in range(NT):
        p_r = tl.make_block_ptr(r + i_bh * s_sk_t * NT, (NT * DM,), (s_sk_m,), ((NT - i) % NT * DM + i_m * BM,), (BM,), (0,))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_r = tl.load(p_r, boundary_check=(0,))
        b_c = tl.load(p_c, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))
        b_z = b_z * b_r
        b_o -= b_c * (b_z[None, :] + tl.dot(m_t, b_s, allow_tf32=False))
        b_z += tl.sum(b_s, 0)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_s = tl.advance(p_s, (-BT, 0))
        p_c = tl.advance(p_c, (-BT, 0))
        p_o = tl.advance(p_o, (-BT, 0))


@triton.jit
def fused_chunk_based_fwd_kernel(q, k, v, o, z, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h_0o = tl.zeros([BV], dtype=tl.float32)
    b_h_1o = tl.zeros([BK, BV], dtype=tl.float32)
    b_h_2o = tl.zeros([BK * BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_z = z + (i_bh + i_k * B * H) * T + tl.arange(0, BT)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)
    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_0o = 0
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k_2o = b_k[:, None, :] * b_k[None, :, :]
        b_k_2o = tl.reshape(b_k_2o, [BK * BK, BT])
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1)) * scale
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_z = tl.zeros([BT], dtype=tl.float32)
        b_o += b_h_0o
        b_z += k_0o
        b_o += tl.dot(b_q, b_h_1o, allow_tf32=False)
        b_z += tl.sum(b_q * k_1o, axis=1)
        b_q_2o = b_q[:, :, None] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BT, BK * BK])
        b_o += tl.dot(b_q_2o, b_h_2o, allow_tf32=False) * 0.5
        b_z += tl.sum(b_q_2o * k_2o, axis=1) * 0.5
        k_1o += tl.sum(b_k, axis=1)[None, :]
        k_2o += tl.sum(b_k_2o, axis=1)[None, :]
        k_0o += BT
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        tl.store(p_z, b_z, mask=i * BT + tl.arange(0, BT) < T)
        b_h_2o = b_h_2o + tl.dot(b_k_2o, b_v, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_k, b_v, allow_tf32=False)
        b_h_0o = b_h_0o + tl.sum(b_v, axis=0)
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_z += BT


@triton.jit
def fused_chunk_based_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h_1o = tl.zeros([BV, BK], dtype=tl.float32)
    b_h_2o = tl.zeros([BV, BK * BK], dtype=tl.float32)
    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dz = dz + i_bh * T + tl.arange(0, BT) + i * BT
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dz = tl.load(p_dz, mask=tl.arange(0, BT) + i * BT < T)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dq += tl.dot(b_do, b_h_1o, allow_tf32=False)
        if i_v == 0:
            b_dq += b_dz[:, None] * k_1o
        b_dq_2o = tl.dot(b_do, b_h_2o, allow_tf32=False) * 0.5
        if i_v == 0:
            b_dq_2o += b_dz[:, None] * k_2o * 0.5
        b_dq_2o = tl.reshape(b_dq_2o, [BT, BK, BK])
        b_dq += tl.sum(b_dq_2o * b_q[:, :, None], axis=1)
        b_dq += tl.sum(b_dq_2o * b_q[:, None, :], axis=2)
        b_dq *= scale
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot(b_ds * (1 + b_s), b_k, allow_tf32=False)
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK])
        b_h_2o = b_h_2o + tl.dot(b_v, b_k_2o, allow_tf32=False)
        b_h_1o = b_h_1o + tl.dot(b_v, b_k, allow_tf32=False)
        if i_v == 0:
            k_1o += tl.sum(b_k, axis=0)[None, :]
            k_2o += tl.sum(b_k_2o, axis=0)[None, :]
    tl.debug_barrier()
    b_h_1o = None
    b_h_2o = None
    b_dh_1o = tl.zeros([BK, BV], dtype=tl.float32)
    b_dh_2o = tl.zeros([BK * BK, BV], dtype=tl.float32)
    b_dh_0o = tl.zeros([BV], dtype=tl.float32)
    m_s = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]
    dq_1o = tl.zeros([1, BK], dtype=tl.float32)
    dq_2o = tl.zeros([BK * BK, 1], dtype=tl.float32)
    for i in range(tl.cdiv(T, BT) * BT - BT, -BT, -BT):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_dz = dz + i_bh * T + tl.arange(0, BT) + i
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        b_dz = tl.load(p_dz, mask=tl.arange(0, BT) + i < T)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_q = b_q * scale
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds *= 1 + b_s
        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s2, b_do, allow_tf32=False)
        b_k_2o = b_k[:, :, None] * b_k[:, None, :]
        b_k_2o = tl.reshape(b_k_2o, [BT, BK * BK])
        b_dv += tl.dot(b_k, b_dh_1o, allow_tf32=False)
        b_dv += tl.dot(b_k_2o, b_dh_2o, allow_tf32=False)
        b_dv += b_dh_0o
        b_dk += tl.dot(b_v, tl.trans(b_dh_1o), allow_tf32=False)
        if i_v == 0:
            b_dk += dq_1o
        b_dk_2o = tl.dot(b_dh_2o, tl.trans(b_v), allow_tf32=False)
        if i_v == 0:
            b_dk_2o += dq_2o
        b_dk_2o = tl.reshape(b_dk_2o, [BK, BK, BT])
        b_k_fp32 = tl.trans(b_k)
        b_dk2 = tl.sum(b_dk_2o * b_k_fp32[:, None, :], axis=0)
        b_dk2 += tl.sum(b_dk_2o * b_k_fp32[None, :, :], axis=1)
        b_dk += tl.trans(b_dk2)
        b_dh_0o += tl.sum(b_do, axis=0)
        b_dh_1o = b_dh_1o + tl.dot(b_q, b_do, allow_tf32=False)
        b_q_2o = b_q[None, :, :] * b_q[:, None, :]
        b_q_2o = tl.reshape(b_q_2o, [BK * BK, BT])
        b_dh_2o = b_dh_2o + tl.dot(b_q_2o, b_do, allow_tf32=False) * 0.5
        if i_v == 0:
            dq_1o += tl.sum(b_dz[None, :] * b_q, axis=1)[None, :]
            dq_2o += (tl.sum(b_dz[None, :] * b_q_2o, axis=1) * 0.5)[:, None]
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.jit
def parallel_based_fwd_kernel(q, k, v, o, z, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_z += tl.sum(b_s, axis=1)
        b_o = b_o + tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    tl.store(p_z, b_z, mask=i_c * BTL + tl.arange(0, BTL) < T)


@triton.jit
def _parallel_based_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q = b_q * scale
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=i_c * BTL + tl.arange(0, BTL) < T)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_dq += tl.dot(b_ds * (1 + b_s), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot(b_ds + b_ds * b_s, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    return


@triton.jit
def _parallel_based_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32)
    for i in range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=i + tl.arange(0, BTS) < T)
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_dv += tl.dot(b_s2, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_dk += tl.dot(b_ds + b_ds * b_s, tl.trans(b_q), allow_tf32=False)
    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=i + tl.arange(0, BTS) < T)
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = 1 + b_s + 0.5 * b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_dv += tl.dot(b_s2, tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot(b_ds + b_ds * b_s, tl.trans(b_q), allow_tf32=False)
        o_q += BTS
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    return


@triton.jit
def parallel_based_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_based_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL=BTL, BTS=BTS, BK=BK, BV=BV, DK=DK, DV=DV)
    tl.debug_barrier()
    _parallel_based_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL, BTS, BK, BV, DK, DV)


@triton.jit
def _fwd_recurrence(S, p2, O, NUM_BLOCK, D_MODEL_K: 'tl.constexpr', D_MODEL_V: 'tl.constexpr', BLOCK_MODEL: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]
    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] + D_MODEL_K * D_MODEL_V
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + D_MODEL_V
    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    acc += tl.load(S)
    S += D_MODEL_K * D_MODEL_V
    tl.store(O, acc)
    O += D_MODEL_K * D_MODEL_V
    for i in range(NUM_BLOCK - 2):
        p_v = tl.load(p2)
        S_i = tl.load(S)
        acc = acc * p_v[None, :] + S_i
        tl.store(O, acc)
        p2 += D_MODEL_V
        S += D_MODEL_K * D_MODEL_V
        O += D_MODEL_K * D_MODEL_V


@triton.jit
def _bwd_recurrence(S, p2, DS, Dp2, NUM_BLOCK, NUM_SPLIT_K, NUM_SPLIT_V, D_MODEL_K: 'tl.constexpr', D_MODEL_V: 'tl.constexpr', BLOCK_MODEL: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V
    Dp2 = Dp2 + offset_bh * NUM_BLOCK * D_MODEL_V * NUM_SPLIT_K + offset_d * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V * NUM_SPLIT_K
    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    for i in range(NUM_BLOCK - 1):
        p_value = tl.load(p2)
        S_i = tl.load(S)
        DS_i = tl.load(DS)
        Dacc += DS_i
        dp_i = Dacc * S_i
        dp_value = tl.sum(dp_i, axis=0)
        tl.store(Dp2, dp_value)
        tl.store(S, Dacc)
        Dacc *= p_value[None, :]
        S -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V
        p2 -= D_MODEL_V
        Dp2 -= D_MODEL_V * NUM_SPLIT_K


@triton.jit
def _fwd_preprocess_cumsum_gk(Q, K, GK, GK_cumsum, Q_exp, K_reduce, GK_last_exp, NUM_CHUNK, L, D_MODEL_K: 'tl.constexpr', D_BLOCK_K: 'tl.constexpr', CHUNK_SIZE: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    offset_nk = tl.program_id(2)
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    Q_exp_ptr = Q_exp + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    GK_last_exp_ptr = GK_last_exp + offset_bh * NUM_CHUNK * D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    cumsum = tl.zeros([D_BLOCK_K], dtype=tl.float32)
    mask = D_BLOCK_K * offset_nk + tl.arange(0, D_BLOCK_K) < D_MODEL_K
    for _ in range(CHUNK_SIZE):
        gk = tl.load(GK_ptr, mask=mask, other=0)
        cumsum += gk
        tl.store(GK_cumsum_ptr, cumsum, mask=mask)
        cumsum_exp = tl.exp(cumsum)
        q = tl.load(Q_ptr, mask=mask, other=0)
        q_exp = q * cumsum_exp
        tl.store(Q_exp_ptr, q_exp, mask=mask)
        Q_ptr += D_MODEL_K
        Q_exp_ptr += D_MODEL_K
        GK_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K
    tl.store(GK_last_exp_ptr, tl.exp(cumsum), mask=mask)
    tl.debug_barrier()
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    K_reduce_ptr = K_reduce + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    for _ in range(CHUNK_SIZE):
        gk_cumsum = tl.load(GK_cumsum_ptr, mask=mask, other=0)
        k = tl.load(K_ptr, mask=mask, other=0)
        k_reduce = k * tl.exp(cumsum - gk_cumsum)
        tl.store(K_reduce_ptr, k_reduce, mask=mask)
        K_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K
        K_reduce_ptr += D_MODEL_K


@triton.jit
def _bwd_preprocess_cumsum_gk(Q, K, GK, GK_cumsum, DQ_exp, DK_reduce, DGK_last_exp, DGK_cumsum, DQ, DK, DGK, NUM_CHUNK, L, D_MODEL_K: 'tl.constexpr', D_BLOCK_K: 'tl.constexpr', CHUNK_SIZE: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    offset_nk = tl.program_id(2)
    mask = D_BLOCK_K * offset_nk + tl.arange(0, D_BLOCK_K) < D_MODEL_K
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    DQ_ptr = DQ + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    DK_ptr = DK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    DQ_exp_ptr = DQ_exp + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    DK_reduce_ptr = DK_reduce + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    DGK_cumsum_ptr = DGK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    DGK_ptr = DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    D_GK_last_exp_ptr = DGK_last_exp + offset_bh * NUM_CHUNK * D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_BLOCK_K) + D_BLOCK_K * offset_nk
    cumsum_gradient = tl.zeros([D_BLOCK_K], dtype=tl.float32)
    grad_gk_last = tl.zeros([D_BLOCK_K], dtype=tl.float32)
    gk_last = tl.load(GK_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_K, mask=mask, other=0)
    cumsum_gradient += tl.load(D_GK_last_exp_ptr, mask=mask, other=0) * tl.exp(gk_last)
    GK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    GK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    Q_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    K_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DQ_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DQ_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        gk_cs = tl.load(GK_cumsum_ptr, mask=mask, other=0)
        k = tl.load(K_ptr, mask=mask, other=0)
        grad_k = tl.exp(gk_last - gk_cs) * tl.load(DK_reduce_ptr, mask=mask, other=0)
        tl.store(DK_ptr, grad_k, mask=mask)
        grad_k *= k
        cumsum_gradient -= grad_k
        grad_gk_last += grad_k
        q = tl.load(Q_ptr, mask=mask, other=0)
        grad_q = tl.exp(gk_cs) * tl.load(DQ_exp_ptr, mask=mask, other=0)
        tl.store(DQ_ptr, grad_q, mask=mask)
        cumsum_gradient += grad_q * q
        cumsum_gradient += tl.load(DGK_cumsum_ptr, mask=mask, other=0)
        tl.store(DGK_ptr, cumsum_gradient, mask=mask)
        Q_ptr -= D_MODEL_K
        DQ_exp_ptr -= D_MODEL_K
        K_ptr -= D_MODEL_K
        DK_reduce_ptr -= D_MODEL_K
        GK_cumsum_ptr -= D_MODEL_K
        DGK_cumsum_ptr -= D_MODEL_K
        DQ_ptr -= D_MODEL_K
        DK_ptr -= D_MODEL_K
        DGK_ptr -= D_MODEL_K
    DGK_ptr = DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + (CHUNK_SIZE - 1) * D_MODEL_K + D_BLOCK_K * offset_nk
    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_BLOCK_K) + (CHUNK_SIZE - 1) * D_MODEL_K + D_BLOCK_K * offset_nk
    grad_gk_last = grad_gk_last + 0.0
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        dgk = tl.load(DGK_ptr, mask=mask, other=0)
        dgk += grad_gk_last
        tl.store(DGK_ptr, dgk, mask=mask)
        DGK_ptr -= D_MODEL_K
        GK_ptr -= D_MODEL_K


@triton.jit
def _fwd_preprocess_cumsum_gv(V, GV, GV_cumsum, GV_exp, V_reduce, GV_last_exp, NUM_CHUNK, L, D_MODEL_V: 'tl.constexpr', CHUNK_SIZE: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_last_exp_ptr = GV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_exp_ptr = GV_exp + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    cumsum = tl.zeros([D_MODEL_V], dtype=tl.float32)
    for _ in range(CHUNK_SIZE):
        gv = tl.load(GV_ptr)
        cumsum += gv
        tl.store(GV_cumsum_ptr, cumsum)
        tl.store(GV_exp_ptr, tl.exp(cumsum))
        GV_cumsum_ptr += D_MODEL_V
        GV_exp_ptr += D_MODEL_V
        GV_ptr += D_MODEL_V
    tl.store(GV_last_exp_ptr, tl.exp(cumsum))
    tl.debug_barrier()
    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    V_reduce_ptr = V_reduce + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    for _ in range(CHUNK_SIZE):
        v = tl.load(V_ptr)
        gv = tl.load(GV_cumsum_ptr)
        v_reduce = v * tl.exp(cumsum - gv)
        tl.store(V_reduce_ptr, v_reduce)
        V_ptr += D_MODEL_V
        V_reduce_ptr += D_MODEL_V
        GV_cumsum_ptr += D_MODEL_V


@triton.jit
def _bwd_preprocess_cumsum_gv(V, GV, GV_cumsum, DGV_cumsum_exp, DV_reduce, DGV_last_exp, DGV_cumsum, DV, DGV, NUM_CHUNK, L, D_MODEL_V: 'tl.constexpr', CHUNK_SIZE: 'tl.constexpr'):
    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DV_ptr = DV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DV_reduce_ptr = DV_reduce + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_ptr = DGV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_exp_ptr = DGV_cumsum_exp + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_ptr = DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    D_GV_last_exp_ptr = DGV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V)
    cumsum_gradient = tl.zeros([D_MODEL_V], dtype=tl.float32)
    grad_gv_last = tl.zeros([D_MODEL_V], dtype=tl.float32)
    gv_last = tl.load(GV_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_V)
    cumsum_gradient += tl.load(D_GV_last_exp_ptr) * tl.exp(gv_last)
    GV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    GV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    V_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DV_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        gv_cs = tl.load(GV_cumsum_ptr)
        v = tl.load(V_ptr)
        grad_v = tl.exp(gv_last - gv_cs) * tl.load(DV_reduce_ptr)
        tl.store(DV_ptr, grad_v)
        grad_v *= v
        cumsum_gradient -= grad_v
        grad_gv_last += grad_v
        grad_v = tl.exp(gv_cs) * tl.load(DGV_cumsum_exp_ptr)
        cumsum_gradient += grad_v
        cumsum_gradient += tl.load(DGV_cumsum_ptr)
        tl.store(DGV_ptr, cumsum_gradient)
        V_ptr -= D_MODEL_V
        DV_reduce_ptr -= D_MODEL_V
        GV_cumsum_ptr -= D_MODEL_V
        DGV_cumsum_ptr -= D_MODEL_V
        DV_ptr -= D_MODEL_V
        DGV_ptr -= D_MODEL_V
        DGV_cumsum_exp_ptr -= D_MODEL_V
    DGV_ptr = DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V
    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V
    grad_gv_last = grad_gv_last + 0.0
    for idx in range(CHUNK_SIZE - 1, -1, -1):
        dgv = tl.load(DGV_ptr)
        dgv += grad_gv_last
        tl.store(DGV_ptr, dgv)
        DGV_ptr -= D_MODEL_V
        GV_ptr -= D_MODEL_V


@triton.jit
def _fwd_kernel_compute_A(Q, K, GK, A, stride_q1, stride_q2, stride_q3, stride_q4, stride_a1, stride_a2, stride_a3, stride_a4, Z, H, N_CTX, D, BLOCK_DMODEL_QK: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_k = tl.program_id(2)
    qk_offset = off_hz * stride_q2 + off_k * BLOCK_DMODEL_QK
    a_offset = (off_k * Z * H + off_hz) * stride_a2
    lo = 0
    hi = BLOCK_N
    Q_ptr = Q + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    K_ptr = K + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4
    GK_K_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[:, None] + tl.arange(0, 16)[None, :] * stride_q4
    GK_Q_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
            k_gk = tl.exp(q_normalizer[:, None] - k_gk)
            k = k * k_gk
            qk = tl.dot(q, k, allow_tf32=False)
            tl.store(A_ptr + q_high * stride_a4 + k_high, qk)
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)
        k = tl.load(K_ptr + q_high * stride_q4)
        k = k * tl.trans(q_gk3)
        qk = tl.dot(q, k, allow_tf32=False)
        qk = tl.where(tl.arange(0, 16)[:, None] >= tl.arange(0, 16)[None, :], qk, 0.0)
        tl.store(A_ptr + q_high * stride_a4 + q_high, qk)


@triton.jit
def _bwd_kernel_dqk(Q, K, GK, DA, DQ, DK, DGK, stride_q1, stride_q2, stride_q3, stride_q4, stride_a1, stride_a2, stride_a3, stride_a4, Z, H, N_CTX, D, BLOCK_DMODEL_QK: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_k = tl.program_id(2)
    qk_offset = off_hz * stride_q2 + BLOCK_DMODEL_QK * off_k
    a_offset = off_hz * stride_a2
    lo = 0
    hi = BLOCK_N
    Q_ptr = Q + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DQ_ptr = DQ + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    K_ptr = K + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    GK_K_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    GK_Q_ptr = GK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DA_ptr = DA + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(lo + 16, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        dq2 = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        for k_high in range(0, q_high, 16):
            k = tl.load(K_ptr + k_high * stride_q4)
            k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)
            k_gk = tl.exp(q_normalizer[None, :] - k_gk)
            k = k * k_gk
            dq2 += tl.dot(dqk, k, allow_tf32=False)
        dq2 = dq2
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_gk = tl.exp(q_gk - q_normalizer[None, :])
        dq = dq2 * q_gk
        dq_gk = dq * q
        DQ_ptr = DQ + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + q_high * stride_q4
        tl.store(DQ_ptr, dq)
        DGK_Q_ptr = DGK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + q_high * stride_q4
        tl.store(DGK_Q_ptr, dq_gk)
    tl.debug_barrier()
    for k_high in range(lo, hi - 16, 16):
        k = tl.load(K_ptr + k_high * stride_q4)
        k_gk = tl.load(GK_K_ptr + k_high * stride_q4)
        dk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        dgk = tl.zeros([16, BLOCK_DMODEL_QK], dtype=tl.float32)
        for q_high in range(k_high + 16, hi, 16):
            q = tl.load(Q_ptr + q_high * stride_q4)
            q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
            q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
            q_gk = tl.exp(q_gk - q_normalizer[None, :])
            q = q * q_gk
            dqk = tl.load(DA_ptr + q_high * stride_a4 + k_high)
            k_gk2 = tl.exp(q_normalizer[None, :] - k_gk)
            dk2 = tl.dot(tl.trans(dqk), q, allow_tf32=False)
            dk += dk2 * k_gk2
            dgk -= dk2 * k * k_gk2
        DK_ptr = DK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + k_high * stride_q4
        tl.store(DK_ptr, dk)
        DGK_K_ptr = DGK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4 + k_high * stride_q4
        prev = tl.load(DGK_K_ptr)
        tl.store(DGK_K_ptr, prev + dgk)
    tl.debug_barrier()
    DK_ptr = DK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DGK_K_ptr = DGK + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    DQ_ptr = DQ + qk_offset + start_m * stride_q3 + tl.arange(0, BLOCK_DMODEL_QK)[None, :] + tl.arange(0, 16)[:, None] * stride_q4
    for q_high in range(lo, hi, 16):
        q = tl.load(Q_ptr + q_high * stride_q4)
        q_gk = tl.load(GK_Q_ptr + q_high * stride_q4)
        q_normalizer = tl.load(GK + qk_offset + start_m * stride_q3 + q_high * stride_q4 + tl.arange(0, BLOCK_DMODEL_QK))
        q_gk2 = tl.exp(q_gk - q_normalizer[None, :])
        q2 = q * q_gk2
        q_gk3 = tl.exp(q_normalizer[None, :] - q_gk)
        k = tl.load(K_ptr + q_high * stride_q4)
        k2 = k * q_gk3
        dqk = tl.load(DA_ptr + q_high * stride_a4 + q_high)
        dqk = tl.where(tl.arange(0, 16)[:, None] >= tl.arange(0, 16)[None, :], dqk, 0.0)
        dk2 = tl.dot(tl.trans(dqk), q2, allow_tf32=False)
        dk = dk2 * q_gk3
        prev_dk = tl.load(DK_ptr + q_high * stride_q4)
        tl.store(DK_ptr + q_high * stride_q4, dk + prev_dk)
        dgk = -dk * k
        dq2 = tl.dot(dqk, k2, allow_tf32=False)
        dq = dq2 * q_gk2
        prev_dq = tl.load(DQ_ptr + q_high * stride_q4)
        tl.store(DQ_ptr + q_high * stride_q4, dq + prev_dq)
        dgk += dq * q
        prev_dq_gk = tl.load(DGK_K_ptr + q_high * stride_q4)
        tl.store(DGK_K_ptr + q_high * stride_q4, dgk + prev_dq_gk)


@triton.jit
def _fwd_compute_O(A, V, GV, O, stride_a2, stride_a3, stride_a4, stride_v2, stride_v3, stride_v4, BLOCK_N: 'tl.constexpr', BLOCK_DMODEL_V: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_v = tl.program_id(2)
    a_offset = off_hz * stride_a2
    v_offset = off_hz * stride_v2 + off_v * BLOCK_DMODEL_V
    lo = 0
    hi = BLOCK_N
    V_ptr = V + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    O_ptr = O + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(lo + 16, hi, 16):
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        acc = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)
        for k_high in range(0, q_high, 16):
            qk = tl.load(A_ptr + q_high * stride_a4 + k_high)
            v = tl.load(V_ptr + k_high * stride_v4)
            k_gv = tl.load(GV_ptr + k_high * stride_v4)
            k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
            v = v * k_gv
            output = tl.dot(qk, v, allow_tf32=False)
            acc += output
        tl.store(O_ptr + q_high * stride_v4, acc)
    tl.store(O_ptr, tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32))
    tl.debug_barrier()
    for q_high in range(lo, hi, 16):
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        qk = tl.load(A_ptr + q_high * stride_a4 + q_high)
        v = tl.load(V_ptr + q_high * stride_v4)
        k_gv = tl.load(GV_ptr + q_high * stride_v4)
        k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv)
        v = v * k_gv2
        output = tl.dot(qk, v, allow_tf32=False)
        q_gv = tl.exp(k_gv - q_gv_normalizer[None, :])
        prev = tl.load(O_ptr + q_high * stride_v4)
        output += prev
        output = output * q_gv
        tl.store(O_ptr + q_high * stride_v4, output)


@triton.jit
def _bwd_kernel_dav(V, GV, A, O, DO, DA, DV, DGV, Z, H, stride_a1, stride_a2, stride_a3, stride_a4, stride_v1, stride_v2, stride_v3, stride_v4, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_DMODEL_V: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_v = tl.program_id(2)
    a_offset = off_hz * stride_a2
    da_offset = (off_v * Z * H + off_hz) * stride_a2
    v_offset = off_hz * stride_v2 + off_v * BLOCK_DMODEL_V
    lo = 0
    hi = BLOCK_N
    DO_ptr = DO + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    O_ptr = O + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    DV_ptr = DV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    DGV_ptr = DGV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4
    DA_ptr = DA + da_offset + start_m * stride_a3 + tl.arange(0, 16)[None, :] + tl.arange(0, 16)[:, None] * stride_a4
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        o = tl.load(O_ptr + q_high * stride_v4)
        tl.store(DGV_ptr + q_high * stride_v4, do * o)
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        q_gv = tl.load(GV_ptr + q_high * stride_v4)
        q_gv = tl.exp(q_gv - q_gv_normalizer[None, :])
        do = do * q_gv
        tl.store(DO_ptr + q_high * stride_v4, do)
    tl.debug_barrier()
    V_ptr = V + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[:, None] + tl.arange(0, 16)[None, :] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[:, None] + tl.arange(0, 16)[None, :] * stride_v4
    for q_high in range(lo + 16, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        for k_high in range(0, q_high, 16):
            v = tl.load(V_ptr + k_high * stride_v4)
            k_gv = tl.load(GV_ptr + k_high * stride_v4)
            k_gv = tl.exp(q_gv_normalizer[:, None] - k_gv)
            v2 = v * k_gv
            dqk = tl.dot(do, v2, allow_tf32=False)
            tl.store(DA_ptr + q_high * stride_a4 + k_high, dqk)
    tl.debug_barrier()
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[:, None] + tl.arange(0, 16)[None, :] * stride_a4
    V_ptr = V + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    GV_ptr = GV + v_offset + start_m * stride_v3 + tl.arange(0, BLOCK_DMODEL_V)[None, :] + tl.arange(0, 16)[:, None] * stride_v4
    for k_high in range(0, hi, 16):
        dv = tl.zeros([16, BLOCK_DMODEL_V], dtype=tl.float32)
        k_gv = tl.load(GV_ptr + k_high * stride_v4)
        for q_high in range(k_high + 16, BLOCK_N, 16):
            do = tl.load(DO_ptr + q_high * stride_v4)
            kq = tl.load(A_ptr + q_high * stride_a4 + k_high)
            q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
            k_gv2 = tl.exp(q_gv_normalizer[None, :] - k_gv)
            dv2 = tl.dot(kq, do, allow_tf32=False)
            dv += dv2 * k_gv2
        v = tl.load(V_ptr + k_high * stride_v4)
        tl.store(DV_ptr + k_high * stride_v4, dv)
        prev_dv = tl.load(DGV_ptr + k_high * stride_v4)
        tl.store(DGV_ptr + k_high * stride_v4, prev_dv - dv * v)
    tl.debug_barrier()
    A_ptr = A + a_offset + start_m * stride_a3 + tl.arange(0, 16)[:, None] + tl.arange(0, 16)[None, :] * stride_a4
    for q_high in range(lo, hi, 16):
        do = tl.load(DO_ptr + q_high * stride_v4)
        q_gv_normalizer = tl.load(GV + v_offset + start_m * stride_v3 + q_high * stride_v4 + tl.arange(0, BLOCK_DMODEL_V))
        v = tl.load(V_ptr + q_high * stride_v4)
        k_gv = tl.load(GV_ptr + q_high * stride_v4)
        k_gv = tl.exp(q_gv_normalizer[None, :] - k_gv)
        v2 = v * k_gv
        dqk = tl.dot(do, tl.trans(v2), allow_tf32=False)
        dqk = tl.where(tl.arange(0, 16)[:, None] >= tl.arange(0, 16)[None, :], dqk, 0.0)
        tl.store(DA_ptr + q_high * stride_a4 + q_high, dqk)
        kq = tl.load(A_ptr + q_high * stride_a4 + q_high)
        dv2 = tl.dot(kq, do, allow_tf32=False)
        dv = dv2 * k_gv
        prev_dv = tl.load(DV_ptr + q_high * stride_v4)
        tl.store(DV_ptr + q_high * stride_v4, prev_dv + dv)
        prev_gdv = tl.load(DGV_ptr + q_high * stride_v4)
        prev_gdv -= dv * v
        tl.store(DGV_ptr + q_high * stride_v4, prev_gdv)


inv_ln2 = 1.44269504


@triton.jit
def fused_chunk_gla_fwd_kernel(q, k, v, g, o, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_g *= inv_ln2
        d_b = tl.load(p_db) * inv_ln2
        b_q = b_q * scale * tl.math.exp2(b_g)
        b_k = b_k * tl.trans(tl.math.exp2(-b_g + d_b[None, :]))
        b_o = tl.dot(b_q, b_h, allow_tf32=False)
        b_h *= tl.math.exp2(d_b)[:, None]
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_g = tl.advance(p_g, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_db += BT * DK
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h, boundary_check=(0, 1))


@triton.jit
def fused_chunk_gla_bwd_kernel(q, k, v, g, do, dq, dk, dv, initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + ((i + 1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)) * inv_ln2
        d_b = tl.load(p_db) * inv_ln2
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_k *= tl.math.exp2(d_b[None, :] - b_g)
        b_h *= tl.math.exp2(d_b)[None, :]
        b_h += tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale * tl.math.exp2(b_g)
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + (T - (i - 1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)) * inv_ln2
        b_db = tl.load(p_db) * inv_ln2
        g_k = tl.math.exp2(b_db[None, :] - b_g)
        b_k *= g_k
        b_q *= tl.math.exp2(tl.trans(b_g))
        b_dk = tl.trans(tl.dot(b_dh, tl.trans(b_v), allow_tf32=False)) * scale * g_k
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * scale
        b_dh *= tl.math.exp2(b_db)[:, None]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.jit
def fused_recurrent_gla_fwd_kernel(q, k, v, gk, gv, o, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    h = tl.zeros([BV, BK], dtype=tl.float32)
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0)
    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _q = tl.load(p_q, mask=mask_bk, other=0) * scale
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0)
            h = h * _gk[None, :]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0)
            h = h * _gv[:, None]
        h += _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o, mask=mask_bv)
        p_q += -DK if REVERSE else DK
        p_k += -DK if REVERSE else DK
        p_o += -DV if REVERSE else DV
        p_v += -DV if REVERSE else DV
        if USE_GK:
            p_gk += -DK if REVERSE else DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV
    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h, mask=mask_kv)


@triton.jit
def fused_recurrent_gla_bwd_kernel(q, k, v, gk, gv, do, dq, dk, dv, initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if REVERSE else 0)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[:, None]) * DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0)
    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _do = tl.load(p_do, mask=mask_bv, other=0)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0)
            h = h * _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0)
            h = h * _gv[None, :]
        h += _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_bk)
        p_k += -DK if REVERSE else DK
        p_v += -DV if REVERSE else DV
        p_q += -DK if REVERSE else DK
        p_do += -DV if REVERSE else DV
        p_dq += -DK if REVERSE else DK
        if USE_GK:
            p_gk += -DK if REVERSE else DK
        if USE_GV:
            p_gv += -DV if REVERSE else DV
    tl.debug_barrier()
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * DK if not REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * DV if not REVERSE else 0)
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0)
        _q = tl.load(p_q, mask=mask_bk, other=0) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)
        if USE_GK:
            _gk = tl.load(p_gk, mask=mask_bk, other=0)
            d_h *= _gk[:, None]
        if USE_GV:
            _gv = tl.load(p_gv, mask=mask_bv, other=0)
            d_h *= _gv[None, :]
        tl.store(p_dk, d_k, mask=mask_bk)
        tl.store(p_dv, d_v, mask=mask_bv)
        p_do += DV if REVERSE else -DV
        p_q += DK if REVERSE else -DK
        p_k += DK if REVERSE else -DK
        p_v += DV if REVERSE else -DV
        p_dk += DK if REVERSE else -DK
        p_dv += DV if REVERSE else -DV
        if USE_GK:
            p_gk += DK if REVERSE else -DK
        if USE_GV:
            p_gv += DV if REVERSE else -DV


@triton.jit
def parallel_rebased_fwd_kernel(q, k, v, o, z, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)
        b_o = b_o + tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    tl.store(p_z, b_z, mask=i_c * BTL + tl.arange(0, BTL) < T)


@triton.jit
def _parallel_rebased_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q = b_q * scale
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=i_c * BTL + tl.arange(0, BTL) < T)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_dq += tl.dot(2 * b_ds * b_s, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dq += tl.dot(2 * b_ds * b_s, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    return


@triton.jit
def _parallel_rebased_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32)
    for i in range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=i + tl.arange(0, BTS) < T)
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_dk += tl.dot(2 * b_ds * b_s, tl.trans(b_q), allow_tf32=False)
    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=i + tl.arange(0, BTS) < T)
        m_s = o_k[:, None] <= o_q[None, :]
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_dv += tl.dot(b_s2, tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot(2 * b_ds * b_s, tl.trans(b_q), allow_tf32=False)
        o_q += BTS
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    return


@triton.jit
def parallel_rebased_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL=BTL, BTS=BTS, BK=BK, BV=BV, DK=DK, DV=DV)
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL, BTS, BK, BV, DK, DV)


@triton.jit
def chunk_retention_fwd_kernel_h(k, v, h, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_hh, s_ht, H, T, TD, DK, DV, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_h = tl.make_block_ptr(h + i_bh * s_hh, (TD, DV), (s_ht, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for _ in range(0, T, BT):
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = d_b * b_h + tl.dot(b_k, b_v * d_i[:, None], allow_tf32=False)
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_h = tl.advance(p_h, (DK, 0))
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.jit
def chunk_retention_fwd_kernel_o(q, k, v, h, o, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_hh, s_ht, H, T, TD, scale, DK, DV, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    for i_v in range(0, tl.cdiv(DV, BV)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, 0), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (0, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (TD, DV), (s_ht, 1), (i_t * DK, i_v * BV), (BK, BV), (1, 0))
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_s = tl.zeros([BT, BT], dtype=tl.float32)
        for _ in range(0, tl.cdiv(DK, BK)):
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = b_q * scale
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_o += tl.dot(b_q * d_i[:, None], b_h, allow_tf32=False)
            b_s += tl.dot(b_q, b_k, allow_tf32=False)
            p_q = tl.advance(p_q, (0, BK))
            p_k = tl.advance(p_k, (BK, 0))
            p_h = tl.advance(p_h, (BK, 0))
        b_s *= d_s
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.jit
def chunk_retention_bwd_kernel_dh(q, do, dh, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_hh, s_ht, H, T, scale, DK, DV, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_hh, ((i + 1) * DK, DV), (s_ht, 1), (i * DK + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = d_b * b_dh + tl.dot(b_q, b_do * d_i[:, None], allow_tf32=False)


@triton.jit
def chunk_retention_bwd_kernel_dqkv(q, k, v, h, do, dh, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_hh, s_ht, H, T, TDK, scale, DK, DV, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    d_q = d_q * scale
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    for i_k in range(0, tl.cdiv(DK, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (DV, TDK), (1, s_ht), (0, i_t * DK + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_hh, (TDK, DV), (s_ht, 1), (i_t * DK + i_k * BK, 0), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_t * BT, 0), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        for _ in range(tl.cdiv(DV, BV)):
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            b_ds = tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
            b_ds = b_ds * d_s
            b_dq += tl.dot(b_do, b_h, allow_tf32=False) * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
            b_ds = tl.trans(b_ds)
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * d_k[:, None]
            b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
            b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s, b_do, allow_tf32=False)
            b_dv += tl.load(p_dv, boundary_check=(0, 1))
            tl.store(p_dv, b_dv, boundary_check=(0, 1))
            p_v = tl.advance(p_v, (0, BV))
            p_h = tl.advance(p_h, (BV, 0))
            p_do = tl.advance(p_do, (0, BV))
            p_dh = tl.advance(p_dh, (0, BV))
            p_dv = tl.advance(p_dv, (0, BV))
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
        tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.jit
def fused_chunk_retention_fwd_kernel(q, k, v, o, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b, d_o, d_h = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        b_o = tl.dot(b_s, b_v, allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h, allow_tf32=False) * d_o[:, None]
            b_h = d_b * b_h + tl.dot(b_k, b_v * d_h[:, None], allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h, allow_tf32=False) * d_o[:, None]
            b_h = d_b * b_h + tl.dot(b_k, b_v * d_h[:, None], allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h, boundary_check=(0, 1))


@triton.jit
def fused_chunk_retention_bwd_kernel(q, k, v, do, dq, dk, dv, initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b) * scale, tl.math.exp2((BT - o_i - 1) * b_b)
    d_b = tl.math.exp2(BT * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = b_do * d_q[:, None]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = b_ds * d_s
        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)
        if CHECK and i == 0:
            b_dq += tl.dot(b_dd, b_h, allow_tf32=False)
            b_h = d_b * b_h + tl.dot(b_v * d_k[None, :], b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_dd, b_h, allow_tf32=False)
            b_h = d_b * b_h + tl.dot(b_v * d_k[None, :], b_k, allow_tf32=False)
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    d_s = tl.trans(d_s)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dd = b_do * d_q[:, None]
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = b_ds * d_s
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * d_k[:, None]
            b_dv += tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False) * d_k[:, None]
            b_dv += tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None]
            b_dh = d_b * b_dh + tl.dot(b_q, b_dd, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.jit
def parallel_retention_fwd_kernel(q, k, v, o, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BTS)
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_h[None, :]
        b_o = b_o * tl.math.exp2(b_b * BTS)
        b_o = b_o + tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    d_q = tl.math.exp2(tl.arange(0, BTL) * b_b)
    b_o *= d_q[:, None]
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_s = tl.dot(b_q, b_k, allow_tf32=False) * d_s
        b_o += tl.dot(b_s, b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.jit
def _parallel_retention_bwd_dq(i_bh, i_c, i_k, i_v, i_h, k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BTS)
    d_h = tl.math.exp2((BTS - tl.arange(0, BTS)) * b_b)
    for _ in range(0, i_c * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_h[None, :]
        b_dq *= d_b
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
    b_dq *= tl.math.exp2(tl.arange(0, BTL) * b_b)[:, None] * scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        m_s = o_q[:, None] >= o_k[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_q[:, None] - o_k[None, :]) * b_b), 0)
        b_ds = tl.dot(b_do, b_v, allow_tf32=False) * d_s * scale
        b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    return


@triton.jit
def _parallel_retention_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    b_b = tl.math.log2(1 - tl.math.pow(2, -5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32)
    d_h = tl.math.exp2((BTL - tl.arange(0, BTL)) * b_b)
    b_kd = b_k * d_h[:, None]
    d_q = tl.math.exp2(tl.arange(0, BTS) * b_b)
    for i in range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * d_q[None, :]
        b_dv *= d_b
        b_s = tl.dot(b_kd, b_q, allow_tf32=False)
        b_dv += tl.dot(b_s, tl.trans(b_do), allow_tf32=False)
        b_dk *= d_b
        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
    b_dk *= d_h[:, None] * scale
    b_dv *= scale
    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.math.exp2((-o_k[:, None] + o_q[None, :]) * b_b), 0) * scale
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * d_s
        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s, tl.trans(b_do), allow_tf32=False)
        o_q += BTS
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    return


@triton.jit
def parallel_retention_bwd_kernel(q, k, v, do, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_retention_bwd_dq(i_bh, i_c, i_k, i_v, i_h, k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL=BTL, BTS=BTS, BK=BK, BV=BV, DK=DK, DV=DV)
    tl.debug_barrier()
    _parallel_retention_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL, BTS, BK, BV, DK, DV)


@triton.jit
def fused_recurrent_retention_fwd_kernel(q, k, v, o, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.pow(2, -5 - i_h * 1.0)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0)
    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _q = tl.load(p_q, mask=mask_bk, other=0) * scale
        h = b_b * h + _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o, mask=mask_bv)
        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV
    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[None, :]) * DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h, mask=mask_kv)


@triton.jit
def fused_recurrent_retention_bwd_kernel(q, k, v, do, dq, dk, dv, initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.pow(2, -5 - i_h * 1.0)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV
    h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_init_s = initial_state + i_bh * DK * DV + (i_k * BK + tl.arange(0, BK)[:, None]) * DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0)
    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        _do = tl.load(p_do, mask=mask_bv, other=0)
        h = b_b * h + _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_bk)
        p_k += DK
        p_do += DV
        p_v += DV
        p_dq += DK
    tl.debug_barrier()
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0)
        _q = tl.load(p_q, mask=mask_bk, other=0) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0)
        _v = tl.load(p_v, mask=mask_bv, other=0)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)
        d_h *= b_b
        tl.store(p_dk, d_k, mask=mask_bk)
        tl.store(p_dv, d_v, mask=mask_bv)
        p_do -= DV
        p_q -= DK
        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV


@triton.jit
def rotary_kernel(OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS, seqlen, nheads, rotary_dim, seqlen_ro, CACHE_KEY_SEQLEN, stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim, stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim, BLOCK_K: 'tl.constexpr', IS_SEQLEN_OFFSETS_TENSOR: 'tl.constexpr', IS_VARLEN: 'tl.constexpr', INTERLEAVED: 'tl.constexpr', CONJUGATE: 'tl.constexpr', BLOCK_M: 'tl.constexpr'):
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


@triton.heuristics({'HAS_DT_BIAS': lambda args: args['dt_bias_ptr'] is not None})
@triton.heuristics({'HAS_D': lambda args: args['D_ptr'] is not None})
@triton.heuristics({'HAS_Z': lambda args: args['z_ptr'] is not None})
@triton.heuristics({'BLOCK_SIZE_DSTATE': lambda args: triton.next_power_of_2(args['dstate'])})
@triton.jit
def _selective_scan_update_kernel(state_ptr, x_ptr, dt_ptr, dt_bias_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, batch, dim, dstate, stride_state_batch, stride_state_dim, stride_state_dstate, stride_x_batch, stride_x_dim, stride_dt_batch, stride_dt_dim, stride_dt_bias_dim, stride_A_dim, stride_A_dstate, stride_B_batch, stride_B_dstate, stride_C_batch, stride_C_dstate, stride_D_dim, stride_z_batch, stride_z_dim, stride_out_batch, stride_out_dim, DT_SOFTPLUS: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr', HAS_DT_BIAS: 'tl.constexpr', HAS_D: 'tl.constexpr', HAS_Z: 'tl.constexpr', BLOCK_SIZE_DSTATE: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    state_ptr += pid_b * stride_state_batch
    x_ptr += pid_b * stride_x_batch
    dt_ptr += pid_b * stride_dt_batch
    B_ptr += pid_b * stride_B_batch
    C_ptr += pid_b * stride_C_batch
    if HAS_Z:
        z_ptr += pid_b * stride_z_batch
    out_ptr += pid_b * stride_out_batch
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate)
    x_ptrs = x_ptr + offs_m * stride_x_dim
    dt_ptrs = dt_ptr + offs_m * stride_dt_dim
    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    A_ptrs = A_ptr + (offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate)
    B_ptrs = B_ptr + offs_n * stride_B_dstate
    C_ptrs = C_ptr + offs_n * stride_C_dstate
    if HAS_D:
        D_ptrs = D_ptr + offs_m * stride_D_dim
    if HAS_Z:
        z_ptrs = z_ptr + offs_m * stride_z_dim
    out_ptrs = out_ptr + offs_m * stride_out_dim
    state = tl.load(state_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0)
    dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0)
    if HAS_DT_BIAS:
        dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0)
    if DT_SOFTPLUS:
        dt = tl.log(1.0 + tl.exp(dt))
    A = tl.load(A_ptrs, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate), other=0.0)
    dA = tl.exp(A * dt[:, None])
    B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0)
    C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0)
    if HAS_D:
        D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0)
    if HAS_Z:
        z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0)
    dB = B[None, :] * dt[:, None]
    state = state * dA + dB * x[:, None]
    tl.store(state_ptrs, state, mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate))
    out = tl.sum(state * C[None, :], axis=1)
    if HAS_D:
        out += x * D
    if HAS_Z:
        out *= z * tl.sigmoid(z)
    tl.store(out_ptrs, out, mask=offs_m < dim)

