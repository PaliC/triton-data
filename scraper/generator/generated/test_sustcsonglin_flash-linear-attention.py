import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
benchmark_generation = _module
benchmark_training_throughput = _module
benchmark_cross_entropy = _module
benchmark_layernorm = _module
benchmark_abc = _module
benchmark_based = _module
benchmark_delta_rule = _module
benchmark_fla = _module
benchmark_gla = _module
benchmark_gsa = _module
benchmark_hgrn = _module
benchmark_retention = _module
benchmark_rwkv6 = _module
benchmark_simple_gla_vs_mamba2 = _module
harness = _module
ppl = _module
fla = _module
layers = _module
abc = _module
attn = _module
based = _module
delta_net = _module
gla = _module
gsa = _module
hgrn = _module
hgrn2 = _module
linear_attn = _module
multiscale_retention = _module
rebased = _module
rwkv6 = _module
simple_gla = _module
models = _module
configuration_abc = _module
modeling_abc = _module
configuration_delta_net = _module
modeling_delta_net = _module
configuration_gla = _module
modeling_gla = _module
configuration_gsa = _module
modeling_gsa = _module
configuration_hgrn = _module
modeling_hgrn = _module
configuration_hgrn2 = _module
modeling_hgrn2 = _module
configuration_linear_attn = _module
modeling_linear_attn = _module
mamba = _module
configuration_mamba = _module
modeling_mamba = _module
mamba2 = _module
configuration_mamba2 = _module
modeling_mamba2 = _module
retnet = _module
configuration_retnet = _module
modeling_retnet = _module
configuration_rwkv6 = _module
modeling_rwkv6 = _module
samba = _module
configuration_samba = _module
modeling_samba = _module
transformer = _module
configuration_transformer = _module
modeling_transformer = _module
utils = _module
modules = _module
activations = _module
convolution = _module
feature_map = _module
fused_bitlinear = _module
fused_cross_entropy = _module
fused_kl_div = _module
fused_linear_cross_entropy = _module
fused_norm_gate = _module
l2norm = _module
layernorm = _module
layernorm_gated = _module
rotary = _module
ops = _module
chunk = _module
naive = _module
fused_chunk = _module
parallel = _module
common = _module
chunk_h = _module
fused_recurrent = _module
delta_rule = _module
chunk = _module
fused_chunk = _module
fused_recurrent = _module
wy_fast = _module
generalized_delta_rule = _module
iplr = _module
fused_recurrent = _module
chunk = _module
fused_chunk = _module
chunk = _module
fused_recurrent = _module
chunk = _module
fused_recurrent = _module
chunk = _module
fused_chunk = _module
fused_recurrent = _module
parallel = _module
retention = _module
chunk = _module
fused_chunk = _module
fused_recurrent = _module
parallel = _module
rotary = _module
rwkv4 = _module
fused_recurrent = _module
chunk = _module
chunk_naive = _module
fused_recurrent = _module
recurrent_naive = _module
chunk = _module
utils = _module
cumsum = _module
logcumsumexp = _module
logsumexp = _module
matmul = _module
softmax = _module
setup = _module
test_based = _module
test_gla = _module
test_conv = _module
test_cross_entropy = _module
test_kl_div = _module
test_layernorm = _module
test_delta = _module
test_gsa = _module
test_hgrn = _module
test_linear_attn = _module
test_retention = _module
test_rwkv6 = _module
test_simple_gla = _module
test_fused_chunk = _module
test_padding = _module
flame = _module
logging = _module
parser = _module
preprocess = _module
run = _module
convert_from_llama = _module
convert_from_rwkv6 = _module

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


import torch.nn as nn


import torch.nn.functional as F


import triton


from torch.nn import functional as F


from typing import TYPE_CHECKING


from typing import Optional


from typing import Tuple


import warnings


import math


from typing import Any


from typing import Dict


from typing import Union


import torch.utils.checkpoint


from torch import nn


import triton.language as tl


from typing import cast


from torch import Tensor


from torch.autograd.function import Function


from torch.autograd.function import FunctionCtx


from torch.autograd.function import once_differentiable


import re


from torch.utils.cpp_extension import CUDA_HOME


@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8), triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton.Config({'BT': 64}, num_warps=8), triton.Config({'BT': 128}, num_warps=2), triton.Config({'BT': 128}, num_warps=4), triton.Config({'BT': 128}, num_warps=8), triton.Config({'BT': 256}, num_warps=2), triton.Config({'BT': 256}, num_warps=4), triton.Config({'BT': 256}, num_warps=8)], key=['D'])
@triton.jit
def logsigmoid_fwd_kernel(x, y, T: 'tl.constexpr', D: 'tl.constexpr', BT: 'tl.constexpr'):
    i = tl.program_id(0)
    o_i = i * BT + tl.arange(0, BT)
    p_x = x + o_i
    p_y = y + o_i
    mask = o_i < T
    b_x = tl.load(p_x, mask=mask, other=0.0)
    b_m = tl.minimum(0.0, b_x)
    b_z = 1.0 + tl.exp(-tl.abs(b_x))
    b_y = b_m - tl.log(b_z)
    tl.store(p_y, b_y, mask=mask)


@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8), triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton.Config({'BT': 64}, num_warps=8), triton.Config({'BT': 128}, num_warps=2), triton.Config({'BT': 128}, num_warps=4), triton.Config({'BT': 128}, num_warps=8), triton.Config({'BT': 256}, num_warps=2), triton.Config({'BT': 256}, num_warps=4), triton.Config({'BT': 256}, num_warps=8)], key=['D'])
@triton.jit
def logsigmoid_bwd_kernel(x, dx, dy, T: 'tl.constexpr', D: 'tl.constexpr', BT: 'tl.constexpr'):
    i = tl.program_id(0)
    o_i = i * BT + tl.arange(0, BT)
    p_x = x + o_i
    p_dx = dx + o_i
    p_dy = dy + o_i
    mask = o_i < T
    b_x = tl.load(p_x, mask=mask, other=0.0)
    b_dy = tl.load(p_dy, mask=mask, other=0.0)
    b_dx = b_dy * (1.0 - tl.sigmoid(b_x))
    tl.store(p_dx, b_dx, mask=mask)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N', 'HAS_RESIDUAL', 'STORE_RESIDUAL_OUT', 'IS_RMS_NORM', 'HAS_BIAS'])
@triton.jit
def _layer_norm_fwd_quant_kernel(X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean, Rstd, stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, N, eps, IS_RMS_NORM: 'tl.constexpr', BLOCK_N: 'tl.constexpr', HAS_RESIDUAL: 'tl.constexpr', STORE_RESIDUAL_OUT: 'tl.constexpr', HAS_WEIGHT: 'tl.constexpr', HAS_BIAS: 'tl.constexpr'):
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
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b
    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-05)
    y = tl.math.round(y * scale)
    y = tl.maximum(tl.minimum(y, 127), -128) / scale
    tl.store(Y + cols, y, mask=mask)


@triton.heuristics({'HAS_BIAS': lambda args: args['B'] is not None})
@triton.heuristics({'HAS_Z': lambda args: args['Z'] is not None})
@triton.heuristics({'RECOMPUTE_OUTPUT': lambda args: args['Y'] is not None})
@triton.jit
def _layer_norm_bwd_kernel(X, W, B, Z, Y, DY, DX, DW, DB, DZ, Mean, Rstd, stride_x_row, stride_z_row, stride_y_row, stride_dy_row, stride_dx_row, stride_dz_row, stride_dw_row, stride_db_row, M, N, eps, rows_per_program, NORM_BEFORE_GATE: 'tl.constexpr', IS_RMS_NORM: 'tl.constexpr', HAS_BIAS: 'tl.constexpr', HAS_Z: 'tl.constexpr', RECOMPUTE_OUTPUT: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    row_block_id = tl.program_id(0)
    group = tl.program_id(1)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row + group * N
    if HAS_Z:
        Z += row_start * stride_z_row + group * N
        DZ += row_start * stride_dz_row + group * N
    DY += row_start * stride_dy_row + group * N
    DX += row_start * stride_dx_row + group * N
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    w = tl.load(W + cols, mask=mask)
    if (RECOMPUTE_OUTPUT or HAS_Z) and HAS_BIAS:
        B += group * N
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
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0)
            x_og = x
            x = x_og * z * tl.sigmoid(z)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z + cols, mask=mask, other=0.0)
            z_sigmoid = tl.sigmoid(z)
            y = xhat * w + b if HAS_BIAS else xhat * w
            if RECOMPUTE_OUTPUT:
                tl.store(Y + cols, y * z * z_sigmoid, mask=mask)
            dz = dy * y * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dy *= z * z_sigmoid
        elif RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        c1 = tl.sum(xhat * wdy, axis=0) / N
        if not IS_RMS_NORM:
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            dx = (wdy - xhat * c1) * rstd
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if HAS_Z and not NORM_BEFORE_GATE:
            z_sigmoid = tl.sigmoid(z)
            dz = dx * x_og * z_sigmoid * (1 + z * (1 - z_sigmoid))
            tl.store(DZ + cols, dz, mask=mask)
            dx *= z * z_sigmoid
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_Z:
            Z += stride_z_row
            DZ += stride_dz_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * stride_dw_row + group * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * stride_db_row + group * N + cols, db, mask=mask)


@triton.heuristics({'HAS_SMOOTHING': lambda args: args['label_smoothing'] > 0.0})
@triton.jit
def cross_entropy_fwd_kernel(loss_ptr, lse_ptr, z_loss_ptr, logits_ptr, labels_ptr, label_smoothing, logit_scale, lse_square_scale, ignore_index, total_classes, class_start_idx, n_cols, n_rows, logits_row_stride, BLOCK_SIZE: 'tl.constexpr', HAS_SMOOTHING: 'tl.constexpr', SPLIT: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)
    logits_ptr = logits_ptr + row_idx * logits_row_stride
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float('inf'))
    logits = logits * logit_scale
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignore_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols, (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
            if HAS_SMOOTHING:
                loss = (lse if not SPLIT else 0.0) - label_smoothing * sum_logits / total_classes - (1 - label_smoothing) * logits_label
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        elif HAS_SMOOTHING:
            loss = label_smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
        else:
            loss = 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    if not SPLIT:
        tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)


@triton.heuristics({'HAS_SMOOTHING': lambda args: args['label_smoothing'] > 0.0})
@triton.jit
def cross_entropy_bwd_kernel(dlogits_ptr, dloss_ptr, logits_ptr, lse_ptr, labels_ptr, label_smoothing, logit_scale, lse_square_scale, ignore_index, total_classes, class_start_idx, n_cols, logits_row_stride, dlogits_row_stride, dloss_row_stride, BLOCK_SIZE: 'tl.constexpr', HAS_SMOOTHING: 'tl.constexpr'):
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
        smooth_negative = label_smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - label_smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, dloss * logit_scale * probs, mask=col_offsets < n_cols)


@triton.jit
def kl_div_kernel(logits, target_logits, loss, s_logits, s_loss, reduction: 'tl.constexpr', N: 'tl.constexpr', V: 'tl.constexpr', BV: 'tl.constexpr'):
    i_n = tl.program_id(0)
    logits += i_n * s_logits
    target_logits += i_n * s_logits
    sm, tm = float('-inf'), float('-inf')
    sd, td = 0.0, 0.0
    NV = tl.cdiv(V, BV)
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sm = tl.max(b_sl)
        m_new = tl.maximum(sm, b_sm)
        sd = sd * tl.exp(sm - m_new) + tl.sum(tl.exp(b_sl - m_new))
        sm = m_new
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tm = tl.max(b_tl)
        m_new = tl.maximum(tm, b_tm)
        td = td * tl.exp(tm - m_new) + tl.sum(tl.exp(b_tl - m_new))
        tm = m_new
    b_loss = 0.0
    for iv in range(0, NV):
        o_x = iv * BV + tl.arange(0, BV)
        b_sl = tl.load(logits + o_x, mask=o_x < V, other=float('-inf'))
        b_tl = tl.load(target_logits + o_x, mask=o_x < V, other=float('-inf'))
        b_sp_log = b_sl - sm - tl.log(sd)
        b_tp_log = b_tl - tm - tl.log(td)
        b_sp = tl.exp(b_sp_log)
        b_tp = tl.exp(b_tp_log)
        b_kl = tl.where(o_x < V, b_tp * (b_tp_log - b_sp_log), 0)
        b_dl = -b_tp + b_sp
        b_loss += tl.sum(b_kl)
        if reduction == 'batchmean':
            b_dl = b_dl / N
        tl.store(logits + o_x, b_dl, mask=o_x < V)
    if reduction == 'batchmean':
        b_loss = b_loss / N
    tl.store(loss + i_n * s_loss, b_loss)


@triton.jit
def elementwise_mul_kernel(x, g, N: 'tl.constexpr', B: 'tl.constexpr'):
    """
    This function multiplies each element of the tensor pointed by x with the value pointed by g.
    The multiplication is performed in-place on the tensor pointed by x.

    Parameters:
    x:
        Pointer to the input tensor.
    g:
        Pointer to the gradient output value.
    N (int):
        The number of columns in the input tensor.
    B (int):
        The block size for Triton operations.
    """
    i_x = tl.program_id(0)
    o_x = i_x * B + tl.arange(0, B)
    b_g = tl.load(g)
    b_x = tl.load(x + o_x, mask=o_x < N)
    tl.store(x + o_x, b_x * b_g, mask=o_x < N)


@triton.jit
def cross_entropy_kernel(logits, lse, target, loss, total, ignore_index, label_smoothing: 'tl.constexpr', logit_scale: 'tl.constexpr', reduction: 'tl.constexpr', V: 'tl.constexpr', BV: 'tl.constexpr'):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now.
    Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.

    Args:
        logits:
            Pointer to logits tensor.
        lse:
            Pointer to logsumexp tensor.
        target: Pointer to target tensor.
        loss:
            Pointer to tensor to store the loss.
        V (int):
            The number of columns in the input tensor.
        total (int):
            The number of non-ignored classes.
        ignore_index (int):
            The index to ignore in the target.
        label_smoothing (float):
            The amount of smoothing when computing the loss, where 0.0 means no smoothing.
        reduction (str):
            The string for the reduction to apply
        BV (int):
            The block size for vocab.
    """
    i_n = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    b_y = tl.load(target + i_n)
    logits += i_n * V
    if b_y == ignore_index:
        for i in range(0, V, BV):
            o_v = i + tl.arange(0, BV)
            tl.store(logits + o_v, 0.0, mask=o_v < V)
        return
    b_l = tl.load(logits + b_y) * logit_scale
    b_lse = tl.load(lse + i_n)
    b_loss = b_lse - b_l
    b_z = 0.0
    eps = label_smoothing / V
    tl.debug_barrier()
    for iv in range(0, NV):
        o_v = iv * BV + tl.arange(0, BV)
        b_logits = tl.load(logits + o_v, mask=o_v < V, other=float('-inf')) * logit_scale
        if label_smoothing > 0:
            b_z += tl.sum(tl.where(o_v < V, -eps * b_logits, 0.0))
        b_p = (tl.exp(b_logits - b_lse) - eps) * logit_scale
        if reduction == 'mean':
            b_p = b_p / total
        tl.store(logits + o_v, b_p, mask=o_v < V)
        tl.debug_barrier()
    if label_smoothing > 0:
        b_loss = b_loss * (1 - label_smoothing) + (b_z + label_smoothing * b_lse)
    b_l = tl.load(logits + b_y)
    if reduction == 'mean':
        b_loss = b_loss / total
        b_l += (label_smoothing - 1) / total * logit_scale
    else:
        b_l += (label_smoothing - 1) * logit_scale
    tl.store(loss + i_n, b_loss)
    tl.store(logits + b_y, b_l)


@triton.heuristics({'HAS_BIAS': lambda args: args['B'] is not None})
@triton.heuristics({'HAS_Z': lambda args: args['Z'] is not None})
@triton.jit
def _layer_norm_fwd_1pass_kernel(X, Y, W, B, Z, Mean, Rstd, stride_x_row, stride_y_row, stride_z_row, M, N, eps, BLOCK_N: 'tl.constexpr', HAS_BIAS: 'tl.constexpr', HAS_Z: 'tl.constexpr', NORM_BEFORE_GATE: 'tl.constexpr', IS_RMS_NORM: 'tl.constexpr'):
    row = tl.program_id(0)
    group = tl.program_id(1)
    X += row * stride_x_row + group * N
    Y += row * stride_y_row + group * N
    if HAS_Z:
        Z += row * stride_z_row + group * N
    if not IS_RMS_NORM:
        Mean += group * M
    Rstd += group * M
    W += group * N
    if HAS_BIAS:
        B += group * N
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N)
        x *= z * tl.sigmoid(z)
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
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=mask)
        y *= z * tl.sigmoid(z)
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N'])
@triton.jit
def _l2_norm_fwd_1pass_kernel(X, Y, stride_x_row, N, eps, BLOCK_N: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['N'])
@triton.jit
def _l2_norm_bwd_kernel(X, DY, DX, stride_x_row, N, eps, BLOCK_N: 'tl.constexpr'):
    row = tl.program_id(0)
    X += row * stride_x_row
    DX += row * stride_x_row
    DY += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols, mask=cols < N, other=0.0)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)


@triton.jit
def chunk_abc_fwd_kernel_h(k, v, z, h, h0, ht, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', NORMK: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    if NORMK:
        p_z0 = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), (i_k * BK,), (BK,), (0,))
    else:
        p_z0 = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), (i_v * BV,), (BV,), (0,))
    b_zp = tl.load(p_z0)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if NORMK:
            p_zc = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zp - b_zc), b_zc
            b_h = b_h * b_r[:, None]
            b_k = tl.exp(b_k - b_zc[:, None])
        else:
            p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zp - b_zc), b_zc
            b_h = b_h * b_r[None, :]
            b_v = tl.exp(b_v - b_zc[None, :])
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_intra_K(v, z, o, A, s_v_h, s_v_t, s_v_d, T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, tl.exp(b_v - b_zn[None, :]), allow_tf32=False)
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_o *= tl.exp(b_zn[None, :] - b_z)
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    for j in range(0, BC):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        b_v = tl.load(p_v, boundary_check=(0,))
        m_i = o_i[:, None] >= j
        b_o += tl.where(m_i, b_A[:, None] * tl.exp(b_v[None, :] - b_z), 0)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_K(q, k, z, h, o, A, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_A += tl.dot(b_q, b_k, allow_tf32=False)
    p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_z = tl.load(p_z, boundary_check=(0, 1))
    p_zp = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), (i_p * V + i_v * BV,), (BV,), (0,))
    b_zp = tl.load(p_zp, boundary_check=(0,))
    b_o = b_o * tl.exp(b_zp[None, :] - b_z)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.where(m_s, b_A, 0.0)
    if i_v == 0:
        tl.store(p_A, b_A, boundary_check=(0, 1))


@triton.jit
def chunk_abc_fwd_kernel_intra_V(q, k, z, A, s_k_h, s_k_t, s_k_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC) % NC
    n_bh = tl.num_programs(2)
    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_zn = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
        b_zn = tl.load(p_zn, boundary_check=(0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_q = b_q * tl.exp(b_zn[None, :] - b_z) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k = tl.exp(b_k - b_zn[:, None])
        b_A = tl.dot(b_q, b_k, allow_tf32=False)
        tl.store(p_A, b_A, boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
        for j in range(0, BC):
            b_k = tl.load(p_k, boundary_check=(0,))
            b_A = tl.sum(b_q * tl.exp(b_k[None, :] - b_z) * scale, 1)
            b_A = tl.where(o_i >= j, b_A, 0.0)
            tl.store(A + o_A + j, b_A, mask=m_A)
            p_k = tl.advance(p_k, (K,))


@triton.jit
def chunk_abc_fwd_kernel_V(q, v, z, h, o, A, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_zp = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), (i_p * K + i_k * BK,), (BK,), (0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_zp = tl.load(p_zp, boundary_check=(0,))
        b_q = b_q * tl.exp(b_zp[None, :] - b_z)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_dh(q, z, do, dh, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', NORMK: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    b_zp = tl.full([BK if NORMK else BV], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        i_p = tl.maximum(i_t * BT - 1, 0)
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        if NORMK:
            p_z = tl.make_block_ptr(z + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            p_zc = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), (i_p * K + i_k * BK,), (BK,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zc - b_zp), b_zc
            b_z = tl.load(p_z, boundary_check=(0, 1))
            b_q = b_q * tl.exp(b_zc[:, None] - b_z)
            b_dh = b_dh * b_r[:, None]
        else:
            p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), (i_p * V + i_v * BV,), (BV,), (0,))
            b_zc = tl.load(p_zc, boundary_check=(0,))
            b_r, b_zp = tl.exp(b_zc - b_zp), b_zc
            b_z = tl.load(p_z, boundary_check=(0,))
            b_do = b_do * tl.exp(b_zc[None, :] - b_z)
            b_dh = b_dh * b_r[None, :]
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def chunk_abc_bwd_kernel_V(k, v, z, h, A, do, dh, dq, dk, dv, dA, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    n_bh = tl.num_programs(2)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_zc = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + BT - 1) * K + i_k * BK,), (BK,), (0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_zc = tl.load(p_zc, boundary_check=(0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_k = tl.exp(b_k - b_zc[None, :])
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False)
        if i_k == 0:
            b_dv += tl.dot(b_A, b_do, allow_tf32=False)
        b_do = b_do * scale
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
    p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_zp = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), (i_p * K + i_k * BK,), (BK,), (0,))
    b_zp = tl.load(p_zp, boundary_check=(0,))
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_z = tl.exp(b_zp[None, :] - b_z)
    b_dq = b_dq * b_z
    b_dk = b_dk * b_k
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_dA = tl.where(m_s, b_dA, 0.0)
    if i_k == 0:
        tl.store(p_dA, b_dA, boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_intra_V(q, k, z, dA, dq, dk, s_k_h, s_k_t, s_k_d, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC) * K + i_k * BK,), (BK,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_z = tl.load(p_z, boundary_check=(0, 1))
    b_zq = tl.exp(b_zn[None, :] - b_z)
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kz = tl.exp(b_k - b_zn[None, :])
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        b_dq += tl.dot(b_dA, b_kz, allow_tf32=False)
    b_dq *= b_zq
    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    for j in range(0, BC):
        p_kj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, boundary_check=(0,))
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i, b_dA[:, None] * tl.exp(b_kj[None, :] - b_z), 0.0)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kz = tl.exp(b_k - b_zn[None, :])
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_qz = b_q * tl.exp(b_zn[None, :] - b_z)
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        b_dk += tl.dot(tl.trans(b_dA), b_qz, allow_tf32=False)
    b_dk *= b_kz
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        p_qj = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_zj = tl.make_block_ptr(z + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_dA = tl.load(dA + o_dA + j * BT, mask=i_t * BT + i_i * BC + j < T, other=0)
        b_qj = tl.load(p_qj, boundary_check=(0,))
        b_zj = tl.load(p_zj, boundary_check=(0,))
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_k - b_zj[None, :]), 0.0)
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_intra_K(v, z, do, dA, s_v_h, s_v_t, s_v_d, scale, T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC) % NC
    n_bh = tl.num_programs(2)
    if i_i > i_j:
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_i * BC) * V + i_v * BV,), (BV,), (0,))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_dA = tl.make_block_ptr(dA + (i_bh + i_v * n_bh) * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        b_zn = tl.load(p_zn, boundary_check=(0,))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_zn[None, :] - b_z) * scale
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.exp(b_v - b_zn[:, None])
        b_dA = tl.dot(b_do, b_v, allow_tf32=False)
        tl.store(p_dA, b_dA, boundary_check=(0, 1))
    elif i_i == i_j:
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_j * BC) * V + i_v * BV,), (BV,), (0,))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        o_i = tl.arange(0, BC)
        o_A = (i_bh + i_v * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
        for j in range(0, BC):
            b_v = tl.load(p_v, boundary_check=(0,))
            b_dA = tl.sum(b_do * tl.exp(b_v[None, :] - b_z), 1)
            b_dA = tl.where(o_i >= j, b_dA, 0)
            tl.store(dA + o_A + j, b_dA, mask=m_A)
            p_v = tl.advance(p_v, (V,))


@triton.jit
def chunk_abc_bwd_kernel_K(q, k, v, z, h, A, do, dh, dq, dk, dv, dA, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_p = tl.maximum(i_t * BT - 1, 0)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_A = tl.dot(b_q * scale, tl.trans(b_k), allow_tf32=False)
    b_A = tl.where(m_s, b_A, 0.0)
    tl.store(p_A, b_A, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_zp = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), (i_p * V + i_v * BV,), (BV,), (0,))
        p_zc = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + BT - 1) * V + i_v * BV,), (BV,), (0,))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_zp = tl.load(p_zp, boundary_check=(0,))
        b_zc = tl.load(p_zc, boundary_check=(0,))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v = tl.exp(b_v - b_zc[None, :])
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_z = tl.exp(b_zp[None, :] - b_z)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * b_z * scale
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = b_v * tl.dot(b_k, b_dh, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    b_dq += tl.dot(b_dA, b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_dA), b_q, allow_tf32=False)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_intra_KV(v, z, A, do, dv, s_v_h, s_v_t, s_v_d, T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (s_v_d,), ((i_t * BT + i_i * BC + BC - 1) * V + i_v * BV,), (BV,), (0,))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_zn[None, :] - b_z)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do, allow_tf32=False)
    b_dv *= tl.exp(b_v - b_zn[None, :])
    o_i = tl.arange(0, BC)
    for j in range(0, BC):
        p_z = tl.make_block_ptr(z + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T * BT,), (1,), ((i_t * BT + i_i * BC + j) * BT + i_i * BC,), (BC,), (0,))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T * V,), (1,), ((i_t * BT + i_i * BC + j) * V + i_v * BV,), (BV,), (0,))
        b_A = tl.load(p_A, boundary_check=(0,))
        b_z = tl.load(p_z, boundary_check=(0,))
        b_do = tl.load(p_do, boundary_check=(0,))
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_v - b_z[None, :]) * b_A[:, None] * b_do[None, :], 0.0)
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.jit
def chunk_abc_bwd_kernel_rcum_inter(s, z, ss, doo, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr', NT: 'tl.constexpr'):
    i_m, i_bh = tl.program_id(0), tl.program_id(1)
    b_sp = tl.zeros([BS], dtype=tl.float32)
    b_zp = tl.full([BS], float('inf'), dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_zc = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (s_s_d,), (i_t * BT * S + i_m * BS,), (BS,), (0,))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        p_doo = tl.make_block_ptr(doo + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_m * BS), (BT, BS), (1, 0))
        b_zc = tl.load(p_zc, boundary_check=(0,))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_ss = tl.load(p_ss, boundary_check=(0, 1))
        b_doo = tl.exp(b_s - b_zp[None, :]) * b_sp[None, :]
        tl.store(p_doo, b_doo, boundary_check=(0, 1))
        b_sp = b_sp * tl.exp(b_zc - b_zp) + tl.sum(b_ss * tl.exp(b_zc[None, :] - b_z), 0)
        b_zp = b_zc


@triton.jit
def chunk_abc_bwd_kernel_rcum_intra(s, z, ss, doo, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BS: 'tl.constexpr', NC: 'tl.constexpr'):
    i_s, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    o_i = tl.arange(0, BC)
    m_o = tl.full([BC, BC], 1.0, dtype=tl.float32)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    p_zn = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (s_s_d,), ((i_t * BT + i_i * BC + BC - 1) * S + i_s * BS,), (BS,), (0,))
    p_doo = tl.make_block_ptr(doo + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_i * BC, i_s * BS), (BC, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_zn = tl.load(p_zn, boundary_check=(0,))
    b_doo = tl.zeros([BC, BS], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT + i_j * BC, i_s * BS), (BC, BS), (1, 0))
        b_z = tl.load(p_z, boundary_check=(0, 1))
        b_ss = tl.load(p_ss, boundary_check=(0, 1))
        b_doo += b_ss * tl.exp(b_zn[None, :] - b_z)
    b_doo = tl.exp(b_s - b_zn[None, :]) * tl.dot(m_o, b_doo, allow_tf32=False)
    for j in range(0, BC):
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T * S,), (1,), ((i_t * BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        p_ss = tl.make_block_ptr(ss + i_bh * s_s_h, (T * S,), (1,), ((i_t * BT + i_i * BC + j) * S + i_s * BS,), (BS,), (0,))
        b_z = tl.load(p_z, boundary_check=(0,))
        b_ss = tl.load(p_ss, boundary_check=(0,))
        m_i = o_i[:, None] <= j
        b_doo += tl.where(m_i, tl.exp(b_s - b_z[None, :]) * b_ss[None, :], 0.0)
    b_doo += tl.load(p_doo, boundary_check=(0, 1))
    tl.store(p_doo, b_doo, boundary_check=(0, 1))


@triton.jit
def fused_chunk_based_fwd_kernel(q, k, v, o, z, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h_0o = tl.zeros([BV], dtype=tl.float32)
    b_h_1o = tl.zeros([BK, BV], dtype=tl.float32)
    b_h_2o = tl.zeros([BK * BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
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
def fused_chunk_based_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h_1o = tl.zeros([BV, BK], dtype=tl.float32)
    b_h_2o = tl.zeros([BV, BK * BK], dtype=tl.float32)
    k_1o = tl.zeros([1, BK], dtype=tl.float32)
    k_2o = tl.zeros([1, BK * BK], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dz = dz + i_bh * T + tl.arange(0, BT) + i * BT
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=tl.arange(0, BT) + i * BT < T)
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
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i, i_v * BV), (BT, BV), (1, 0))
        p_dz = dz + i_bh * T + tl.arange(0, BT) + i
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dz = tl.load(p_dz, mask=tl.arange(0, BT) + i < T)
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
def parallel_based_fwd_kernel(q, k, v, o, z, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
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
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
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
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    tl.store(p_z, b_z, mask=i_c * BTL + tl.arange(0, BTL) < T)


@triton.jit
def _parallel_based_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr'):
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q = b_q * scale
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, 0), (BV, BTS), (0, 1))
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
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
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
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    return


@triton.jit
def _parallel_based_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr'):
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32)
    for i in range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
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
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
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
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    return


@triton.jit
def parallel_based_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_based_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL=BTL, BTS=BTS, BK=BK, BV=BV, K=K, V=V)
    tl.debug_barrier()
    _parallel_based_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BTL, BTS, BK, BV, K, V)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK', 'BV', 'USE_G', 'USE_GK', 'USE_GV'])
@triton.heuristics({'USE_INITIAL_STATE': lambda args: args['h0'] is not None, 'STORE_FINAL_STATE': lambda args: args['ht'] is not None})
@triton.jit
def chunk_fwd_kernel_h(k, v, h, g, gk, gv, h0, ht, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', USE_G: 'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + i_bh * T + last_idx)
            b_h *= tl.exp(b_g_last)
            p_g = g + i_bh * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
            b_g = tl.load(p_g, mask=i_t * BT + tl.arange(0, BT) < T, other=0.0)
            b_v = b_v * tl.exp(b_g_last - b_g)[:, None]
        if USE_GK:
            p_gk_last = gk + i_bh * s_k_h + last_idx * K + i_k * BK + tl.arange(0, BK)
            p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
            b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) < K, other=0.0)
            b_h *= tl.exp(b_gk_last)[:, None]
            p_gk = tl.make_block_ptr(gk + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_k = b_k * tl.exp(b_gk_last[:, None] - b_gk)
        if USE_GV:
            p_gv_last = gv + i_bh * s_v_h + last_idx * V + i_v * BV + tl.arange(0, BV)
            p_gv_last = tl.max_contiguous(tl.multiple_of(p_gv_last, BV), BV)
            b_gv_last = tl.load(p_gv_last, mask=i_v * BV + tl.arange(0, BV) < V, other=0.0)
            b_h *= tl.exp(b_gv_last)[None, :]
            p_gv = tl.make_block_ptr(gv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_v = b_v * tl.exp(b_gv_last[None, :] - b_gv)
        b_h += tl.dot(b_k, b_v)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK', 'BV', 'USE_G', 'USE_GK', 'USE_GV'])
@triton.heuristics({'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None})
@triton.jit
def chunk_bwd_kernel_dh(q, g, gk, gv, do, dh, dht, dh0, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', NG: 'tl.constexpr', USE_G: 'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr', STORE_INITIAL_STATE_GRADIENT: 'tl.constexpr', USE_FINAL_STATE_GRADIENT: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        if USE_G:
            p_g = g + i_bg * T + i_t * BT + tl.arange(0, BT)
            p_g = tl.max_contiguous(tl.multiple_of(p_g, BT), BT)
            b_g = tl.load(p_g, mask=i_t * BT + tl.arange(0, BT) < T, other=0.0)
            b_q = b_q * tl.exp(b_g)[None, :]
            b_g_last = tl.load(g + i_bg * T + last_idx)
            b_dh *= tl.exp(b_g_last)
        if USE_GK:
            p_gk = tl.make_block_ptr(gk + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_q = b_q * tl.exp(b_gk)
            p_gk_last = gk + i_bg * s_k_h + last_idx * K + i_k * BK + tl.arange(0, BK)
            p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
            b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) < K, other=0.0)
            b_dh *= tl.exp(b_gk_last)[:, None]
        if USE_GV:
            p_gv = tl.make_block_ptr(gv + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            b_gv = tl.load(p_gv, boundary_check=(0, 1))
            b_do = b_do * tl.exp(b_gv)
            p_gv_last = gv + i_bg * s_v_h + last_idx * V + i_v * BV + tl.arange(0, BV)
            p_gv_last = tl.max_contiguous(tl.multiple_of(p_gv_last, BV), BV)
            b_gv_last = tl.load(p_gv_last, mask=i_v * BV + tl.arange(0, BV) < V, other=0.0)
            b_dh *= tl.exp(b_gv_last)[None, :]
        b_dh += tl.dot(b_q, b_do)
    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh, boundary_check=(0, 1))


@triton.jit
def fused_recurrent_fwd_kernel(q, k, v, alpha, beta, o, ha, h0, ht, s_qk_h, s_vo_h, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_alpha = alpha + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_beta = beta + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_ha = ha + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_alpha = tl.load(p_alpha, mask=mask_bk, other=0)
        b_beta = tl.load(p_beta, mask=mask_bk, other=0)
        tmp = tl.sum(h * b_alpha[None, :], axis=1)
        h += tmp[:, None] * b_beta[None, :] + b_k[None, :] * b_v[:, None]
        _o = h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o, mask=mask_bv)
        tl.store(p_ha, tmp, mask=mask_bv)
        p_q += K
        p_k += K
        p_o += V
        p_v += V
        p_ha += V
        p_alpha += K
        p_beta += K
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h, mask=mask_kv)


@triton.jit
def fused_recurrent_bwd_kernel(q, k, v, alpha, beta, ha, dht, dh0, do, dq, dk, dv, dalpha, dbeta, dha, h0, s_qk_h, s_vo_h, NK, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', USE_DH0: 'tl.constexpr', USE_DHT: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_ha = ha + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_alpha = alpha + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_beta = beta + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dbeta = dbeta + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_dha = dha + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_beta = tl.load(p_beta, mask=mask_bk, other=0)
        b_alpha = tl.load(p_alpha, mask=mask_bk, other=0)
        b_ha = tl.load(p_ha, mask=mask_bv, other=0)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * b_v[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)
        tl.store(p_dk, d_k, mask=mask_bk)
        tl.store(p_dv, d_v, mask=mask_bv)
        b_dha = tl.sum(d_h * b_beta[:, None], axis=0)
        tl.store(p_dha, b_dha, mask=mask_bv)
        b_dbeta = tl.sum(d_h * b_ha[None, :], axis=1)
        tl.store(p_dbeta, b_dbeta, mask=mask_bk)
        d_h += b_dha[None, :] * b_alpha[:, None]
        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V
        p_beta -= K
        p_dbeta -= K
        p_alpha -= K
        p_dha -= V
        p_ha -= V
    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, d_h, mask=mask_bk[:, None] & mask_bv[None, :])
    tl.debug_barrier()
    h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_beta = beta + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_ha = ha + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dha = dha + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_alpha = alpha + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dalpha = dalpha + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0)
    for i in range(0, T):
        d_ha = tl.load(p_dha, mask=mask_bv, other=0)
        d_alpha = tl.sum(d_ha[None, :] * h, axis=1)
        tl.store(p_dalpha, d_alpha, mask=mask_bk)
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_beta = tl.load(p_beta, mask=mask_bk, other=0)
        b_ha = tl.load(p_ha, mask=mask_bv, other=0)
        h += b_k[:, None] * b_v[None, :] + b_beta[:, None] * b_ha[None, :]
        _d_q = h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_bk)
        p_k += K
        p_do += V
        p_v += V
        p_dk += K
        p_dalpha += K
        p_dha += V
        p_ha += V
        p_dq += K
        p_beta += K


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def fwd_prepare_dv_kernel(q, k, do, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, T, K, V, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_dv = tl.dot(b_A, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_delta_rule_fwd_kernel_h(k, v, d, v_new, h, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_h_cumsum = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC)):
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_v -= tl.dot(b_d, b_h)
            tl.store(p_v_new, b_v, boundary_check=(0, 1))
            b_h_cumsum += tl.dot(b_k, b_v, allow_tf32=False)
        b_h += b_h_cumsum
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.jit
def chunk_linear_attn_fwd_kernel_o(q, k, v, h, o, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s = tl.where(m_s, b_s, 0)
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s, b_v, allow_tf32=False)) * scale
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_delta_rule_bwd_kernel_dhu(q, k, d, dht, dh0, do, dh, dv, dv2, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', USE_DHT: 'tl.constexpr', USE_DH0: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        b_dh_tmp = tl.zeros([BK, BV], dtype=tl.float32)
        for i_c in range(tl.cdiv(BT, BC) - 1, -1, -1):
            p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT + i_c * BC, i_k * BK), (BC, BK), (1, 0))
            p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT + i_c * BC), (BK, BC), (0, 1))
            p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = b_q * scale
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_d = tl.load(p_d, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_dv += tl.dot(b_k, b_dh, allow_tf32=False)
            p_dv2 = tl.make_block_ptr(dv2 + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT + i_c * BC, i_v * BV), (BC, BV), (1, 0))
            tl.store(p_dv2, b_dv, boundary_check=(0, 1))
            b_dh_tmp += tl.dot(b_q, b_do, allow_tf32=False)
            b_dh_tmp -= tl.dot(b_d, b_dv, allow_tf32=False)
        b_dh += b_dh_tmp
    if USE_DH0:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_delta_rule_bwd_kernel_dqkw(q, k, v, w, h, do, dh, dq, dk, dv, dw, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, b_dh, allow_tf32=False)
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dw += tl.dot(b_dv, b_h, allow_tf32=False)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds, 0)
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dq *= scale
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dw = tl.make_block_ptr(dw + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dw, -b_dw, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK'])
@triton.jit
def fused_chunk_delta_rule_fwd_kernel(q, k, v, v_new, d, o, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_v_new = tl.make_block_ptr(v_new + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_q = b_q * scale
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_v_prime = tl.dot(b_d, b_h, allow_tf32=False)
        b_v = b_v - b_v_prime
        tl.store(p_v_new, b_v, boundary_check=(0, 1))
        b_o = tl.dot(b_s, b_v, allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_v_new = tl.advance(p_v_new, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_d = tl.advance(p_d, (BT, 0))
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(final_state + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def fused_chunk_delta_rule_bwd_kernel(q, k, v, d, dht, dh0, do, dq, dk, dv, dd, initial_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', USE_DHT: 'tl.constexpr', USE_DHO: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_DHT:
        p_dht = tl.make_block_ptr(dht + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))
    m_s = o_i[:, None] <= o_i[None, :]
    for i in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_d = tl.make_block_ptr(d + i_bh * s_qk_h, (DK, T), (s_qk_d, s_qk_t), (i_k * BK, i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        b_d = tl.load(p_d, boundary_check=(0, 1))
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv += tl.dot(b_k, b_dh, allow_tf32=False)
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        b_dh -= tl.dot(b_d, b_dv, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    if USE_DHO:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * DK * DV, (DK, DV), (DV, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh, boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(initial_state + i_bh * DK * DV, (DV, DK), (1, DV), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    NT = tl.cdiv(T, BT)
    for i in range(0, NT):
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        b_dv = tl.load(p_dv, boundary_check=(0, 1))
        b_dd = tl.dot(b_dv, b_h, allow_tf32=False)
        p_dd = tl.make_block_ptr(dd + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dd, -b_dd, boundary_check=(0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (DV, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, DV), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, DK), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16)], key=['BK'])
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk32(k, v, beta, w, u, A, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, T, K, V, BT: 'tl.constexpr', BK: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)
    b_A = -tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A, 0)
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BT) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
    b_A += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_A, b_A, boundary_check=(0, 1))
    b_A = b_A


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16)], key=['BK'])
@triton.jit
def fwd_prepare_wy_repr_kernel_chunk64(k, v, beta, w, u, A, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, T, K, V, BT: 'tl.constexpr', BK: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    b_A2 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A3 = tl.zeros([BC, BC], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BC,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_beta2 = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT + BC,), (BC,), (0,))
    b_beta2 = tl.load(p_beta2, boundary_check=(0,))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BC, BK), (1, 0))
        p_k2 = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT + BC, i_k * BK), (BC, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_kb2 = b_k2 * b_beta2[:, None]
        b_A += tl.dot(b_kb, tl.trans(b_k), allow_tf32=False)
        b_A2 += tl.dot(b_kb2, tl.trans(b_k2), allow_tf32=False)
        b_A3 += tl.dot(b_kb2, tl.trans(b_k), allow_tf32=False)
    b_A = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0)
    b_A2 = -tl.where(tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0)
    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = -tl.dot(tl.dot(b_A2, b_A3), b_A)
    p_A1 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BC, BC), (1, 0))
    p_A2 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, BC), (BC, BC), (1, 0))
    p_A3 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + BC, 0), (BC, BC), (1, 0))
    p_A4 = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, BC), (BC, BC), (1, 0))
    tl.store(p_A1, b_A, boundary_check=(0, 1))
    tl.store(p_A2, b_A2, boundary_check=(0, 1))
    tl.store(p_A3, b_A3, boundary_check=(0, 1))
    tl.store(p_A4, tl.zeros([BC, BC], dtype=tl.float32), boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def fwd_recompute_w_u_kernel(k, v, beta, w, u, A, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, T, K, V, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = b_v * b_beta[:, None]
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        p_u = tl.make_block_ptr(u + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_u, b_u, boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_w = tl.dot(b_A, b_kb, allow_tf32=False)
        p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_w, b_w, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK'])
@triton.jit
def fwd_recompute_w_kernel(k, beta, w, A, s_qk_h, s_qk_t, s_qk_d, T, K, BT: 'tl.constexpr', BK: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_w = tl.dot(b_A, b_kb, allow_tf32=False)
        p_w = tl.make_block_ptr(w + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_w, b_w, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16)], key=['BT', 'BK', 'BV'])
@triton.jit
def bwd_prepare_wy_repr_kernel(k, v, beta, A, dw, du, dk, dv, dbeta, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, T, K, V, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dbeta = tl.zeros([BT], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    p_beta = tl.make_block_ptr(beta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_beta = b_v * b_beta[:, None]
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_v_beta), allow_tf32=False)
        b_dv_beta = tl.dot(b_A, b_du, allow_tf32=False)
        b_dv = b_dv_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dv_beta * b_v, 1)
        p_dv = tl.make_block_ptr(dv + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_k_beta = b_k * b_beta[:, None]
        b_dw = tl.load(p_dw, boundary_check=(0, 1))
        b_dA += tl.dot(b_dw, tl.trans(b_k_beta), allow_tf32=False)
        b_dk_beta = tl.dot(b_A, b_dw, allow_tf32=False)
        b_dk = b_dk_beta * b_beta[:, None]
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_dA, 0)
    b_dA = tl.dot(b_dA, b_A)
    b_dA = tl.dot(b_A, b_dA)
    b_dA = tl.where(tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], -b_dA, 0)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dk = tl.load(p_dk, boundary_check=(0, 1))
        b_k_beta = b_k * b_beta[:, None]
        b_dk_beta = tl.dot(b_dA, b_k, allow_tf32=False)
        b_dbeta += tl.sum(b_dk_beta * b_k, 1)
        b_dk += tl.dot(tl.trans(b_dA), b_k_beta, allow_tf32=False)
        b_dk += b_dk_beta * b_beta[:, None]
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
    p_dbeta = tl.make_block_ptr(dbeta + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dbeta, b_dbeta, boundary_check=(0,))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BC', 'BK'])
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(q, k, g, A, s_k_h, s_k_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'BT'])
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(q, k, g, A, s_k_h, s_k_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr'):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_k = o_k < K
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0)
        b_gk = tl.load(p_gk, mask=m_k, other=0)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BC', 'BK'])
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(q, k, g, A, s_k_h, s_k_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * BT + i_i * BC >= T:
        return
    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_k = o_k < K
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0)
        b_gk = tl.load(p_gk, mask=m_k, other=0)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.0)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BC'])
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(A, A2, T: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', NK: 'tl.constexpr'):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_c * BC >= T:
        return
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_bh + i_k * n_bh) * T * BC, (T, BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_gla_fwd_kernel_o(q, v, g, h, o, A, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.0)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'NC', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_intra(q, k, g, dA, dq, dk, s_k_h, s_k_t, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * tl.exp(b_gn[None, :] - b_gk)
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    p_kj = tl.max_contiguous(tl.multiple_of(k + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gkj = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, mask=m_k, other=0)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0)
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_g - b_gkj[None, :]), 0.0)
        p_kj += K
        p_gkj += K
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC + BC - 1) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_qg = b_q * tl.exp(b_g - b_gn[None, :])
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= tl.exp(b_gn[None, :] - b_gk)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    p_qj = tl.max_contiguous(tl.multiple_of(q + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gqj = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_dA = tl.load(dA + o_dA + j * BT)
        b_qj = tl.load(p_qj, mask=m_k, other=0)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0)
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.0)
        p_qj += K
        p_gqj += K
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BV', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_dA(v, do, dA, s_v_h, s_v_t, scale, T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BV: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, b_v)
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.0)
    tl.store(p_dA, b_dA, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_dv(k, g, A, do, dh, dv, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0.0)
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = tl.dot(b_A, b_do, allow_tf32=False)
    last_idx = min(i_t * BT + BT, T) - 1
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + last_idx * K + o_k, BK), BK)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_gn = tl.exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = b_k * b_gn
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh)
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_gla_bwd_kernel_inter(q, k, v, h, g, do, dh, dq, dk, dq2, dk2, dg, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    last_idx = min(T, i_t * BT + BT) - 1
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bh * s_k_h + last_idx * K + o_k, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * V * K, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, b_dh)
    b_dgk *= tl.exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gn = tl.exp(b_gn[None, :] - b_gk)
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * b_gn
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2 + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))


@triton.jit
def prepare_qg_kg(q, k, g, qg, kg, s_qk_h, scale, K: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_g = g + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_qg = qg + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    p_kg = kg + i_bh * s_qk_h + i_c * BT * K + i_k * BK + tl.arange(0, BK)
    mask = i_k * BK + tl.arange(0, BK) < K
    last_decay = tl.load(g + i_bh * s_qk_h + (i_c * BT + BT - 1) * K + i_k * BK + tl.arange(0, BK))
    for i in range(BT):
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        _g = tl.load(p_g, mask=mask, other=0)
        b_q *= tl.exp(_g) * scale
        b_k *= tl.exp(last_decay - _g)
        tl.store(p_kg, b_k, mask=mask)
        tl.store(p_qg, b_q, mask=mask)
        p_q += K
        p_g += K
        p_k += K
        p_kg += K
        p_qg += K


@triton.jit
def bwd_decay_global_cumsum(dq_inner, dq_inter, dk_inner, dk_inter, q, k, g, dg, s_qk_h, BT: 'tl.constexpr', BK: 'tl.constexpr', K: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_g = g + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dg = dg + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inner = dq_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inner = dk_inner + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dq_inter = dq_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    p_dk_inter = dk_inter + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (i_c * BT + BT - 1) * K
    cum_grad_dg = tl.zeros([BK], dtype=tl.float32)
    mask = i_k * BK + tl.arange(0, BK) < K
    last_g = tl.zeros([BK], dtype=tl.float32)
    for j in range(BT - 1, -1, -1):
        _g = tl.load(p_g, mask=mask, other=0)
        if j == BT - 1:
            last_g = _g
        b_dq1 = tl.load(p_dq_inner, mask=mask, other=0)
        b_dq2 = tl.load(p_dq_inter, mask=mask, other=0)
        b_dq2 *= tl.exp(_g)
        b_dq = b_dq1 + b_dq2
        tl.store(p_dq_inter, b_dq, mask=mask)
        b_dk1 = tl.load(p_dk_inner, mask=mask, other=0)
        b_dk2 = tl.load(p_dk_inter, mask=mask, other=0)
        b_dk2 *= tl.exp(last_g - _g)
        b_dk = b_dk1 + b_dk2
        tl.store(p_dk_inter, b_dk, mask=mask)
        b_q = tl.load(p_q, mask=mask, other=0)
        b_k = tl.load(p_k, mask=mask, other=0)
        b_dg = b_dq * b_q - b_dk * b_k
        cum_grad_dg += b_dg
        tl.store(p_dg, cum_grad_dg, mask=mask)
        p_g -= K
        p_k -= K
        p_q -= K
        p_dq_inner -= K
        p_dk_inner -= K
        p_dq_inter -= K
        p_dk_inter -= K
        p_dg -= K


@triton.jit
def fused_chunk_gla_fwd_kernel(q, k, v, g, o, h0, ht, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_db = g + i_bh * s_qk_h + (BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    mask = i_k * BK + tl.arange(0, BK) < K
    for i in range(0, tl.cdiv(T, BT)):
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        d_b = tl.load(p_db, mask=mask, other=0)
        if CHECK and i == 0:
            b_o = tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h * tl.exp(d_b)[:, None] + tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o = tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h * tl.exp(d_b)[:, None] + tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
        p_db += BT * K
    if STORE_FINAL_STATE:
        p_final = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_final, b_h, boundary_check=(0, 1))


@triton.jit
def fused_chunk_gla_bwd_kernel(q, k, v, g, do, dq, dk, dv, h0, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h += tl.load(p_h, boundary_check=(0, 1))
    mask = i_k * BK + tl.arange(0, BK) < K
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + ((i + 1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_k = tl.load(p_k, boundary_check=(0, 1))
        d_b = tl.load(p_db, mask=mask, other=0)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h * tl.exp(d_b)[None, :] + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h * tl.exp(d_b)[None, :] + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_db = g + i_bh * s_qk_h + (T - (i - 1) * BT - 1) * s_qk_t + i_k * BK + tl.arange(0, BK)
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_db = tl.load(p_db, mask=mask, other=0)
        if CHECK and i == 1:
            b_dk = tl.trans(tl.dot(b_dh, tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot(b_k, b_dh, allow_tf32=False)
            b_dh = b_dh * tl.exp(b_db)[:, None] + tl.dot(b_q, b_do, allow_tf32=False)
        else:
            b_dk = tl.trans(tl.dot(b_dh, tl.trans(b_v), allow_tf32=False))
            b_dv = tl.dot(b_k, b_dh, allow_tf32=False)
            b_dh = b_dh * tl.exp(b_db)[:, None] + tl.dot(b_q, b_do, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.jit
def fwd_inner_chunk(q, k, g, A, s_qk_h, s_qk_t, s_qk_d, B, H, T, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', K: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    mask = i_k * BK + tl.arange(0, BK) < K
    o_i = tl.arange(0, BT)
    p_q = q + i_bh * s_qk_h + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_gq = g + i_bh * s_qk_h + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_A = A + (i_bh + i_k * B * H) * (tl.cdiv(T, BT) * BT * BT) + i_t * BT * BT + tl.arange(0, BT)
    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0) * scale
        gq = tl.load(p_gq, mask=mask, other=0)
        s = _q[None, :] * b_k * tl.exp(gq[None, :] - b_g)
        score = tl.sum(s, axis=1)
        score = tl.where(o_i <= i, score, 0)
        tl.store(p_A, score)
        p_q += K
        p_gq += K
        p_A += BT


@triton.jit
def bwd_inner_chunk(q, k, g, dA, dq, dk, s_qk_h, s_qk_t, s_qk_d, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    p_g = tl.make_block_ptr(g + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    mask = i_k * BK + tl.arange(0, BK) < K
    o_i = tl.arange(0, BT)
    p_q = q + i_bh * s_qk_h + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_dq = dq + i_bh * s_qk_h + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_gq = g + i_bh * s_qk_h + i_k * BK + i_t * BT * K + tl.arange(0, BK)
    p_dA = dA + i_bh * (tl.cdiv(T, BT) * BT * BT) + i_t * BT * BT + tl.arange(0, BT)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i in range(BT):
        _q = tl.load(p_q, mask=mask, other=0)
        gq = tl.load(p_gq, mask=mask, other=0)
        score = tl.exp(gq[None, :] - b_g)
        score = tl.where(o_i[:, None] <= i, score, 0)
        _dA = tl.load(p_dA)
        _dA = tl.where(o_i <= i, _dA, 0)
        b_dk += _dA[:, None] * score * _q[None, :]
        b_dq = tl.sum(_dA[:, None] * score * b_k, axis=0)
        tl.store(p_dq, b_dq, mask=mask)
        p_q += K
        p_dq += K
        p_gq += K
        p_dA += BT
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_kernel_intra_K(v, g, o, A, s_v_h, s_v_t, T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)
    p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
    b_gn = tl.load(p_gn, mask=o_v < V, other=0)
    b_o = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_gv = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = b_v * tl.exp(b_gn[None, :] - b_gv)
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_o += tl.dot(b_A, b_vg)
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o *= tl.exp(b_g - b_gn[None, :])
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    for j in range(0, BC):
        p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * s_v_h + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
        p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC + j) * V + o_v, BV), BV)
        b_A = tl.load(A + o_A + j, mask=m_A, other=0)
        b_v = tl.load(p_v, mask=o_v < V, other=0)
        b_gv = tl.load(p_gv, mask=o_v < V, other=0)
        b_vg = b_v[None, :] * tl.exp(b_g - b_gv[None, :])
        b_o += tl.where(o_i[:, None] >= j, b_A[:, None] * b_vg, 0.0)
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    b_o += tl.load(p_o, boundary_check=(0, 1))
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_kernel_K(q, k, h, g, o, A, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NG: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)
    p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_o = b_o * tl.exp(b_g)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.where(m_s, b_A, 0.0)
    if i_v == 0:
        tl.store(p_A, b_A, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_kernel_intra_Vk(q, k, g, A, s_k_h, s_k_t, i_k, i_c, i_bh, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr'):
    i_bg = i_bh // NG
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC) % NC
    o_k = i_k * BK + tl.arange(0, BK)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    b_A = tl.zeros([BC, BC], tl.float32)
    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
        b_gn = tl.load(p_gn, mask=o_k < K, other=0.0)
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A = tl.dot(b_qg, b_kg)
        if i_k != 0:
            b_A += tl.load(p_A, boundary_check=(0, 1))
        tl.store(p_A, b_A, boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bg * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_j * BC) * K + o_k, BK), BK)
        m_k = o_k < K
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        o_i = tl.arange(0, BC)
        m_A = o_i[:, None] >= o_i[None, :]
        for j in range(0, BC):
            b_k = tl.load(p_k, mask=m_k & i_t * BT + i_j * BC + j < T, other=0.0)
            b_gk = tl.load(p_gk, mask=m_k & i_t * BT + i_j * BC + j < T, other=0.0)
            b_Aj = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]) * scale, 1)
            b_A = tl.where((o_i == j)[None, :], b_Aj[:, None], b_A)
            p_k += K
            p_gk += K
        b_A = tl.where(m_A, b_A, 0.0)
        if i_k != 0:
            b_A += tl.load(p_A, boundary_check=(0, 1))
        tl.store(p_A, b_A, boundary_check=(0, 1))
    elif i_k == 0:
        tl.store(p_A, b_A, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_fwd_kernel_intra_V(q, k, g, A, s_k_h, s_k_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr', NK: 'tl.constexpr', NG: 'tl.constexpr'):
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    for i_k in range(0, NK):
        chunk_gsa_fwd_kernel_intra_Vk(q, k, g, A, s_k_h, s_k_t, i_k, i_c, i_bh, scale, T, K, BT, BC, BK, NC, NG)


@triton.jit
def chunk_gsa_fwd_kernel_V(q, v, g, h, o, A, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NG: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h)
    p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_V(k, v, h, g, A, do, dh, dq, dk, dv, dA, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NG: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_t = min(i_t * BT + BT, T)
    o_k = i_k * BK + tl.arange(0, BK)
    p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (o_t - 1) * K + o_k, BK), BK)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, i_t * BT), (BT, BT), (0, 1))
    m_k = o_k < K
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gn = tl.exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
    b_k = b_k * b_gn
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * V * K, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh = b_dh
        b_dv = tl.dot(b_k, b_dh)
        if i_k == 0:
            b_dv += tl.dot(b_A, b_do)
        b_do = b_do * scale
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
        b_dA += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, tl.trans(b_dh))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * b_gn
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_dA = tl.where(m_s, b_dA, 0.0)
    if i_k == 0:
        tl.store(p_dA, b_dA, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_intra_V(q, k, g, dA, dq, dk, dg, s_k_h, s_k_t, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr', OVERWRITE: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_k = i_k * BK + tl.arange(0, BK)
    p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    m_k = o_k < K
    b_gn = tl.load(p_gn, mask=m_k, other=0.0)
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[None, :] - b_gk)
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        b_dq += tl.dot(b_dA, b_kg)
    b_dq *= tl.exp(b_g - b_gn[None, :])
    p_kj = tl.max_contiguous(tl.multiple_of(k + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gkj = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    m_k = o_k < K
    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    for j in range(0, BC):
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, mask=m_k, other=0)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0)
        m_i = o_i[:, None] >= j
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tl.exp(b_g - b_gkj[None, :]), 0.0)
        p_kj += K
        p_gkj += K
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_dq = b_dq + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC + BC - 1) * K + o_k, BK), BK)
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g - b_gn[None, :])
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        b_dk += tl.dot(tl.trans(b_dA), b_qg)
    b_dk *= tl.exp(b_gn[None, :] - b_gk)
    p_qj = tl.max_contiguous(tl.multiple_of(q + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    p_gqj = tl.max_contiguous(tl.multiple_of(g + i_bg * s_k_h + (i_t * BT + i_i * BC) * K + o_k, BK), BK)
    m_k = o_k < K
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        b_dA = tl.load(dA + o_dA + j * BT, mask=i_t * BT + i_i * BC + j < T, other=0)
        b_qj = tl.load(p_qj, mask=m_k, other=0.0)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0.0)
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.0)
        p_qj += K
        p_gqj += K
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_dk = b_dk + tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    if not OVERWRITE:
        b_dg = b_dg + tl.load(p_dg, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_intra_K(v, g, do, dA, s_v_h, s_v_t, scale, T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i, i_j = i_c // (NC * NC), i_c % (NC * NC) // NC, i_c % (NC * NC) % NC
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    p_dA = tl.make_block_ptr(dA + (i_bh + i_v * n_bh) * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    b_dA = tl.zeros([BC, BC], dtype=tl.float32)
    if i_i > i_j:
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gv = tl.make_block_ptr(g + i_bg * s_v_h, (V, T), (1, s_v_t), (i_v * BV, i_t * BT + i_j * BC), (BV, BC), (0, 1))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        b_gn = tl.load(p_gn, mask=m_v, other=0.0)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_g - b_gn[None, :]) * scale
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_gv = tl.load(p_gv, boundary_check=(0, 1))
        b_vg = b_v * tl.exp(b_gn[:, None] - b_gv)
        b_dA = tl.dot(b_do, b_vg)
    elif i_i == i_j:
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
        p_v = tl.max_contiguous(tl.multiple_of(v + i_bg * s_v_h + (i_t * BT + i_j * BC) * V + o_v, BV), BV)
        p_gv = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_j * BC) * V + o_v, BV), BV)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1)) * scale
        m_v = o_v < V
        o_i = tl.arange(0, BC)
        m_dA = o_i[:, None] >= o_i[None, :]
        for j in range(0, BC):
            b_v = tl.load(p_v, mask=m_v, other=0)
            b_gv = tl.load(p_gv, mask=m_v, other=0)
            b_dAj = tl.sum(b_do * b_v[None, :] * tl.exp(b_g - b_gv[None, :]), 1)
            b_dA = tl.where((o_i == j)[None, :], b_dAj[:, None], b_dA)
            p_v += V
            p_gv += V
        b_dA = tl.where(m_dA, b_dA, 0.0)
    tl.store(p_dA, b_dA, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_K(q, k, v, h, g, A, do, dh, dq, dk, dv, dA, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NG: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)
    o_t = min(i_t * BT + BT, T)
    m_s = o_i[:, None] >= o_i[None, :]
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bg * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_A = tl.dot(b_q * scale, tl.trans(b_k))
    b_A = tl.where(m_s, b_A, 0.0)
    tl.store(p_A, b_A, boundary_check=(0, 1))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bg * s_h_h + i_t * K * V, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_gn = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (o_t - 1) * V + o_v, BV), BV)
        m_v = o_v < V
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_gn = tl.load(p_gn, mask=m_v, other=0)
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_v = b_v * tl.exp(b_gn[None, :] - b_g)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_h = b_h
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_g) * scale
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh = b_dh
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, tl.trans(b_dh))
        b_dv = tl.exp(b_gn[None, :] - b_g) * tl.dot(b_k, b_dh)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.load(p_dA, boundary_check=(0, 1))
    b_dq += tl.dot(b_dA, b_k)
    b_dk += tl.dot(tl.trans(b_dA), b_q)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.jit
def chunk_gsa_bwd_kernel_intra_KV(v, g, o, A, do, dv, dg, s_v_h, s_v_t, T: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BV: 'tl.constexpr', NC: 'tl.constexpr', NG: 'tl.constexpr', OVERWRITE: 'tl.constexpr'):
    i_v, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    i_t, i_i = i_c // NC, i_c % NC
    o_v = i_v * BV + tl.arange(0, BV)
    p_gv = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_gn = g + i_bg * s_v_h + (i_t * BT + i_i * BC + BC - 1) * V + o_v
    p_gn = tl.max_contiguous(tl.multiple_of(p_gn, BV), BV)
    m_v = o_v < V
    b_gn = tl.load(p_gn, mask=m_v, other=0)
    b_gv = tl.load(p_gv, boundary_check=(0, 1))
    b_dv = tl.zeros([BC, BV], dtype=tl.float32)
    for i_j in range(i_i + 1, NC):
        p_g = tl.make_block_ptr(g + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_j * BC, i_v * BV), (BC, BV), (1, 0))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_do = b_do * tl.exp(b_g - b_gn[None, :])
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dv += tl.dot(b_A, b_do)
    b_dv *= tl.exp(b_gn[None, :] - b_gv)
    o_i = tl.arange(0, BC)
    o_c = i_i * BC + tl.arange(0, BC)
    p_g = tl.max_contiguous(tl.multiple_of(g + i_bg * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
    p_A = tl.max_contiguous(tl.multiple_of(A + i_bh * T * BT + (i_t * BT + i_i * BC) * BT + o_c, BV), BV)
    p_do = tl.max_contiguous(tl.multiple_of(do + i_bh * s_v_h + (i_t * BT + i_i * BC) * V + o_v, BV), BV)
    for j in range(0, BC):
        m_j = i_t * BT + i_i * BC + j < T
        b_A = tl.load(p_A, mask=m_j, other=0)
        b_g = tl.load(p_g, mask=m_j & m_v, other=0)
        b_do = tl.load(p_do, mask=m_j & m_v, other=0)
        m_i = o_i[:, None] <= j
        b_dv += tl.where(m_i, tl.exp(b_g[None, :] - b_gv) * b_A[:, None] * b_do[None, :], 0.0)
        p_g += V
        p_A += BT
        p_do += V
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_v = tl.make_block_ptr(v + i_bg * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT + i_i * BC, i_v * BV), (BC, BV), (1, 0))
    b_o = tl.load(p_o, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = b_dv + tl.load(p_dv, boundary_check=(0, 1))
    b_dg = b_o * b_do - b_v * b_dv
    if not OVERWRITE:
        b_dg = b_dg + tl.load(p_dg, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))


@triton.jit
def fused_recurrent_gsa_inference_kernel(q, k, v, s, g, o, hk0, hv0, hkt, hvt, scale, K: 'tl.constexpr', V: 'tl.constexpr', M: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NG: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    i_bg = i_bh // NG
    b_s = tl.load(s + i_bg * M + tl.arange(0, M))
    b_g = tl.load(g + i_bg * M + tl.arange(0, M))
    b_g = tl.exp(b_g)
    b_ok = tl.zeros([M], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_hk0 = hk0 + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None]
        mask_k = o_k < K
        mask_hk = (tl.arange(0, M) < M)[:, None] & mask_k[None, :]
        b_hk = tl.load(p_hk0, mask=mask_hk, other=0.0)
        b_q = tl.load(q + i_bh * K + o_k, mask=mask_k, other=0.0) * scale
        b_k = tl.load(k + i_bg * K + o_k, mask=mask_k, other=0.0)
        b_hk = b_hk * b_g[:, None] + b_k[None, :] * b_s[:, None]
        b_ok += tl.sum(b_hk * b_q[None, :], axis=1)
        if i_bh % NG == 0:
            p_hkt = hkt + i_bg * K * M + o_k[None, :] * M + tl.arange(0, M)[:, None]
            tl.store(p_hkt, b_hk, mask=mask_hk)
    b_qv = tl.softmax(b_ok)
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        p_hv0 = hv0 + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
        mask_v = o_v < V
        mask_hv = mask_v[:, None] & (tl.arange(0, M) < M)[None, :]
        b_hv = tl.load(p_hv0, mask=mask_hv, other=0)
        b_v = tl.load(v + i_bg * V + o_v, mask=mask_v, other=0)
        b_hv = b_hv * b_g[None, :] + b_s[None, :] * b_v[:, None]
        b_ov = tl.sum(b_hv * b_qv[None, :], axis=1)
        tl.store(o + i_bh * V + o_v, b_ov, mask=mask_v)
        if i_bh % NG == 0:
            p_hvt = hvt + i_bg * M * V + tl.arange(0, M)[None, :] * V + o_v[:, None]
            tl.store(p_hvt, b_hv, mask=mask_hv)


@triton.jit
def fused_recurrent_gsa_fwd_kernel(q, k, v, gk, gv, o, h0, ht, s_k_h, s_v_h, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    mask_h = mask_k[None, :] & mask_v[:, None]
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0)
    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0)
            b_h = b_h * tl.exp(b_gk)[None, :]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0)
            b_h = b_h * tl.exp(b_gv)[:, None]
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o, mask=mask_v)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h, mask=mask_h)


@triton.jit
def fused_recurrent_gsa_bwd_kernel(q, k, v, gk, gv, do, dq, dk, dv, dh0, h0, s_k_h, s_v_h, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr', USE_GK: 'tl.constexpr', USE_GV: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    mask_k = i_k * BK + tl.arange(0, BK) < K
    mask_v = i_v * BV + tl.arange(0, BV) < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_h, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_do = tl.load(p_do, mask=mask_v, other=0)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0)
            b_h = b_h * tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0)
            b_h = b_h * tl.exp(b_gv)[None, :]
        b_h += b_k[:, None] * b_v[None, :]
        b_dq = tl.sum(b_h * b_do[None, :], axis=1) * scale
        tl.store(p_dq, b_dq, mask=mask_k)
        p_k += -K if REVERSE else K
        p_v += -V if REVERSE else V
        p_q += -K if REVERSE else K
        p_do += -V if REVERSE else V
        p_dq += -K if REVERSE else K
        if USE_GK:
            p_gk += -K if REVERSE else K
        if USE_GV:
            p_gv += -V if REVERSE else V
    tl.debug_barrier()
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    if USE_GK:
        p_gk = gk + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    if USE_GV:
        p_gv = gv + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0)
        b_v = tl.load(p_v, mask=mask_v, other=0)
        b_do = tl.load(p_do, mask=mask_v, other=0)
        b_dh += b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        b_dv = tl.sum(b_dh * b_k[:, None], axis=0)
        if USE_GK:
            b_gk = tl.load(p_gk, mask=mask_k, other=0)
            b_dh *= tl.exp(b_gk)[:, None]
        if USE_GV:
            b_gv = tl.load(p_gv, mask=mask_v, other=0)
            b_dh *= tl.exp(b_gv)[None, :]
        tl.store(p_dk, b_dk, mask=mask_k)
        tl.store(p_dv, b_dv, mask=mask_v)
        p_q += K if REVERSE else -K
        p_k += K if REVERSE else -K
        p_v += V if REVERSE else -V
        p_do += V if REVERSE else -V
        p_dk += K if REVERSE else -K
        p_dv += V if REVERSE else -V
        if USE_GK:
            p_gk += K if REVERSE else -K
        if USE_GV:
            p_gv += V if REVERSE else -V
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh, mask=mask_h)


@triton.autotune(configs=[triton.Config({'BD': 32}, num_warps=1), triton.Config({'BD': 32}, num_warps=2), triton.Config({'BD': 32}, num_warps=4), triton.Config({'BD': 32}, num_warps=8), triton.Config({'BD': 64}, num_warps=1), triton.Config({'BD': 64}, num_warps=2), triton.Config({'BD': 64}, num_warps=4), triton.Config({'BD': 64}, num_warps=8), triton.Config({'BD': 128}, num_warps=1), triton.Config({'BD': 128}, num_warps=2), triton.Config({'BD': 128}, num_warps=4), triton.Config({'BD': 128}, num_warps=8)], key=['D'])
@triton.jit
def chunk_hgrn_fwd_kernel_h(x, g, gc, o, h0, T: 'tl.constexpr', D: 'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr'):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_x = x + i_b * T * D + i_t * BT * D + o_d
    p_g = g + i_b * T * D + i_t * BT * D + o_d
    p_gc = gc + i_b * T * D + i_t * BT * D + o_d
    p_o = o + i_b * T * D + i_t * BT * D + o_d
    b_h = tl.zeros([BD], dtype=tl.float32)
    b_gc = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if i_t == 0:
            b_h += tl.load(h0 + i_b * D + o_d, mask=mask, other=0)
    for i in range(0, BT):
        mask_t = mask & (i_t * BT + i < T)
        b_x = tl.load(p_x, mask=mask_t, other=0)
        b_g = tl.load(p_g, mask=mask_t, other=0)
        b_h = tl.exp(b_g) * b_h + b_x
        b_gc = b_gc + b_g
        tl.store(p_gc, b_gc, mask=mask_t)
        tl.store(p_o, b_h, mask=mask_t)
        p_x += D
        p_g += D
        p_gc += D
        p_o += D


@triton.jit
def chunk_hgrn_fwd_kernel_o(gc, o, s_b, s_t, s_d, T: 'tl.constexpr', D: 'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    for i_t in range(1, tl.cdiv(T, BT)):
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_h0 = tl.load(o + i_b * T * D + i_t * BT * D - D + o_d, mask=mask, other=0)
        b_gc = tl.load(p_gc, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))
        b_o = b_o + tl.exp(b_gc) * b_h0[None, :]
        tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({'BD': 32}, num_warps=1), triton.Config({'BD': 32}, num_warps=2), triton.Config({'BD': 32}, num_warps=4), triton.Config({'BD': 32}, num_warps=8), triton.Config({'BD': 64}, num_warps=1), triton.Config({'BD': 64}, num_warps=2), triton.Config({'BD': 64}, num_warps=4), triton.Config({'BD': 64}, num_warps=8), triton.Config({'BD': 128}, num_warps=1), triton.Config({'BD': 128}, num_warps=2), triton.Config({'BD': 128}, num_warps=4), triton.Config({'BD': 128}, num_warps=8)], key=['D'])
@triton.jit
def chunk_hgrn_bwd_kernel_h(g, gc, dx, do, T: 'tl.constexpr', D: 'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr'):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    BC = min(BT, T - i_t * BT)
    NT = tl.num_programs(1)
    p_g = g + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_gc = gc + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_dx = dx + (i_b * T + i_t * BT + BC - 1) * D + o_d
    p_do = do + (i_b * T + i_t * BT + BC - 1) * D + o_d
    if i_t == NT - 1:
        b_gc = tl.zeros([BD], dtype=tl.float32)
    else:
        b_gc = tl.load(g + (i_b * T + i_t * BT + BT) * D + o_d, mask=mask, other=0)
    b_dh = tl.zeros([BD], dtype=tl.float32)
    for _ in range(BC - 1, -1, -1):
        tl.store(p_gc, b_gc, mask=mask)
        b_g = tl.load(p_g, mask=mask, other=0)
        b_do = tl.load(p_do, mask=mask, other=0)
        b_gc = b_gc + b_g
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * tl.exp(b_g)
        tl.store(p_dx, b_dx, mask=mask)
        p_g -= D
        p_gc -= D
        p_dx -= D
        p_do -= D


@triton.jit
def chunk_hgrn_bwd_kernel_o(g, gc, o, dx, dg, s_b, s_t, s_d, T: 'tl.constexpr', D: 'tl.constexpr', BT: 'tl.constexpr', BD: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_g = tl.make_block_ptr(g + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_gc = tl.make_block_ptr(gc + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT - 1, i_d * BD), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg + i_b * s_b, (T, D), (s_t, s_d), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        mask_t = mask & ((i_t + 1) * BT < T)
        b_ht = tl.load(dx + i_b * T * D + (i_t + 1) * BT * D + o_d, mask=mask_t, other=0)
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_gc = tl.load(p_gc, boundary_check=(0, 1))
        b_o = tl.load(p_o, boundary_check=(0, 1))
        b_dx = tl.load(p_dx, boundary_check=(0, 1))
        b_dx = b_dx + tl.exp(b_gc) * b_ht[None, :]
        b_dg = b_o * b_dx * tl.exp(b_g)
        tl.store(p_dx, b_dx, boundary_check=(0, 1))
        tl.store(p_dg, b_dg, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({'BD': 32}, num_warps=1), triton.Config({'BD': 32}, num_warps=2), triton.Config({'BD': 32}, num_warps=4), triton.Config({'BD': 32}, num_warps=8), triton.Config({'BD': 64}, num_warps=1), triton.Config({'BD': 64}, num_warps=2), triton.Config({'BD': 64}, num_warps=4), triton.Config({'BD': 64}, num_warps=8), triton.Config({'BD': 128}, num_warps=1), triton.Config({'BD': 128}, num_warps=2), triton.Config({'BD': 128}, num_warps=4), triton.Config({'BD': 128}, num_warps=8)], key=['D'])
@triton.jit
def fused_recurrent_hgrn_fwd_kernel(x, g, o, h0, ht, T: 'tl.constexpr', D: 'tl.constexpr', BD: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_x = x + i_b * T * D + o_d
    p_g = g + i_b * T * D + o_d
    p_o = o + i_b * T * D + o_d
    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_b * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0)
        b_g = tl.load(p_g, mask=mask, other=0)
        b_h = tl.exp(b_g) * b_h + b_x
        tl.store(p_o, b_h, mask=mask)
        p_x += D
        p_g += D
        p_o += D
    if STORE_FINAL_STATE:
        p_ht = ht + i_b * D + o_d
        tl.store(p_ht, b_h, mask=mask)


@triton.autotune(configs=[triton.Config({'BD': 32}, num_warps=1), triton.Config({'BD': 32}, num_warps=2), triton.Config({'BD': 32}, num_warps=4), triton.Config({'BD': 32}, num_warps=8), triton.Config({'BD': 64}, num_warps=1), triton.Config({'BD': 64}, num_warps=2), triton.Config({'BD': 64}, num_warps=4), triton.Config({'BD': 64}, num_warps=8), triton.Config({'BD': 128}, num_warps=1), triton.Config({'BD': 128}, num_warps=2), triton.Config({'BD': 128}, num_warps=4), triton.Config({'BD': 128}, num_warps=8)], key=['D'])
@triton.jit
def fused_recurrent_hgrn_bwd_kernel(g, o, dx, dg, do, h0, T: 'tl.constexpr', D: 'tl.constexpr', BD: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr'):
    i_d, i_b = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D
    p_g = g + (i_b * T + T - 1) * D + o_d
    p_o = o + (i_b * T + T - 2) * D + o_d
    p_dx = dx + (i_b * T + T - 1) * D + o_d
    p_dg = dg + (i_b * T + T - 1) * D + o_d
    p_do = do + (i_b * T + T - 1) * D + o_d
    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(T - 1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0)
        b_do = tl.load(p_do, mask=mask, other=0)
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_b * D + o_d, mask=mask, other=0)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)
        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dh = b_dh * tl.exp(b_g)
        b_dg = b_dh * b_o
        tl.store(p_dx, b_dx, mask=mask)
        tl.store(p_dg, b_dg, mask=mask)
        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D


@triton.jit
def chunk_linear_attn_fwd_kernel_h(k, v, h, h0, ht, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_h += tl.dot(b_k, b_v, allow_tf32=False)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.jit
def chunk_linear_attn_bwd_kernel_dh(q, do, dh, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh += tl.dot(b_q, b_do, allow_tf32=False)


@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(q, k, v, h, do, dh, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
    b_s = tl.where(o_i[:, None] <= o_i[None, :], b_s, 0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False) * scale
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot(b_s, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale, 0)
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.jit
def fused_chunk_linear_attn_fwd_kernel(q, k, v, o, h0, ht, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_o = tl.dot(b_s, b_v, allow_tf32=False)
        if CHECK and i == 0:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.jit
def fused_chunk_linear_attn_bwd_kernel(q, k, v, do, dq, dk, dv, h0, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_dq = tl.dot(b_ds, b_k, allow_tf32=False)
        if CHECK and i == 0:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        else:
            b_dq += tl.dot(b_do, b_h, allow_tf32=False)
            b_h = b_h + tl.dot(b_v, b_k, allow_tf32=False)
        b_dq *= scale
        tl.store(p_dq, b_dq, boundary_check=(0, 1))
    b_h = None
    tl.debug_barrier()
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    m_s = o_i[:, None] <= o_i[None, :]
    for i in range(1, tl.cdiv(T, BT) + 1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_s = tl.dot(b_k, b_q, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        b_ds = tl.dot(b_v, tl.trans(b_do), allow_tf32=False)
        b_ds = tl.where(m_s, b_ds, 0)
        b_dk = tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv = tl.dot(b_s, b_do, allow_tf32=False)
        if CHECK and i == 1:
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh, allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        else:
            b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
            b_dv += tl.dot(b_k, b_dh, allow_tf32=False)
            b_dh += tl.dot(b_q, b_do, allow_tf32=False)
        tl.store(p_dk, b_dk, boundary_check=(0, 1))
        tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.jit
def fused_recurrent_linear_attn_fwd_kernel(q, k, v, o, h0, ht, s_qk_h, s_vo_h, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_h += b_k[None, :] * b_v[:, None]
        b_o = b_h * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        tl.store(p_o, b_o, mask=mask_bv)
        p_q += K
        p_k += K
        p_o += V
        p_v += V
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h, mask=mask_kv)


@triton.jit
def fused_recurrent_linear_attn_bwd_kernel(q, k, v, do, dq, dk, dv, h0, s_qk_h, s_vo_h, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        b_h += tl.load(p_h0, mask=mask_kv, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_h += b_k[:, None] * b_v[None, :]
        _d_q = b_h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q, mask=mask_bk)
        p_k += K
        p_do += V
        p_v += V
        p_dq += K
    tl.debug_barrier()
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    for _ in range(T):
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * b_v[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)
        tl.store(p_dk, d_k, mask=mask_bk)
        tl.store(p_dv, d_v, mask=mask_bv)
        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V


@triton.jit
def parallel_rebased_fwd_kernel(q, k, v, o, z, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B, H, T, K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
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
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
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
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    tl.store(p_z, b_z, mask=i_c * BTL + tl.arange(0, BTL) < T)


@triton.jit
def _parallel_rebased_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q = b_q * scale
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, 0), (BV, BTS), (0, 1))
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
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
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
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    return


@triton.jit
def _parallel_rebased_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32)
    for i in range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
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
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
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
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    return


@triton.jit
def parallel_rebased_bwd_kernel(q, k, v, do, dz, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV)
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_retention_fwd_kernel_h(k, v, h, h0, ht, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1))
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        if i_t == NT - 1 and T % BT != 0:
            d_b = tl.math.exp2(T % BT * b_b)
            d_i = tl.math.exp2((T % BT - o_i - 1) * b_b)
        b_h = d_b * b_h + tl.dot(b_k, b_v * d_i[:, None], allow_tf32=False)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_retention_fwd_kernel_o(q, k, v, h, o, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q * d_i[:, None], b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s *= d_s
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s, b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_retention_bwd_kernel_dh(q, do, dh, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_dh = d_b * b_dh + tl.dot(b_q, b_do * d_i[:, None], allow_tf32=False)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_retention_bwd_kernel_dqkv(q, k, v, h, do, dh, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    n_bh = tl.num_programs(2)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    d_q = d_q * scale
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s, b_do, allow_tf32=False)
        tl.store(p_dv, b_dv, boundary_check=(0, 1))
    b_ds = b_ds * d_s
    b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.jit
def fused_chunk_retention_fwd_kernel(q, k, v, o, h0, ht, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    d_b, d_o, d_h = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BT), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BT, BV), (1, 0))
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h, boundary_check=(0, 1))
    NT = tl.cdiv(T, BT)
    for i in range(0, NT):
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
            if i == NT - 1 and T % BT != 0:
                d_b = tl.math.exp2(T % BT * b_b)
                d_h = tl.math.exp2((T % BT - o_i - 1) * b_b)
            b_h = d_b * b_h + tl.dot(b_k, b_v * d_h[:, None], allow_tf32=False)
        tl.store(p_o, b_o, boundary_check=(0, 1))
        p_q = tl.advance(p_q, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        p_o = tl.advance(p_o, (BT, 0))
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h, boundary_check=(0, 1))


@triton.jit
def fused_chunk_retention_bwd_kernel(q, k, v, do, dq, dk, dv, h0, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', CHECK: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    o_i = tl.arange(0, BT)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b) * scale, tl.math.exp2((BT - o_i - 1) * b_b)
    d_b = tl.math.exp2(BT * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h = tl.make_block_ptr(h0 + i_bh * K * V, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
    for i in range(0, tl.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dq = tl.make_block_ptr(dq + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i * BT, i_k * BK), (BT, BK), (1, 0))
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
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, T - i * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
        p_dk = tl.make_block_ptr(dk + (i_bh + i_v * B * H) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (T - i * BT, i_k * BK), (BT, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_bh + i_k * B * H) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (T - i * BT, i_v * BV), (BT, BV), (1, 0))
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
def fused_recurrent_retention_fwd_kernel(q, k, v, o, initial_state, final_state, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, B, H, T, scale, BK: 'tl.constexpr', BV: 'tl.constexpr', DK: 'tl.constexpr', DV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
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
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
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
def parallel_retention_fwd_kernel(q, k, v, o, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_k = tl.arange(0, BTS)
    d_h = tl.math.exp2((BTS - o_k) * b_b)
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))
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
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
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
    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.jit
def _parallel_retention_bwd_dq(i_bh, i_c, i_k, i_v, i_h, k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
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
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
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
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    return


@triton.jit
def _parallel_retention_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    d_b = tl.math.exp2(b_b * BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1)), tl.load(p_v, boundary_check=(0, 1))
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros([BTL, BV], dtype=tl.float32)
    d_h = tl.math.exp2((BTL - tl.arange(0, BTL)) * b_b)
    b_kd = b_k * d_h[:, None]
    d_q = tl.math.exp2(tl.arange(0, BTS) * b_b)
    for i in range(tl.cdiv(T, BTS) * BTS - BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
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
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (V, T), (s_vo_d, s_vo_t), (i_v * BV, i), (BV, BTS), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        m_s = o_k[:, None] <= o_q[None, :]
        d_s = tl.where(m_s, tl.math.exp2((-o_k[:, None] + o_q[None, :]) * b_b), 0) * scale
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * d_s
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * d_s
        b_dk += tl.dot(b_ds, tl.trans(b_q), allow_tf32=False)
        b_dv += tl.dot(b_s, tl.trans(b_do), allow_tf32=False)
        o_q += BTS
    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))
    return


@triton.jit
def parallel_retention_bwd_kernel(q, k, v, do, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BTL: 'tl.constexpr', BTS: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // NV
    i_v = i_kv % NV
    i_h = i_bh % H
    _parallel_retention_bwd_dq(i_bh, i_c, i_k, i_v, i_h, k, v, do, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV)
    tl.debug_barrier()
    _parallel_retention_bwd_dkv(i_bh, i_c, i_k, i_v, i_h, q, k, v, do, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, scale, B, H, T, K, V, BTL, BTS, BK, BV)


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


@triton.jit
def fused_recurrent_rwkv4_forward_kernel(w_ptr, w_s_c, u_ptr, u_s_c, k_ptr, k_s_b, k_s_t, k_s_c, v_ptr, v_s_b, v_s_t, v_s_c, state_ptr, state_s_b, state_s_abe, state_s_c, wkv_ptr, wkv_s_b, wkv_s_t, wkv_s_c, state_out_ptr, state_out_s_b, state_out_s_abe, state_out_s_t, state_out_s_c, chans, tsz, BLOCK_SIZE_C: 'tl.constexpr'):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    cs = c_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe
    wkv_ptr = wkv_ptr + b_idx * wkv_s_b
    alpha_out_ptr = state_out_ptr + b_idx * state_out_s_b
    beta_out_ptr = state_out_ptr + b_idx * state_out_s_b + state_out_s_abe
    eps_out_ptr = state_out_ptr + b_idx * state_out_s_b + 2 * state_out_s_abe
    alpha = tl.load(alpha_ptr + cs * state_s_c, mask=cmask)
    beta = tl.load(beta_ptr + cs * state_s_c, mask=cmask)
    eps = tl.load(eps_ptr + cs * state_s_c, mask=cmask)
    w = tl.load(w_ptr + cs * w_s_c, mask=cmask)
    u = tl.load(u_ptr + cs * u_s_c, mask=cmask)
    for t in range(tsz):
        kt = tl.load(k_ptr + t * k_s_t + cs * k_s_c, mask=cmask)
        vt = tl.load(v_ptr + t * v_s_t + cs * v_s_c, mask=cmask)
        ukt = u + kt
        tau = tl.maximum(ukt, eps)
        e1a = tl.exp(eps - tau)
        e2a = tl.exp(ukt - tau)
        wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)
        tl.store(wkv_ptr + t * wkv_s_t + cs * wkv_s_c, wkv, mask=cmask)
        w_eps = w + eps
        eps = tl.maximum(w_eps, kt)
        e1b = tl.exp(w_eps - eps)
        e2b = tl.exp(kt - eps)
        alpha = e1b * alpha + e2b * vt
        beta = e1b * beta + e2b
        tl.store(alpha_out_ptr + t * state_out_s_t + cs * state_out_s_c, alpha, mask=cmask)
        tl.store(beta_out_ptr + t * state_out_s_t + cs * state_out_s_c, beta, mask=cmask)
        tl.store(eps_out_ptr + t * state_out_s_t + cs * state_out_s_c, eps, mask=cmask)


@triton.jit
def fused_recurrent_rwkv4_backward_kernel(w_ptr, w_s_c, u_ptr, u_s_c, k_ptr, k_s_b, k_s_t, k_s_c, v_ptr, v_s_b, v_s_t, v_s_c, state_ptr, state_s_b, state_s_abe, state_s_t, state_s_c, gwkv_ptr, gwkv_s_b, gwkv_s_t, gwkv_s_c, gstate_out_ptr, gstate_out_s_b, gstate_out_s_abe, gstate_out_s_c, gw_ptr, gw_s_c, gu_ptr, gu_s_c, gk_ptr, gk_s_b, gk_s_t, gk_s_c, gv_ptr, gv_s_b, gv_s_t, gv_s_c, gstate_ptr, gstate_s_b, gstate_s_abe, gstate_s_c, tsz, chans, BLOCK_SIZE_C: 'tl.constexpr'):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    cs = c_idx * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    cmask = cs < chans
    k_ptr = k_ptr + b_idx * k_s_b
    v_ptr = v_ptr + b_idx * v_s_b
    alpha_ptr = state_ptr + b_idx * state_s_b
    beta_ptr = state_ptr + b_idx * state_s_b + state_s_abe
    eps_ptr = state_ptr + b_idx * state_s_b + 2 * state_s_abe
    gk_ptr = gk_ptr + b_idx * gk_s_b
    gv_ptr = gv_ptr + b_idx * gv_s_b
    gwkv_ptr = gwkv_ptr + b_idx * gwkv_s_b
    galpha_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b
    gbeta_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + gstate_out_s_abe
    geps_out_ptr = gstate_out_ptr + b_idx * gstate_out_s_b + 2 * gstate_out_s_abe
    galpha = tl.load(galpha_out_ptr + gstate_out_s_c * cs, mask=cmask)
    gbeta = tl.load(gbeta_out_ptr + gstate_out_s_c * cs, mask=cmask)
    geps = tl.load(geps_out_ptr + gstate_out_s_c * cs, mask=cmask)
    w = tl.load(w_ptr + w_s_c * cs, mask=cmask)
    u = tl.load(u_ptr + u_s_c * cs, mask=cmask)
    gw = tl.zeros_like(w)
    gu = tl.zeros_like(u)
    alpha_prev = tl.load(alpha_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask)
    beta_prev = tl.load(beta_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask)
    eps_prev = tl.load(eps_ptr + tsz * state_s_t + state_s_c * cs, mask=cmask)
    for t in range(tsz):
        tc = tsz - t - 1
        kt = tl.load(k_ptr + tc * k_s_t + k_s_c * cs, mask=cmask)
        vt = tl.load(v_ptr + tc * v_s_t + v_s_c * cs, mask=cmask)
        alpha_curr = alpha_prev
        beta_curr = beta_prev
        eps_curr = eps_prev
        alpha_prev = tl.load(alpha_ptr + tc * state_s_t + state_s_c * cs, mask=cmask)
        beta_prev = tl.load(beta_ptr + tc * state_s_t + state_s_c * cs, mask=cmask)
        eps_prev = tl.load(eps_ptr + tc * state_s_t + state_s_c * cs, mask=cmask)
        ukt = u + kt
        tau = tl.maximum(ukt, eps_prev)
        e1 = tl.exp(eps_prev - tau)
        e2 = tl.exp(ukt - tau)
        euke = tl.exp(ukt + eps_prev - 2 * tau)
        denom = e1 * beta_prev + e2
        denom_sq = denom * denom
        gwkvt = tl.load(gwkv_ptr + tc * gwkv_s_t + gwkv_s_c * cs, mask=cmask)
        guk = gwkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
        gu += guk
        gk = guk
        gv = gwkvt * e2 / denom
        galpha_wkv = gwkvt * e1 / denom
        gbeta_wkv = -gwkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
        geps_wkv_denom = e1 * beta_prev + e2
        geps_wkv = gwkvt * euke * (alpha_prev - vt * beta_prev) / (geps_wkv_denom * geps_wkv_denom)
        e1 = tl.exp(w + eps_prev - eps_curr)
        e2 = tl.exp(kt - eps_curr)
        galpha_we = galpha * e1 * alpha_prev
        gw += galpha_we
        gk += galpha * e2 * vt
        gv += galpha * e2
        geps += galpha * -alpha_curr
        gbeta_we = gbeta * e1 * beta_prev
        gw += gbeta_we
        gk += gbeta * e2
        geps += gbeta * -beta_curr
        geps_mask = w + eps_prev > kt
        geps_we = tl.where(geps_mask, geps, tl.zeros_like(geps))
        gw += geps_we
        gk += tl.where(geps_mask, tl.zeros_like(geps), geps)
        tl.store(gk_ptr + tc * gk_s_t + gk_s_c * cs, gk, mask=cmask)
        tl.store(gv_ptr + tc * gv_s_t + gv_s_c * cs, gv, mask=cmask)
        galpha = galpha * e1 + galpha_wkv
        gbeta = gbeta * e1 + gbeta_wkv
        geps = galpha_we + gbeta_we + geps_we + geps_wkv
    galpha_ptr = gstate_ptr + b_idx * gstate_s_b
    gbeta_ptr = gstate_ptr + b_idx * gstate_s_b + gstate_s_abe
    geps_ptr = gstate_ptr + b_idx * gstate_s_b + 2 * gstate_s_abe
    tl.store(galpha_ptr + gstate_s_c * cs, galpha, mask=cmask)
    tl.store(gbeta_ptr + gstate_s_c * cs, gbeta, mask=cmask)
    tl.store(geps_ptr + gstate_s_c * cs, geps, mask=cmask)
    gw_temp = tl.load(gw_ptr + gw_s_c * cs, mask=cmask)
    gw_temp += gw
    tl.store(gw_ptr + gw_s_c * cs, gw_temp, mask=cmask)
    gu_temp = tl.load(gu_ptr + gu_s_c * cs, mask=cmask)
    gu_temp += gu
    tl.store(gu_ptr + gu_s_c * cs, gu_temp, mask=cmask)


@triton.autotune(configs=[triton.Config({'BS': 16}, num_warps=2), triton.Config({'BS': 16}, num_warps=4), triton.Config({'BS': 16}, num_warps=8), triton.Config({'BS': 32}, num_warps=2), triton.Config({'BS': 32}, num_warps=4), triton.Config({'BS': 32}, num_warps=8), triton.Config({'BS': 64}, num_warps=2), triton.Config({'BS': 64}, num_warps=4), triton.Config({'BS': 64}, num_warps=8)], key=['S'])
@triton.jit
def chunk_rwkv6_fwd_cumsum_kernel(s, o, o_minus_s, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o_minus_s = tl.make_block_ptr(o_minus_s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))
    tl.store(p_o_minus_s, b_o - b_s, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BC', 'BK'])
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_inter(q, k, gi, ge, A, s_k_h, s_k_t, s_k_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_i <= i_j:
        return
    if i_t * BT + i_i * BC >= T:
        return
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gq = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
        b_gn = tl.load(p_gn, boundary_check=(0,))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gq = tl.load(p_gq, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_gq - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'BT'])
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra(q, k, gi, ge, u, A, s_k_h, s_k_t, s_k_d, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_i * BC >= T:
        return
    i_j = i_i
    i_h = i_bh % H
    o_i = tl.arange(0, BC)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    i_k = 0
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_u = tl.make_block_ptr(u + i_h * s_k_t, (s_k_t,), (1,), i_k * BK, (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_k = tl.load(p_k, boundary_check=(0,))
        b_gk = tl.load(p_gk, boundary_check=(0,))
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.0)
        p_qj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qj = tl.load(p_qj, boundary_check=(0,))
        p_qi = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qi = tl.load(p_qi, boundary_check=(0,))
        A_jj = tl.sum(b_qi * b_k * b_u * scale)
        b_A = tl.where(o_i != j, b_A, A_jj)
        tl.store(A + o_A + j, b_A, mask=m_A)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BC', 'BK'])
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_split(q, k, gi, ge, u, A, s_k_h, s_k_t, s_k_d, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    if i_t * BT + i_i * BC >= T:
        return
    i_j = i_i
    i_h = i_bh % H
    o_i = tl.arange(0, BC)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_A = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    p_u = tl.make_block_ptr(u + i_h * s_k_t, (s_k_t,), (1,), i_k * BK, (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_k = tl.load(p_k, boundary_check=(0,))
        b_gk = tl.load(p_gk, boundary_check=(0,))
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i > j, b_A * scale, 0.0)
        p_qj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qj = tl.load(p_qj, boundary_check=(0,))
        p_qi = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_qi = tl.load(p_qi, boundary_check=(0,))
        A_jj = tl.sum(b_qi * b_k * b_u * scale)
        b_A = tl.where(o_i != j, b_A, A_jj)
        tl.store(A + o_A + j, b_A, mask=m_A)


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BC'])
@triton.jit
def chunk_rwkv6_fwd_A_kernel_intra_sub_intra_merge(A, A2, T: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', NK: 'tl.constexpr'):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_c * BC >= T:
        return
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_bh + i_k * n_bh) * T * BC, (T, BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_rwkv6_fwd_kernel_inter(q, v, g, h, o, A, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_ge = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, s_h_d), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = b_q * scale
        b_g = tl.load(p_ge, boundary_check=(0, 1))
        b_qg = b_q * tl.exp(b_g)
        b_h = tl.load(p_h, boundary_check=(0, 1))
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_A = tl.where(m_s, b_A, 0.0)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'NC', 'BT'])
@triton.jit
def chunk_rwkv6_bwd_kernel_intra(q, k, gi, ge, dA, dq, dk, s_k_h, s_k_t, s_k_d, scale, T: 'tl.constexpr', K: 'tl.constexpr', BT: 'tl.constexpr', BC: 'tl.constexpr', BK: 'tl.constexpr', NC: 'tl.constexpr'):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    i_t, i_i = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_q = i_t * BT + i_i * BC
    m_k = o_k < K
    p_ge = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_ge = tl.load(p_ge, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    o_i = tl.arange(0, BC)
    m_dA = i_t * BT + i_i * BC + tl.arange(0, BC) < T
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        b_gn = tl.load(gi + i_bh * T * K + (o_q - 1) * K + o_k, mask=m_k & (i_i > 0) & (o_q <= T), other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * tl.exp(b_gn[None, :] - b_gk)
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= tl.exp(b_ge - b_gn[None, :])
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_kj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gkj = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        b_kj = tl.load(p_kj, boundary_check=(0,))
        b_gkj = tl.load(p_gkj, boundary_check=(0,))
        m_i = o_i[:, None] > j
        tmp = tl.exp(b_ge - b_gkj[None, :])
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * tmp, 0.0)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.debug_barrier()
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)
    p_gk = tl.make_block_ptr(gi + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    max_block_idx = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < max_block_idx - 1:
        p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_i * BC + BC - 1) * K + i_k * BK,), (BK,), (0,))
        b_gn = tl.load(p_gn, boundary_check=(0,))
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_ge = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_j * BC, i_i * BC), (BC, BC), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_ge = tl.load(p_ge, boundary_check=(0, 1))
            b_qg = b_q * tl.exp(b_ge - b_gn[None, :])
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            b_dk += tl.dot(tl.trans(b_dA), b_qg, allow_tf32=False)
        b_dk *= tl.exp(b_gn[None, :] - b_gk)
    o_dA = i_bh * T * BT + (i_t * BT + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_qj = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        p_gqj = tl.make_block_ptr(ge + i_bh * s_k_h, (T * K,), (1,), ((i_t * BT + i_i * BC + j) * K + i_k * BK,), (BK,), (0,))
        b_dA = tl.load(dA + o_dA + j * BT, mask=i_t * BT + i_i * BC + j < T, other=0)
        b_qj = tl.load(p_qj, boundary_check=(0,))
        b_gqj = tl.load(p_gqj, boundary_check=(0,))
        m_i = o_i[:, None] < j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * tl.exp(b_gqj[None, :] - b_gk), 0.0)
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BK', 'BV', 'BT'])
@triton.jit
def chunk_rwkv6_bwd_kernel_inter(q, k, v, h, gi, ge, u, do, dh, dA, dq, dk, dq2, dk2, dg, du, s_k_h, s_k_t, s_k_d, s_v_h, s_v_t, s_v_d, s_h_h, s_h_t, s_h_d, scale, H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    n_bh = tl.num_programs(2)
    last_idx = min(T, i_t * BT + BT) - 1
    p_gn = tl.make_block_ptr(gi + i_bh * s_k_h, (T * K,), (s_k_d,), (last_idx * K + i_k * BK,), (BK,), (0,))
    b_gn = tl.load(p_gn, boundary_check=(0,))
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * V * K, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * V * K, (V, K), (s_h_d, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, b_dh)
    p_gk = tl.make_block_ptr(ge + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_dgk *= tl.exp(b_gn)
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    p_gi = tl.make_block_ptr(gi + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_gi = tl.load(p_gi, boundary_check=(0, 1))
    b_dq = b_dq * tl.exp(b_gk)
    b_dk = b_dk * tl.exp(b_gn[None, :] - b_gi)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :] - b_q * b_dq
    o_i = tl.arange(0, BT)
    p_dA_dig = dA + i_bh * T * BT + (i_t * BT + o_i) * BT + o_i
    b_dA_dig = tl.load(p_dA_dig, mask=i_t * BT + o_i < T, other=0)
    p_u = tl.make_block_ptr(u + i_h * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    b_u = tl.load(p_u, boundary_check=(0,))
    b_dq += b_dA_dig[:, None] * b_u[None, :] * b_k
    b_dk += b_dA_dig[:, None] * b_u[None, :] * b_q
    b_du = tl.sum(b_dA_dig[:, None] * b_q * b_k, axis=0)
    p_du = tl.make_block_ptr(du + (i_h + i_t * n_bh) * K, (K,), (1,), (i_k * BK,), (BK,), (0,))
    tl.store(p_du, b_du, boundary_check=(0,))
    p_dg = tl.make_block_ptr(dg + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq2 + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2 + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.heuristics({'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None, 'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None})
@triton.jit
def chunk_rwkv6_bwd_kernel_dh(q, gi, ge, do, dh, dht, dh0, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr', NG: 'tl.constexpr', STORE_INITIAL_STATE_GRADIENT: 'tl.constexpr', USE_FINAL_STATE_GRADIENT: 'tl.constexpr'):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_bg = i_bh // NG
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1))
    for i_t in range(NT - 1, -1, -1):
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh, boundary_check=(0, 1))
        last_idx = min(i_t * BT + BT, T) - 1
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        p_gk = tl.make_block_ptr(ge + i_bg * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_q = b_q * tl.exp(b_gk) * scale
        p_gk_last = gi + i_bg * s_k_h + last_idx * K + i_k * BK + tl.arange(0, BK)
        p_gk_last = tl.max_contiguous(tl.multiple_of(p_gk_last, BK), BK)
        b_gk_last = tl.load(p_gk_last, mask=i_k * BK + tl.arange(0, BK) < K, other=0.0)
        b_dh *= tl.exp(b_gk_last)[:, None]
        b_dh += tl.dot(b_q, b_do)
    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh, boundary_check=(0, 1))


@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(q, k, v, w, u, o, h0, ht, s_k_h, s_v_h, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', STORE_FINAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0)
    b_u = tl.load(p_u, mask=mask_bk, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_w = tl.load(p_w, mask=mask_bk, other=0)
        b_w = tl.exp(b_w)
        b_kv = b_k[None, :] * b_v[:, None]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_o, b_o, mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K
    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h, mask=mask_kv)


@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dq(k, v, w, u, do, dq, dq_aux, h0, s_k_h, s_v_h, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if REVERSE else 0)
    p_dq = dq + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_dq_aux = dq_aux + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]
    b_u = tl.load(p_u, mask=mask_bk, other=0)
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_kv = b_k[None, :] * b_v[:, None]
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_w = tl.load(p_w, mask=mask_bk, other=0)
        b_w = tl.exp(b_w)
        h_q = b_h * b_do[:, None]
        b_dq = tl.sum(h_q + b_kv * b_u[None, :] * b_do[:, None], axis=0)
        b_dq *= scale
        b_dq_aux = tl.sum(h_q, axis=0)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_dq, b_dq, mask=mask_bk)
        tl.store(p_dq_aux, b_dq_aux, mask=mask_bk)
        p_k += -K if REVERSE else K
        p_do += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K
        p_dq += -K if REVERSE else K
        p_dq_aux += -K if REVERSE else K


@triton.jit
def fused_recurrent_rwkv6_bwd_kernel_dkv(q, k, v, w, u, do, dk, dk_aux, dv, dh0, s_k_h, s_v_h, scale, B: 'tl.constexpr', H: 'tl.constexpr', T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', USE_INITIAL_STATE: 'tl.constexpr', REVERSE: 'tl.constexpr'):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_do = do + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    p_dk = dk + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_dk_aux = dk_aux + (i_bh + i_v * B * H) * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    p_dv = dv + (i_bh + i_k * B * H) * s_v_h + i_v * BV + tl.arange(0, BV) + ((T - 1) * V if not REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T - 1) * K if not REVERSE else 0)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V
    mask_kv = mask_bk[:, None] & mask_bv[None, :]
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK
    b_u = tl.load(p_u, mask=mask_bk, other=0)
    for _ in range(T - 1, -1, -1):
        b_q = tl.load(p_q, mask=mask_bk, other=0) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0)
        b_v = tl.load(p_v, mask=mask_bv, other=0)
        b_w = tl.load(p_w, mask=mask_bk, other=0)
        b_do = tl.load(p_do, mask=mask_bv, other=0)
        b_dkv = b_q[:, None] * b_do[None, :]
        b_dk = tl.sum(b_dh * b_v[None, :], axis=1)
        tl.store(p_dk_aux, b_dk, mask=mask_bk)
        b_dk += tl.sum(b_dkv * b_u[:, None] * b_v[None, :], axis=1)
        b_dv = tl.sum((b_dh + b_dkv * b_u[:, None]) * b_k[:, None], axis=0)
        tl.store(p_dk, b_dk, mask=mask_bk)
        tl.store(p_dv, b_dv, mask=mask_bv)
        b_dh *= tl.exp(b_w)[:, None]
        b_dh += b_dkv
        p_q += K if REVERSE else -K
        p_k += K if REVERSE else -K
        p_v += V if REVERSE else -V
        p_w += K if REVERSE else -K
        p_do += V if REVERSE else -V
        p_dk += K if REVERSE else -K
        p_dk_aux += K if REVERSE else -K
        p_dv += V if REVERSE else -V
    if USE_INITIAL_STATE:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, b_dh, mask=mask_kv)


@triton.autotune(configs=[triton.Config({}, num_warps=4)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_simple_gla_fwd_kernel_o(q, k, v, h, g, o, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_o = (b_o + tl.dot(b_s, b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(q, k, v, h, g, do, dh, dq, dk, dg, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, scale, T: 'tl.constexpr', K: 'tl.constexpr', V: 'tl.constexpr', BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)
    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,))
    last_idx = min(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + i_bh * T + last_idx)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1], dtype=tl.float32)
    b_dg = tl.zeros([BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dg_last += tl.sum(b_h * b_dh)
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v, b_dh)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_dg_last *= tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_ds = b_ds
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    b_dg = tl.where(o_i < min(BT, T - i_t * BT) - 1, b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_k * n_bh + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dq, b_dq, boundary_check=(0, 1))
    tl.store(p_dk, b_dk, boundary_check=(0, 1))
    tl.store(p_dg, b_dg, boundary_check=(0,))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['BT'])
@triton.jit
def compute_final_dg(dg, o, T: 'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_o = tl.make_block_ptr(dg + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_o = tl.load(p_o, boundary_check=(0,))
    b_o = b_o - tl.cumsum(b_o, axis=0) + tl.sum(b_o, axis=0)
    p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_o, b_o, boundary_check=(0,))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT', 'BK', 'BV'])
@triton.jit
def chunk_bwd_dv_kernel(q, k, g, do, dv, dh, s_k_h, s_k_t, s_v_h, s_v_t, s_h_h, s_h_t, T, K, V, scale, BT: 'tl.constexpr', BK: 'tl.constexpr', BV: 'tl.constexpr', NT: 'tl.constexpr'):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    last_idx = min(i_t * BT + BT, T) - 1
    b_g = tl.load(g + i_bh * T + i_t * BT + tl.arange(0, BT))
    b_g_last = tl.load(g + i_bh * T + last_idx)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dv += tl.dot(b_k, b_dh) * tl.exp(-b_g + b_g_last)[:, None]
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_A += tl.dot(b_k, b_q, allow_tf32=False)
    b_A = b_A * tl.exp(b_g[None, :] - b_g[:, None]) * scale
    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0)
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv += tl.dot(b_A, b_do)
    p_dv = tl.make_block_ptr(dv + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_dv, b_dv, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8), triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton.Config({'BT': 64}, num_warps=8)], key=['S'])
@triton.jit
def logcumsumexp_fwd_kernel(s, z, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_mp = tl.full([S], float('-inf'), dtype=tl.float32)
    b_zp = tl.zeros([S], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_mc = tl.max(b_s, 0)
        if i_t > 0:
            b_mc = tl.maximum(b_mp, b_mc)
        b_zp = b_zp * tl.exp(b_mp - b_mc)
        b_s = tl.exp(b_s - b_mc)
        b_z = tl.dot(m_s, b_s, allow_tf32=False) + b_zp
        b_zc = tl.max(b_z, 0)
        b_mp = b_mc
        b_zp = b_zc
        b_z = tl.log(tl.where(b_z != 0, b_z, 1e-20)) + b_mc
        tl.store(p_z, b_z, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['S'])
@triton.jit
def softmax_fwd_kernel(s, p, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_m = tl.max(b_s, 1)
    b_s = tl.exp(b_s - b_m[:, None])
    b_z = tl.sum(b_s, 1)
    b_p = tl.where(b_s != 0, b_s / b_z[:, None], 0.0)
    tl.store(p_p, b_p, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['S'])
@triton.jit
def softmax_bwd_kernel(p, dp, ds, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_p = tl.make_block_ptr(p + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_dp = tl.make_block_ptr(dp + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    p_ds = tl.make_block_ptr(ds + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, 0), (BT, S), (1, 0))
    b_p = tl.load(p_p, boundary_check=(0, 1))
    b_dp = tl.load(p_dp, boundary_check=(0, 1))
    b_pp = tl.sum(b_p * b_dp, 1)
    b_ds = b_p * b_dp - b_p * b_pp[:, None]
    tl.store(p_ds, b_ds, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8), triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton.Config({'BT': 64}, num_warps=8)], key=['S'])
@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(s, z, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c, boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)


@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.Config({'BT': 16}, num_warps=4), triton.Config({'BT': 16}, num_warps=8), triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=8), triton.Config({'BT': 64}, num_warps=2), triton.Config({'BT': 64}, num_warps=4), triton.Config({'BT': 64}, num_warps=8)], key=['S'])
@triton.jit
def chunk_global_cumsum_vector_kernel(s, z, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_bh = tl.program_id(0), tl.program_id(1)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_z = tl.make_block_ptr(z + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        b_s = tl.load(p_s, boundary_check=(0, 1))
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c, boundary_check=(0, 1))
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)


@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 64}, num_warps=8), triton.Config({'BT': 64}, num_warps=4)], key=[])
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel(s, o, T: 'tl.constexpr', BT: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT) - 1, -1, -1):
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,))
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o, boundary_check=(0,))


@triton.autotune(configs=[triton.Config({'BT': 16}, num_warps=2), triton.Config({'BT': 32}, num_warps=4), triton.Config({'BT': 32}, num_warps=2), triton.Config({'BT': 64}, num_warps=8), triton.Config({'BT': 64}, num_warps=4)], key=[])
@triton.jit
def chunk_global_cumsum_scalar_kernel(s, o, T: 'tl.constexpr', BT: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cdiv(T, BT)):
        p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,))
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        tl.store(p_o, b_o, boundary_check=(0,))


@triton.autotune(configs=[triton.Config({'BS': 16}, num_warps=2), triton.Config({'BS': 16}, num_warps=4), triton.Config({'BS': 16}, num_warps=8), triton.Config({'BS': 32}, num_warps=2), triton.Config({'BS': 32}, num_warps=4), triton.Config({'BS': 32}, num_warps=8), triton.Config({'BS': 64}, num_warps=2), triton.Config({'BS': 64}, num_warps=4), triton.Config({'BS': 64}, num_warps=8)], key=['S', 'BT'])
@triton.jit
def chunk_local_cumsum_vector_kernel(s, o, s_s_h, s_s_t, s_s_d, T: 'tl.constexpr', S: 'tl.constexpr', BT: 'tl.constexpr', BS: 'tl.constexpr'):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    b_s = tl.load(p_s, boundary_check=(0, 1))
    b_o = tl.dot(m_s, b_s, allow_tf32=False)
    tl.store(p_o, b_o, boundary_check=(0, 1))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=['BT'])
@triton.jit
def chunk_local_cumsum_scalar_kernel(s, o, T: 'tl.constexpr', BT: 'tl.constexpr'):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    p_s = tl.make_block_ptr(s + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,))
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o, boundary_check=(0,))


@triton.autotune(configs=[triton.Config({}, num_warps=1), triton.Config({}, num_warps=2), triton.Config({}, num_warps=4), triton.Config({}, num_warps=8), triton.Config({}, num_warps=16), triton.Config({}, num_warps=32)], key=['D'])
@triton.heuristics({'HAS_SCALE': lambda args: args['scale'] is not None})
@triton.jit
def logsumexp_fwd_kernel(x, z, scale, D: 'tl.constexpr', B: 'tl.constexpr', HAS_SCALE: 'tl.constexpr'):
    i_n, i_d = tl.program_id(0), tl.program_id(1)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    b_m = tl.max(b_x, 0)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.autotune(configs=[triton.Config({'BM': 128, 'BK': 64, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8), triton.Config({'BM': 64, 'BK': 32, 'BN': 256, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 128, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 128, 'BK': 32, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 64, 'BK': 32, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 128, 'BK': 32, 'BN': 32, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 64, 'BK': 32, 'BN': 32, 'G': 4}, num_stages=5, num_warps=2), triton.Config({'BM': 32, 'BK': 32, 'BN': 64, 'G': 4}, num_stages=5, num_warps=2), triton.Config({'BM': 128, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=3, num_warps=8), triton.Config({'BM': 256, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=3, num_warps=8), triton.Config({'BM': 256, 'BK': 128, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 64, 'BK': 128, 'BN': 256, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 128, 'BK': 128, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 128, 'BK': 64, 'BN': 64, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 64, 'BK': 64, 'BN': 128, 'G': 4}, num_stages=4, num_warps=4), triton.Config({'BM': 128, 'BK': 64, 'BN': 32, 'G': 4}, num_stages=4, num_warps=4)], key=['M', 'N', 'K'])
@triton.heuristics({'HAS_INPUT': lambda args: args['input'] is not None, 'HAS_ALPHA': lambda args: args['alpha'] is not None, 'HAS_BETA': lambda args: args['beta'] is not None})
@triton.jit
def matmul_kernel(a, b, c, input, alpha, beta, M, N, K, s_am, s_ak, s_bk, s_bn, s_cm, s_cn, BM: 'tl.constexpr', BK: 'tl.constexpr', BN: 'tl.constexpr', G: 'tl.constexpr', ACTIVATION: 'tl.constexpr', HAS_INPUT: 'tl.constexpr', HAS_ALPHA: 'tl.constexpr', HAS_BETA: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    NM, NN = tl.num_programs(0), tl.num_programs(1)
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    i_m, i_n = tl.swizzle2d(i_m, i_n, NM, NN, G)
    o_am = (i_m * BM + tl.arange(0, BM)) % M
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)
    p_a = a + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[:, None] * s_bk + o_bn[None, :] * s_bn)
    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        p_a += BK * s_ak
        p_b += BK * s_bk
    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cm[:, None] < M) & (o_cn[None, :] < N)
    b_c = b_acc
    if ACTIVATION == 'leaky_relu':
        b_c = leaky_relu(b_c)
    if HAS_ALPHA:
        b_c *= tl.load(alpha)
    if HAS_INPUT:
        p_i = input + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]
        b_i = tl.load(p_i, mask=mask, other=0.0)
        if HAS_BETA:
            b_i *= tl.load(beta)
        b_c += b_i
    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]
    tl.store(p_c, b_c, mask=mask)


@triton.jit
def attention_fwd_kernel(q, k, v, h, o, s_qh, s_qt, s_qd, s_hh, s_ht, T, scale, BT: 'tl.constexpr', BD: 'tl.constexpr', NT: 'tl.constexpr', STORE: 'tl.constexpr', IFCOND: 'tl.constexpr'):
    i_bh = tl.program_id(0)
    b_h = tl.zeros([BD, BD], dtype=tl.float32)
    for i in range(0, tl.cdiv(T, BT)):
        p_q = tl.make_block_ptr(q + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qh, (BD, T), (s_qd, s_qt), (0, i * BT), (BD, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_hh, (NT * BD, BD), (s_ht, s_qd), (i * BD, 0), (BD, BD), (1, 0))
        p_o = tl.make_block_ptr(o + i_bh * s_qh, (T, BD), (s_qt, s_qd), (i * BT, 0), (BT, BD), (1, 0))
        if STORE:
            tl.store(p_h, b_h)
        b_q = tl.load(p_q)
        b_q = b_q * scale
        b_k = tl.load(p_k)
        b_v = tl.load(p_v)
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_o = tl.dot(b_s, b_v, allow_tf32=False)
        if IFCOND:
            if i == 0:
                b_h = tl.dot(b_k, b_v, allow_tf32=False)
            else:
                b_o += tl.dot(b_q, b_h, allow_tf32=False)
                b_h += tl.dot(b_k, b_v, allow_tf32=False)
        else:
            b_o += tl.dot(b_q, b_h, allow_tf32=False)
            b_h += tl.dot(b_k, b_v, allow_tf32=False)
        tl.store(p_o, b_o)

