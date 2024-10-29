import sys
_module = sys.modules[__name__]
del sys
unsloth = _module
_auto_install = _module
chat_templates = _module
kernels = _module
cross_entropy_loss = _module
fast_lora = _module
flex_attention = _module
geglu = _module
layernorm = _module
rms_layernorm = _module
rope_embedding = _module
swiglu = _module
utils = _module
models = _module
_utils = _module
cohere = _module
dpo = _module
gemma = _module
gemma2 = _module
llama = _module
loader = _module
mapper = _module
mistral = _module
qwen2 = _module
vision = _module
save = _module
tokenizer_utils = _module
trainer = _module

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


import warnings


import re


import inspect


import numpy as np


import triton


import triton.language as tl


import torch


from functools import lru_cache


from torch.nn import LayerNorm


from typing import Union


from typing import Optional


from typing import List


from typing import Any


from typing import Callable


from typing import Tuple


import math


import logging


import functools


import torch._inductor.utils


import torch._inductor.config as config


import torch._dynamo.config as config


import torch.utils


from inspect import getsource


@triton.heuristics({'DO_SOFTCAPPING': lambda args: args['DO_SOFTCAPPING'], 'DO_LOGIT_SCALING': lambda args: args['DO_LOGIT_SCALING']})
@triton.jit
def _cross_entropy_forward(logits_ptr, logits_row_stride, loss_ptr, logsumexp_ptr, labels_ptr, VOCAB_SIZE: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', DO_SOFTCAPPING: 'tl.constexpr', SOFTCAP: 'tl.constexpr', DO_LOGIT_SCALING: 'tl.constexpr', LOGIT_SCALE: 'tl.constexpr'):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        logsumexp is also stable
        Take    y =         log[sum(exp(x))]
           exp(y) =             sum(exp(x))
           exp(y) =             sum(exp(x - c)*exp(c)) Since e^(x-c)*e^c = e^x
           exp(y) =      exp(c)*sum(exp(x - c))
               y  = log(exp(c)*sum(exp(x - c)))
               y  = c + log[sum(exp(x - c))]
        This means we can set c = max(x) to make sure
        exp(x - c) always is exp(x - max(x)).
        This ensures exp(x - max(x))'s maximum is 1 as exp(0) = 1.
    """
    row_idx = tl.program_id(0)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf'))
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    logits = logits
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING:
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:
            x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


@triton.heuristics({'DO_SOFTCAPPING': lambda args: args['DO_SOFTCAPPING'], 'DO_LOGIT_SCALING': lambda args: args['DO_LOGIT_SCALING']})
@triton.jit
def _chunked_cross_entropy_forward(logits_ptr, logits_row_stride, loss_ptr, logsumexp_ptr, labels_ptr, VOCAB_SIZE: 'tl.constexpr', N_CHUNKS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', DO_SOFTCAPPING: 'tl.constexpr', SOFTCAP: 'tl.constexpr', DO_LOGIT_SCALING: 'tl.constexpr', LOGIT_SCALE: 'tl.constexpr'):
    """
        256K vocab divided in 4 chunks

        |-65536-| |-65536-| |-65536-| |-65536-|
        |-------| |-------| |-------| |-------|
        |-------| |-------| |-------| |-------|

        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        Notice we can do logsumexp for each chunk and then
        logsumexp[chunk_sum(logsumexp)] == logsumexp

        chunk_sum = log[chunk_sum(logsumexp)]
                  = log[exp(logsumexp(a)) + ... + exp(logsumexp(z))]
                  = log[exp(log[sum(exp(a))]) + ... + exp(log[sum(exp(z))])]
                  = log[sum(exp(a)) + ... + sum(exp(z))]
                  = logsumexp(x)

        This means we can perform a logsumexp for each chunk, then do a
        final logsumexp reduction!

        Ie do: logsumexp(chunked_logsumexp) - x
    """
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx
    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf'))
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING:
        logits = SOFTCAP * triton_tanh(logits / SOFTCAP)
    logits = logits
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx)
            if DO_LOGIT_SCALING:
                x = LOGIT_SCALE * x
            if DO_SOFTCAPPING:
                x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    pass
    tl.store(logsumexp_ptr, logsumexp)


@triton.heuristics({'DO_SOFTCAPPING': lambda args: args['DO_SOFTCAPPING'], 'DO_LOGIT_SCALING': lambda args: args['DO_LOGIT_SCALING']})
@triton.jit
def _cross_entropy_backward(logits_ptr, logits_row_stride, dloss_ptr, dloss_row_stride, logsumexp_ptr, labels_ptr, VOCAB_SIZE: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', DO_SOFTCAPPING: 'tl.constexpr', SOFTCAP: 'tl.constexpr', DO_LOGIT_SCALING: 'tl.constexpr', LOGIT_SCALE: 'tl.constexpr'):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    logits_ptr += row_idx * logits_row_stride
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx)
    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0
    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float('inf'))
    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE
    pass
    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial
    pass
    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x - logsumexp)
    y = tl.where(col_offsets == label_idx, y - 1.0, y)
    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE
    pass
    if DO_SOFTCAPPING:
        y = y * (1.0 - partial * partial)
    pass
    tl.store(logits_ptr + col_offsets, dloss * y, mask=mask)


@triton.jit
def _exact_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row
    h_row = f_row * g_row
    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _exact_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    """
    f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    h = f * up

    df/de (with help of Wolfram :)
    df/de = 1/2 * (1 + erf(1/sqrt(2) * e)) + 1/sqrt(2*pi) * e * exp(-1/2 * e^2)

    Reuse via
    f =        1/2 * (1 + erf(1/sqrt(2) * e)) * e
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_partial_row = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_partial_row * e_row
    f_row = f_row
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    t = 0.3989422804014327
    df_de = f_partial_row + t * e_row * tl.exp(-0.5 * e_row * e_row)
    de_row = dg_row * df_de
    de_row = de_row
    tl.store(DW + offsets, h_row, mask=mask)
    tl.store(e + offsets, df_row, mask=mask)
    tl.store(g + offsets, de_row, mask=mask)


@triton.jit
def _approx_forward_kernel(e, g, h, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    s = 0.7978845608028654
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_row = 0.5 * e_row * (triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) + 1.0)
    f_row = f_row
    h_row = f_row * g_row
    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _approx_backward_kernel(DW, e, g, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    """
    f = 1/2 * e * (1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) ))
    h = f * up

    df/de (with help from https://arxiv.org/pdf/2305.12073.pdf :))
    df/de = 1/2 * [1 + tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) )] +
            1/2 * sech^2 [   sqrt(2/pi) * x * (1 + 0.044715 * x^2 )  ] *                            ( sqrt(2/pi) * x * (1 + 0.044715 * x^2 * 3 ) )

    Notice sech^2(x) = 1 - tanh^2(x)
    So reuse tanh( sqrt(2/pi) * x * (1 + 0.044715 * x^2 ) )

    See https://www.desmos.com/calculator/nqprfoni6x
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    s = 0.7978845608028654
    a = s * e_row
    b = a * 0.044715 * e_row * e_row
    T = 1.0 + triton_tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2
    f_row = T2 * e_row
    f_row = f_row
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row * df_de
    de_row = de_row
    tl.store(DW + offsets, h_row, mask=mask)
    tl.store(e + offsets, df_row, mask=mask)
    tl.store(g + offsets, de_row, mask=mask)


@triton.jit
def layernorm_forward(Y, Y_row_stride, X, X_row_stride, W, b, r, mu, n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    mean_X = tl.sum(X_row, axis=0) / n_cols
    XX = X_row - mean_X
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = XX * inv_var * W_row + b_row
    tl.store(Y + col_offsets, output, mask=mask)


@triton.jit
def layernorm_backward(dY, dY_row_stride, X, X_row_stride, W, b, r, mu, n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0)
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    inv_var = tl.load(r)
    mean = tl.load(mu)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = dY_W - tl.sum(dY_W, axis=0) / n_cols - normed * tl.sum(dY_W * normed, axis=0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dY + col_offsets, dX_row, mask=mask)


@triton.jit
def _rms_layernorm_forward(Y, Y_row_stride, X, X_row_stride, W, W_row_stride, r, r_row_stride, n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)


@triton.heuristics({'GEMMA': lambda args: args['GEMMA']})
@triton.jit
def _rms_layernorm_backward(dY, dY_row_stride, X, X_row_stride, W, W_row_stride, r, r_row_stride, dW, dW_row_stride, n_cols, eps, GEMMA: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0)
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    inv_var = tl.load(r)
    normed = X_row * inv_var
    if GEMMA:
        dY_W = dY_row * (W_row + 1.0)
    else:
        dY_W = dY_row * W_row
    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask=mask)


@triton.jit
def _gemma_rms_layernorm_forward(Y, Y_row_stride, X, X_row_stride, W, W_row_stride, r, r_row_stride, n_cols, eps, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride
    X_row = tl.load(X + col_offsets, mask=mask, other=0)
    W_row = tl.load(W + col_offsets, mask=mask, other=0)
    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)
    tl.store(Y + col_offsets, output, mask=mask)


@triton.heuristics({'BACKWARD_PASS': lambda args: args['BACKWARD_PASS']})
@triton.jit
def _rope_embedding(Q, Q_row_stride, cos, cos_row_stride, sin, sin_row_stride, seqlen, head_dim: 'tl.constexpr', n_heads: 'tl.constexpr', BACKWARD_PASS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
        Calculates the RoPE Embedding quickly
        RoPE is Q * cos + rotate_half(Q) * sin
        See our blog post for more info
    """
    ROPE_GROUP_SIZE = 4
    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim
    sin1 = tl.load(sin + row_position % seqlen * sin_row_stride + half_head_dim * 0 + col_offsets, mask=mask, other=0)
    cos1 = tl.load(cos + row_position % seqlen * cos_row_stride + half_head_dim * 0 + col_offsets, mask=mask, other=0)
    if BACKWARD_PASS:
        sin1 = -sin1
    pass
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, n_heads)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
        Q1 = tl.load(Q + offs_q1, mask=mask, other=0)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0)
        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)
    pass


@triton.jit
def _fg_kernel(e, g, h, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row
    h_row = f_row * g_row
    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _DWf_DW_dfg_kernel(DW, e, g, n_elements, BLOCK_SIZE: 'tl.constexpr'):
    """
    e = e.float()
    se = 1.0 / (1.0 + torch.exp(-e))
    f = (se * e).to(dtype)
    h = f * g
    df = DW * f
    dg = DW * g
    de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)
    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row
    tl.store(DW + offsets, h_row, mask=mask)
    tl.store(e + offsets, df_row, mask=mask)
    tl.store(g + offsets, de_row, mask=mask)

