import sys
_module = sys.modules[__name__]
del sys
colt5_attention = _module
attend = _module
coor_descent = _module
topk = _module
transformer_block = _module
triton_coor_descent = _module
vit = _module
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


from math import log


import torch


from torch import Tensor


from torch import autograd


import torch.nn.functional as F


from torch.amp import autocast


from torch.cuda.amp import custom_fwd


from torch.cuda.amp import custom_bwd


@triton.jit
def coor_descent_kernel_forward(a_ptr, b_ptr, input_ptr, mask_ptr, k_ptr, a_iter_stride, b_row_stride, b_iter_stride, input_row_stride, mask_row_stride, n_iters, current_eps, eps_decay, eps, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets
    mask_ints = tl.load(mask_ptrs, mask=col_mask, other=0)
    mask = mask_ints == 1
    a_ptr = a_ptr + row_idx
    a = tl.load(a_ptr)
    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    b = tl.load(b_ptrs, mask=col_mask, other=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)
    logk = tl.log(k)
    for _ in range(n_iters):
        a = (s + b) / current_eps
        a = tl.where(mask, a, -float('inf'))
        a_max = tl.max(a, axis=0)
        a_minus_max = tl.where(mask, a - a_max, -float('inf'))
        exp = tl.exp(a_minus_max)
        sum_exp = tl.sum(exp, axis=0)
        log_sum_exp = tl.log(sum_exp) + a_max
        a = current_eps * (logk - log_sum_exp)
        b = s + a
        b = tl.where(b >= 0.0, -b, 0.0)
        current_eps *= eps_decay
        if current_eps < eps:
            current_eps = eps
    next_a_ptrs = a_ptr + a_iter_stride
    next_b_ptrs = b_ptrs + b_iter_stride
    tl.store(next_a_ptrs, a)
    tl.store(next_b_ptrs, b, mask=col_mask)


@triton.jit
def coor_descent_kernel_backward(dk_ptr, input_ptr, a_ptr, b_ptr, mask_ptr, ds_ptr, db_ptr, k_ptr, last_da_ptr, input_row_stride, b_row_stride, mask_row_stride, ds_row_stride, db_row_stride, n_iters, eps_init, eps_decay, eps, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols
    mask_start_ptr = mask_ptr + row_idx * mask_row_stride
    mask_ptrs = mask_start_ptr + col_offsets
    mask_ints = tl.load(mask_ptrs, mask=col_mask, other=0)
    mask = mask_ints == 1
    a_ptr = a_ptr + row_idx
    init_a = tl.load(a_ptr)
    b_start_ptr = b_ptr + row_idx * b_row_stride
    b_ptrs = b_start_ptr + col_offsets
    init_b = tl.load(b_ptrs, mask=mask, other=0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    s = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    k_ptr = k_ptr + row_idx
    k = tl.load(k_ptr)
    logk = tl.log(k)
    last_da_ptr = last_da_ptr + row_idx
    last_da = tl.load(last_da_ptr)
    ds_row_start_ptr = ds_ptr + row_idx * ds_row_stride
    ds_ptrs = ds_row_start_ptr + col_offsets
    ds = tl.load(ds_ptrs, mask=mask, other=0.0)
    db_row_start_ptr = db_ptr + row_idx * db_row_stride
    db_ptrs = db_row_start_ptr + col_offsets
    db = tl.load(db_ptrs, mask=mask, other=0.0)
    dk_ptr = dk_ptr + row_idx
    dk = tl.load(dk_ptr)
    for ind in range(n_iters):
        a = init_a
        b = init_b
        sa = s * 0
        softmax = s * 0
        current_eps = eps_init / eps_decay
        for _ in range(n_iters - ind):
            current_eps *= eps_decay
            if current_eps < eps:
                current_eps = eps
            sb = (s + b) / current_eps
            sb = tl.where(mask, sb, -float('inf'))
            sb_max = tl.max(sb, axis=0)
            sb_minus_max = tl.where(mask, sb - sb_max, -float('inf'))
            exp = tl.exp(sb_minus_max)
            sum_exp = tl.sum(exp, axis=0)
            softmax = exp / sum_exp
            log_sum_exp = tl.log(sum_exp) + sb_max
            a = current_eps * (logk - log_sum_exp)
            sa = s + a
            b = tl.where(sa > 0.0, -sa, 0.0)
        dsa = db * tl.where(sa > 0, -1.0, 0.0)
        ds += dsa
        da = tl.sum(dsa, axis=0) + last_da
        dk += da * current_eps
        dsb = da * -softmax
        ds += dsb
        db = dsb
        last_da *= 0.0
    tl.store(dk_ptr, dk)
    tl.store(ds_ptrs, ds, mask=col_mask)
    tl.store(db_ptrs, db, mask=col_mask)

