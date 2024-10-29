import sys
_module = sys.modules[__name__]
del sys
amg_example = _module
data = _module
eval_combo = _module
metrics = _module
run_experiments = _module
summary_chart = _module
segment_anything_fast = _module
automatic_mask_generator = _module
build_sam = _module
configs = _module
flash_4 = _module
modeling = _module
common = _module
image_encoder = _module
mask_decoder = _module
prompt_encoder = _module
sam = _module
transformer = _module
predictor = _module
sparse = _module
tools = _module
utils = _module
amg = _module
onnx = _module
transforms = _module
setup = _module
test_flash_4 = _module
test_mask_to_rle = _module

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


import time


import triton


import triton.language as tl


@triton.jit
def _fwd_kernel_aligned(Q, K, V, B0, sm_scale, Out, stride_qh, stride_qm, stride_qk, stride_kh, stride_kn, stride_kk, stride_vh, stride_vk, stride_vn, stride_oh, stride_om, stride_on, stride_b0h, stride_b0m, Z, H, N_CTX, P_SEQ, OUT_DTYPE: 'tl.constexpr', BIAS_LAST_SIZE: 'tl.constexpr', B0_NUMEL: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    q_offset = off_hz * stride_qh
    kv_offset = off_hz * stride_kh
    Q_block_ptr = tl.make_block_ptr(base=Q + q_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K + kv_offset, shape=(BLOCK_DMODEL, N_CTX + P_SEQ), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V + kv_offset, shape=(N_CTX + P_SEQ, BLOCK_DMODEL), strides=(stride_vk, stride_vn), offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr)
    q = q * qk_scale
    lo = 0
    hi = N_CTX + P_SEQ
    b_ptr_offsets_m = tl.arange(0, BLOCK_M)
    b_offset = off_hz * stride_b0h
    b_ptr_offsets_n_1 = tl.arange(0, BLOCK_N) % BIAS_LAST_SIZE + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :])
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k)
        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m) * stride_b0m)[:, None] + start_n // BLOCK_N)
        qk += (b0 + b1) * 1.44269504
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    acc = acc / l_i[:, None]
    O_block_ptr = tl.make_block_ptr(base=Out + q_offset, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(O_block_ptr, acc)

