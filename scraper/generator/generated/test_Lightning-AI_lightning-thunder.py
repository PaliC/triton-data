import sys
_module = sys.modules[__name__]
del sys
conf = _module
simple_sampling = _module
ggmltensor = _module
configurator = _module
export = _module
model = _module
sample = _module
tinystories = _module
tokenizer = _module
train = _module
bisect_nvfuser = _module
validate_build = _module
setup = _module
__about__ = _module
thunder = _module
benchmarks = _module
benchmark_litgpt = _module
conftest = _module
distributed = _module
einsum = _module
targets = _module
test_benchmark_litgpt = _module
clang = _module
langctx = _module
common = _module
core = _module
baseutils = _module
codeutils = _module
compile_data = _module
devices = _module
dtypes = _module
functionalization = _module
interpreter = _module
jit_ext = _module
langctxs = _module
module = _module
options = _module
patterns = _module
prims = _module
profile = _module
proxies = _module
pytree = _module
rematerialization = _module
symbol = _module
trace = _module
trace_interpreter = _module
transform_common = _module
transforms = _module
utils = _module
vjp_utils = _module
dev_utils = _module
debug_transform = _module
nvtx_profile_transform = _module
bucketing = _module
checkpoint = _module
tensor_parallel = _module
column_wise = _module
optimize_comm = _module
row_wise = _module
ddp = _module
ddp_v2 = _module
fsdp = _module
fsdp_v2 = _module
dynamo = _module
compiler = _module
compiler_graph_benchmark = _module
splitter = _module
examine = _module
memory_caculation = _module
executors = _module
apex_entropyex_impl = _module
apex_fused_rms_norm_impl = _module
apexex = _module
cudnn_layernormex = _module
cudnnex = _module
data_dependent_partition = _module
fa3ex = _module
nvfuserex = _module
nvfuserex_impl = _module
passes = _module
pythonex = _module
sdpaex = _module
torch_autograd = _module
torch_compile = _module
torchex = _module
transformer_engineex = _module
triton_crossentropy = _module
triton_crossentropy_impl = _module
triton_utils = _module
extend = _module
numpy = _module
tests = _module
bf16 = _module
helper = _module
modules = _module
test_checkpoint = _module
test_ddp = _module
test_fsdp = _module
test_ops = _module
test_tensor_parallel = _module
framework = _module
hf_bart_self_attn = _module
litgpt_model = _module
llama2_model = _module
make_tensor = _module
module_example = _module
nanogpt_model = _module
opinfos = _module
test_apex_cross_entropy_executor = _module
test_apex_fused_norms = _module
test_auto_register_torchops = _module
test_autocast = _module
test_core = _module
test_cudnn_executor = _module
test_dynamo = _module
test_einops = _module
test_elementwise = _module
test_examine_memory = _module
test_extend = _module
test_fa3_executor = _module
test_grad = _module
test_inplace_copy = _module
test_inplace_functionalization = _module
test_interpreter = _module
test_jit_general = _module
test_networks = _module
test_nvfuser = _module
test_nvfuser_remat = _module
test_patterns = _module
test_pythonex = _module
test_randomness = _module
test_reductions = _module
test_sdpaex_executor = _module
test_shape_ops = _module
test_torch_compile_executor = _module
test_transformer_engine_executor = _module
test_transforms = _module
test_triton_ce = _module
default_torch_ops = _module
autocast = _module
constant_folding = _module
cudagraph = _module
materialization = _module
qlora = _module
quantization = _module

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


import time


from collections import UserDict


from collections.abc import Callable


from collections.abc import Sequence


from functools import partial


from numbers import Number


from typing import Any


import torch


import torch.multiprocessing as mp


import torch.nn as nn


from torch.testing import make_tensor


from collections.abc import Hashable


import math


from enum import Enum


import triton


import triton.language as tl


import numpy as np


import re


from torch.testing import assert_close


class TritonDtype(Enum):
    kFP16 = 0
    kBF16 = 1
    kFP32 = 2
    kFP64 = 3


_DTYPE2TRITON = {TritonDtype.kFP16: tl.float16, TritonDtype.kBF16: tl.bfloat16, TritonDtype.kFP32: tl.float32, TritonDtype.kFP64: tl.float64}


@triton.jit
def _class_indices_forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS: 'tl.constexpr', CLASS_INDICES: 'tl.constexpr', LABEL_SMOOTHING: 'tl.constexpr', IGNORE_INDEX: 'tl.constexpr', BUFFER_DTYPE: 'tl.constexpr', BLOCK: 'tl.constexpr'):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float('inf')
    l_prev = 0.0
    m_prev = m_prev
    l_prev = l_prev
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK, other=-float('inf'))
        m_curr = tl.maximum(tl.max(row_logits, 0), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(row_logits - m_curr)
        l_curr = tl.sum(p, 0) + l_prev
        l_prev = l_curr
        m_prev = m_curr
        logit_ptrs += BLOCK
    logit_ptrs = logit_start_ptrs + cols
    output_ptrs = PROBS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    if LABEL_SMOOTHING:
        sum_total = 0.0
        sum_total = sum_total
        weights_total = 0.0
        weights_total = weights_total
        if WEIGHTS:
            weight_ptr = weight + cols
    l_prev_log = tl.log(l_prev)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK, other=l_prev_log + m_prev)
        if LABEL_SMOOTHING and WEIGHTS:
            full_weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
            weights_total += tl.sum(full_weights_val, 0)
        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max
        if LABEL_SMOOTHING and WEIGHTS:
            log_softmax *= full_weights_val
        if LABEL_SMOOTHING:
            sum_total += tl.sum(log_softmax, 0)
        tl.store(WRIT_PROBS, log_softmax, mask=cols < N - start_n * BLOCK)
        logit_ptrs += BLOCK
        WRIT_PROBS += BLOCK
        if LABEL_SMOOTHING and WEIGHTS:
            weight_ptr += BLOCK
    idx = tl.load(IDX + row)
    use_class = 0.0
    if IGNORE_INDEX >= 0:
        use_class = idx == IGNORE_INDEX
    READ_PROBS = PROBS + row * N + idx
    tl.debug_barrier()
    probs = tl.load(READ_PROBS)
    if WEIGHTS and not LABEL_SMOOTHING:
        weight_ptr = weight + idx
        weights_val = tl.load(weight_ptr)
        probs = weights_val * probs
    if LABEL_SMOOTHING:
        tl.store(WEIGHT_BUFFER + row, weights_total)
        probs = (1 - smoothing_factor) * probs + smoothing_factor * sum_total / N
    probs = probs * (1.0 - use_class)
    tl.store(LOSS + row, probs)


@triton.jit
def _class_probs_forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS: 'tl.constexpr', CLASS_INDICES: 'tl.constexpr', LABEL_SMOOTHING: 'tl.constexpr', IGNORE_INDEX: 'tl.constexpr', BUFFER_DTYPE: 'tl.constexpr', BLOCK: 'tl.constexpr'):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    logit_start_ptrs = LOGITS + row * N
    logit_ptrs = logit_start_ptrs + cols
    m_prev = -float('inf')
    l_prev = 0.0
    m_prev = m_prev
    l_prev = l_prev
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK, other=-float('inf'))
        m_curr = tl.maximum(tl.max(row_logits, 0), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(row_logits - m_curr)
        l_curr = tl.sum(p, 0) + l_prev
        l_prev = l_curr
        m_prev = m_curr
        logit_ptrs += BLOCK
    logit_ptrs = logit_start_ptrs + cols
    output_ptrs = PROBS + row * N + cols
    WRIT_PROBS = PROBS + row * N + cols
    sum_total = 0.0
    weights_total = 0.0
    sum_total = sum_total
    weights_total = weights_total
    idx_ptr = IDX + row * N + cols
    if WEIGHTS:
        weight_ptr = weight + cols
    l_prev_log = tl.log(l_prev)
    for start_n in range(0, tl.cdiv(N, BLOCK)):
        row_logits = tl.load(logit_ptrs, mask=cols < N - start_n * BLOCK, other=l_prev_log + m_prev)
        idx = tl.load(idx_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
        full_weights_val = (1.0 - smoothing_factor) * idx + smoothing_factor / N
        if WEIGHTS:
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
            full_weights_val = weights_val * full_weights_val
        else:
            full_weights_val = tl.where(cols < N - start_n * BLOCK, full_weights_val, 0.0)
        weights_total += tl.sum(full_weights_val, 0)
        row_minus_max = row_logits - m_prev
        log_softmax = l_prev_log - row_minus_max
        log_softmax *= full_weights_val
        sum_total += tl.sum(log_softmax, 0)
        tl.store(WRIT_PROBS, log_softmax, mask=cols < N - start_n * BLOCK)
        logit_ptrs += BLOCK
        WRIT_PROBS += BLOCK
        idx_ptr += BLOCK
        if WEIGHTS:
            weight_ptr += BLOCK
    tl.store(WEIGHT_BUFFER + row, weights_total)
    probs = sum_total
    tl.store(LOSS + row, probs)


FORWARD_NUM_STAGES = 1


@triton.autotune(configs=[triton.Config({'BLOCK': 1024}, num_stages=FORWARD_NUM_STAGES, num_warps=1), triton.Config({'BLOCK': 2048}, num_stages=FORWARD_NUM_STAGES, num_warps=8), triton.Config({'BLOCK': 4096}, num_stages=FORWARD_NUM_STAGES, num_warps=8), triton.Config({'BLOCK': 8192}, num_stages=FORWARD_NUM_STAGES, num_warps=16), triton.Config({'BLOCK': 16384}, num_stages=FORWARD_NUM_STAGES, num_warps=16)], key=['N', 'CLASS_INDICES', 'log_size_logits', 'BUFFER_DTYPE'])
@triton.jit
def _forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS: 'tl.constexpr', CLASS_INDICES: 'tl.constexpr', LABEL_SMOOTHING: 'tl.constexpr', IGNORE_INDEX: 'tl.constexpr', BUFFER_DTYPE: 'tl.constexpr', BLOCK: 'tl.constexpr'):
    if CLASS_INDICES:
        _class_indices_forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS, CLASS_INDICES, LABEL_SMOOTHING, IGNORE_INDEX, BUFFER_DTYPE, BLOCK)
    else:
        _class_probs_forward(LOGITS, PROBS, IDX, LOSS, weight, N, WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS, CLASS_INDICES, LABEL_SMOOTHING, IGNORE_INDEX, BUFFER_DTYPE, BLOCK)


@triton.autotune(configs=[triton.Config({'BLOCK': 1024}, num_stages=1, num_warps=1), triton.Config({'BLOCK': 2048}, num_stages=1, num_warps=8), triton.Config({'BLOCK': 4096}, num_stages=1, num_warps=8), triton.Config({'BLOCK': 8192}, num_stages=1, num_warps=16), triton.Config({'BLOCK': 16384}, num_stages=1, num_warps=16)], key=['N', 'CLASS_INDICES', 'log_size_logits', 'BUFFER_DTYPE'])
@triton.jit
def _backward(PROBS, IDX, DPROBS, dprob_stride, DIN, weight, N, WEIGHT_BUFFER, smoothing_factor, log_size_logits, WEIGHTS: 'tl.constexpr', CLASS_INDICES: 'tl.constexpr', LABEL_SMOOTHING: 'tl.constexpr', IGNORE_INDEX: 'tl.constexpr', BUFFER_DTYPE: 'tl.constexpr', BLOCK: 'tl.constexpr'):
    buffer_dtype = _DTYPE2TRITON[BUFFER_DTYPE.value]
    row = tl.program_id(0)
    start_n = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    PROBS = PROBS + row * N
    probs_start = PROBS + cols + BLOCK * start_n
    probs = -tl.load(probs_start, mask=cols < N - start_n * BLOCK, other=float('inf'))
    DIN = DIN + row * N + cols + BLOCK * start_n
    dout = tl.load(DPROBS + row * dprob_stride)
    if CLASS_INDICES:
        idx = tl.load(IDX + row)
        delta = start_n * BLOCK + cols == idx
        if IGNORE_INDEX >= 0:
            use_class = idx == IGNORE_INDEX
            dout = dout * (1 - use_class)
        if LABEL_SMOOTHING:
            if WEIGHTS:
                weight_ptr = weight + cols + BLOCK * start_n
                full_weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
                weights_val = tl.load(weight + idx)
                probs = probs / full_weights_val
            probs = tl.exp(probs)
            if WEIGHTS:
                weights_total = tl.load(WEIGHT_BUFFER + row)
                numerator_contrib = weights_val * (1.0 - smoothing_factor) * (probs - delta)
                mean_contrib = (weights_total * probs - full_weights_val) * smoothing_factor / N
            else:
                numerator_contrib = (1.0 - smoothing_factor) * (probs - delta)
                mean_contrib = smoothing_factor * probs - smoothing_factor / N
            din = (numerator_contrib + mean_contrib) * dout
        else:
            probs = tl.exp(probs)
            din = (probs - delta) * dout
            if WEIGHTS:
                weight_ptr = weight + idx
                weights_val = tl.load(weight_ptr)
                din = weights_val * din
    else:
        idx = tl.load(IDX + row * N + cols + BLOCK * start_n, mask=cols < N - start_n * BLOCK, other=0.0)
        full_weights_val = (1.0 - smoothing_factor) * idx + smoothing_factor / N
        weights_total = tl.load(WEIGHT_BUFFER + row)
        if WEIGHTS:
            weight_ptr = weight + cols + BLOCK * start_n
            weights_val = tl.load(weight_ptr, mask=cols < N - start_n * BLOCK, other=0.0)
            full_weights_val = weights_val * full_weights_val
        probs = probs / full_weights_val
        probs = tl.exp(probs)
        weighted_probs = probs * weights_total
        weighted_probs_per_class = weighted_probs - full_weights_val
        din = weighted_probs_per_class * dout
    tl.store(DIN, din, mask=cols + BLOCK * start_n < N)

