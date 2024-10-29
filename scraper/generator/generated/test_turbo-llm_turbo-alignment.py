import sys
_module = sys.modules[__name__]
del sys
tests = _module
cli = _module
test_classification = _module
test_dpo_train = _module
test_inference = _module
test_kto_train = _module
test_merge_adapters_to_base = _module
test_multimodal_inference = _module
test_multimodal_preprocessing = _module
test_multimodal_train = _module
test_rag_train = _module
test_rm_train = _module
test_rso_sampling = _module
test_sft = _module
conftest = _module
constants = _module
fixtures = _module
integration = _module
test_dummy = _module
test_trainers = _module
unit = _module
test_collators = _module
test_datasets = _module
utils = _module
turbo_alignment = _module
cherry_picks = _module
base = _module
chat = _module
classification = _module
multimodal = _module
rag = _module
rm = _module
app = _module
common = _module
inference = _module
sampling = _module
train = _module
data = _module
io = _module
audio = _module
imagebind = _module
whisper = _module
image = _module
clip = _module
registry = _module
logging = _module
clearml = _module
logger = _module
weights_and_biases = _module
checks = _module
file_utils = _module
from_params = _module
lazy = _module
params = _module
registrable = _module
s3 = _module
checkpoints_handler = _module
singleton = _module
tf = _module
callbacks = _module
sync_ref_model = _module
loaders = _module
model = _module
tokenizer = _module
merge_adapters_to_base = _module
special_tokens_setter = _module
dataset = _module
models = _module
conversation = _module
ddpo = _module
collators = _module
kto = _module
loader = _module
pair_preferences = _module
pair_preference = _module
generators = _module
vllm_chat = _module
metrics = _module
distinctness = _module
diversity = _module
kl = _module
length = _module
meteor = _module
metric = _module
perplexity = _module
retrieval_utility = _module
reward = _module
rouge = _module
self_bleu = _module
modeling = _module
helpers = _module
transformer = _module
heads = _module
postprocessors = _module
preprocessors = _module
impl = _module
trunks = _module
liger_kernels = _module
cross_entropy = _module
geglu = _module
monkey_patch_liger = _module
rope = _module
utils = _module
encoders = _module
lm = _module
projection = _module
projectors = _module
c_abstractor = _module
llava = _module
rag_index = _module
rag_model = _module
rag_output = _module
rag_tokenizer = _module
retriever_model = _module
pipelines = _module
mixin = _module
preprocessing = _module
random = _module
rso = _module
dpo = _module
sft = _module
settings = _module
cherry_pick = _module
datasets = _module
outputs = _module
modality = _module
generation = _module
peft = _module
trainer = _module
trainers = _module
custom_loss = _module
multigpu = _module
create_tutorial_dataset = _module

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


import triton


import triton.language as tl


from torch.nn import CrossEntropyLoss


import torch.nn as nn


import functools


from typing import Callable


@triton.jit
def liger_cross_entropy_kernel(X_ptr, X_stride, Y_ptr, Y_stride, loss_ptr, loss_stride, n_cols, n_non_ignore, ignore_index, label_smoothing: 'tl.constexpr', reduction: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    X_ptr += program_id * X_stride
    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    loss_ptr += program_id * loss_stride
    m = float('-inf')
    d = 0.0
    ori_X_y = tl.load(X_ptr + y)
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float('-inf'))
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float('-inf'))
        if reduction == 'mean':
            X_block = (tl.exp(X_block - m) / d - eps) / n_non_ignore
        else:
            X_block = tl.exp(X_block - m) / d - eps
        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)
    tl.debug_barrier()
    loss = -(ori_X_y - m - tl.log(d))
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    if reduction == 'mean':
        loss = loss / n_non_ignore
    X_y = tl.load(X_ptr + y)
    if reduction == 'mean':
        X_y += -(1 - label_smoothing) / n_non_ignore
    else:
        X_y += -(1 - label_smoothing)
    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


@triton.jit
def element_mul_kernel(X_ptr, X_stride, grad_output_ptr, n_cols, BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    X_ptr += program_id * X_stride
    grad_output = tl.load(grad_output_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


@triton.jit
def _geglu_tanh_forward_kernel(a, b, c, stride, n_cols: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel(dc, a, b, stride, n_cols: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    program_id = tl.program_id(0)
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    db_row = dc_row * geglu_a
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = 0.5 * a_row * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    da_row = dc_row * b_row * (term1 + term2)
    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)


@triton.jit
def _triton_rope(q_ptr, q_row_stride, k_ptr, k_row_stride, cos, cos_row_stride, sin, sin_row_stride, sl, bs: 'tl.constexpr', n_qh: 'tl.constexpr', n_kh: 'tl.constexpr', hd: 'tl.constexpr', pad_n_qh: 'tl.constexpr', pad_n_kh: 'tl.constexpr', pad_hd: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', BACKWARD_PASS: 'tl.constexpr'=False):
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * q_row_stride
    k_ptr = k_ptr + pid * k_row_stride
    cos_row_idx = pid % sl
    cos = cos + cos_row_idx * cos_row_stride
    sin = sin + cos_row_idx * sin_row_stride
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0)
    second_half_q_offsets = first_half_q_offsets + hd // 2
    second_half_k_offsets = first_half_k_offsets + hd // 2
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0)
    if not BACKWARD_PASS:
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
        new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)

