import sys
_module = sys.modules[__name__]
del sys
auto_gptq = _module
eval_tasks = _module
_base = _module
_utils = _module
classification_utils = _module
generation_utils = _module
language_modeling_task = _module
sequence_classification_task = _module
text_summarization_task = _module
modeling = _module
_base = _module
_const = _module
auto = _module
baichuan = _module
bloom = _module
codegen = _module
cohere = _module
decilm = _module
gemma = _module
gemma2 = _module
gpt2 = _module
gpt_bigcode = _module
gpt_neox = _module
gptj = _module
internlm = _module
llama = _module
longllama = _module
minicpm3 = _module
mistral = _module
mixtral = _module
moss = _module
mpt = _module
opt = _module
phi = _module
qwen = _module
qwen2 = _module
rw = _module
stablelmepoch = _module
starcoder2 = _module
xverse = _module
yi = _module
nn_modules = _module
_fused_base = _module
fused_gptj_attn = _module
fused_llama_attn = _module
fused_llama_mlp = _module
qlinear = _module
qlinear_cuda = _module
qlinear_cuda_old = _module
qlinear_exllama = _module
qlinear_exllamav2 = _module
qlinear_hpu = _module
qlinear_marlin = _module
qlinear_qigen = _module
qlinear_triton = _module
qlinear_tritonv2 = _module
triton_utils = _module
custom_autotune = _module
dequant = _module
kernels = _module
mixin = _module
quantization = _module
config = _module
gptq = _module
quantizer = _module
utils = _module
accelerate_utils = _module
data_utils = _module
exllama_utils = _module
import_utils = _module
marlin_utils = _module
modeling_utils = _module
peft_utils = _module
perplexity_utils = _module
generate = _module
intrin = _module
template = _module
generation_speed = _module
perplexity = _module
run_language_modeling_task = _module
run_sequence_classification_task = _module
run_text_summarization_task = _module
peft_adalora_clm_instruction_tuning = _module
peft_adaption_prompt_clm_instruction_tuning = _module
peft_lora_clm_instruction_tuning = _module
basic_usage = _module
basic_usage_gpt_xl = _module
basic_usage_wikitext2 = _module
quant_with_alpaca = _module
setup = _module
tests = _module
bench_autoawq_autogptq = _module
test_awq_compatibility_generation = _module
test_hpu_linear = _module
test_peft_conversion = _module
test_q4 = _module
test_quantization = _module
test_repacking = _module
test_serialization = _module
test_sharded_loading = _module
test_triton = _module

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


import copy


import logging


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import torch


import torch.nn as nn


import math


from logging import getLogger


import numpy as np


import time


import triton


import itertools


import triton.language as tl


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


import random


from itertools import chain


def make_dequant_configs(block_sizes, num_warps):
    configs = []
    for bs, ws in itertools.product(block_sizes, num_warps):
        configs.append(triton.Config({'X_BLOCK': bs}, num_warps=ws))
    return configs


DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8])


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=['numels'])
@triton.jit
def dequant_kernel_248(g_idx_ptr, scales_ptr, qweight_ptr, qzeros_ptr, out_ptr, numels, maxq: 'tl.constexpr', bits: 'tl.constexpr', outfeatures: 'tl.constexpr', num_groups: 'tl.constexpr', X_BLOCK: 'tl.constexpr'):
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures
    elements_per_feature: 'tl.constexpr' = 32 // bits
    g_idx = tl.load(g_idx_ptr + row_idx, None, eviction_policy='evict_last')
    qweights = tl.load(qweight_ptr + (col_idx + outfeatures * (row_idx // elements_per_feature)), None)
    wf_weights = row_idx % elements_per_feature * bits
    wf_zeros = col_idx % elements_per_feature * bits
    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, 'index out of bounds: 0 <= tmp0 < 0')
    groups = tl.where(tmp2, tmp1, g_idx)
    scales = tl.load(scales_ptr + (col_idx + outfeatures * groups), None)
    weights = qweights >> wf_weights
    weights = weights & maxq
    qzero_ncols: 'tl.constexpr' = outfeatures // elements_per_feature
    qzeros = tl.load(qzeros_ptr + (qzero_ncols * groups + col_idx // elements_per_feature), None, eviction_policy='evict_last')
    zeros = qzeros >> wf_zeros
    zeros = zeros & maxq
    zeros = zeros + 1
    weights = weights - zeros
    weights = weights
    weights = scales * weights
    tl.store(out_ptr + x_index, weights, mask=xmask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

