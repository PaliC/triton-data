import sys
_module = sys.modules[__name__]
del sys
generation_speed = _module
perplexity = _module
run_language_modeling_task = _module
run_sequence_classification_task = _module
run_text_summarization_task = _module
run_with_different_backends = _module
basic_usage = _module
basic_usage_autoround = _module
basic_usage_wikitext2 = _module
gptqmodel = _module
eval_tasks = _module
_base = _module
_utils = _module
classification_utils = _module
data_utils = _module
generation_utils = _module
language_modeling_task = _module
sequence_classification_task = _module
text_summarization_task = _module
integration = _module
optimum = _module
constants = _module
data = _module
hf_quantizer_gptq = _module
quantizer = _module
utils = _module
models = _module
_const = _module
auto = _module
baichuan = _module
base = _module
bloom = _module
chatglm = _module
codegen = _module
cohere = _module
dbrx = _module
dbrx_converted = _module
decilm = _module
deepseek_v2 = _module
exaone = _module
gemma = _module
gemma2 = _module
gpt2 = _module
gpt_bigcode = _module
gpt_neox = _module
gptj = _module
granite = _module
grinmoe = _module
internlm = _module
internlm2 = _module
llama = _module
loader = _module
longllama = _module
minicpm = _module
minicpm3 = _module
mistral = _module
mixtral = _module
mllama = _module
moss = _module
mpt = _module
opt = _module
phi = _module
phi3 = _module
qwen = _module
qwen2 = _module
qwen2_moe = _module
rw = _module
stablelmepoch = _module
starcoder2 = _module
writer = _module
xverse = _module
yi = _module
nn_modules = _module
qlinear = _module
bitblas_target_detector = _module
qlinear_bitblas = _module
qlinear_exllamav2 = _module
qlinear_marlin = _module
qlinear_marlin_inference = _module
qlinear_qbits = _module
qlinear_tritonv2 = _module
triton_utils = _module
custom_autotune = _module
dequant = _module
kernels = _module
mixin = _module
quantization = _module
config = _module
gptq = _module
backend = _module
bitblas = _module
device = _module
importer = _module
marlin = _module
model = _module
sglang = _module
vllm = _module
vram = _module
version = _module
setup = _module
model_test = _module
test_model_baichuan = _module
test_model_bloom = _module
test_model_chatglm = _module
test_model_codegen = _module
test_model_cohere = _module
test_model_deci = _module
test_model_deepseekv2_lite = _module
test_model_exaone = _module
test_model_falcon = _module
test_model_gemma = _module
test_model_glm = _module
test_model_gpt2 = _module
test_model_gptbigcode = _module
test_model_gptj = _module
test_model_gptneox = _module
test_model_granite = _module
test_model_grinmoe = _module
test_model_internlm = _module
test_model_internlm2_5 = _module
test_model_llama = _module
test_model_llama2 = _module
test_model_llama3_1 = _module
test_model_llama3_2 = _module
test_model_longllama = _module
test_model_minicpm = _module
test_model_mistral = _module
test_model_mixtral = _module
test_model_moss = _module
test_model_mpt = _module
test_model_opt = _module
test_model_phi_1 = _module
test_model_phi_3 = _module
test_model_qwen1_5 = _module
test_model_qwen2_5 = _module
test_model_stablelm = _module
test_model_starcode2 = _module
test_model_tinyllama = _module
test_model_xverse = _module
test_model_yi = _module
test_dynamic = _module
test_estimate_vram = _module
test_lm_eval = _module
test_lm_head = _module
test_packing = _module
test_perplexity = _module
test_pt = _module
test_q4_bitblas = _module
test_q4_exllama_v2 = _module
test_q4_marlin = _module
test_q4_triton = _module
test_qbits = _module
test_quant_batch = _module
test_quant_formats = _module
test_quant_trust_remote = _module
test_save_loaded_quantized_model = _module
test_serialization = _module
test_sglang = _module
test_sharded = _module
test_tgi = _module
test_transformers_integration = _module
test_triton = _module
test_verify_hash = _module
test_vllm = _module

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


from logging import getLogger


import numpy as np


import torch


import torch.nn as nn


import time


from typing import Dict


import triton


import itertools


import triton.language as tl


from torch.amp import custom_bwd


from torch.amp import custom_fwd


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


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
    weights = weights - zeros
    weights = weights
    weights = scales * weights
    tl.store(out_ptr + x_index, weights, mask=xmask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

