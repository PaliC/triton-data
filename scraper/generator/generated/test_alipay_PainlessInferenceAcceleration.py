import sys
_module = sys.modules[__name__]
del sys
pia = _module
lookahead = _module
benchmarks = _module
benchmark = _module
chatglm_benchmark = _module
codellama_benchmark = _module
glm_benchmark = _module
llama_benchmark = _module
pia_lantency = _module
preprocess_sample = _module
trie_benchmark = _module
vllm_latency = _module
common = _module
lookahead_cache = _module
lookahead_generation_utils = _module
pretrained_model = _module
pretrained_model_batch = _module
csrc = _module
triton = _module
rms_norm = _module
dataset = _module
examples = _module
baichuan2_13b_example = _module
baichuan2_7b_batch_example = _module
baichuan2_7b_example = _module
baichuan_13b_example = _module
baichuan_7b_example = _module
bloom_example = _module
chatglm3_example = _module
chatglm_example = _module
codellama_example = _module
glm_batch_example = _module
glm_example = _module
gpt2_example = _module
gptj_example = _module
internlm_example = _module
llama_batch_example = _module
llama_example = _module
llama_flash_example = _module
llama_stream_example = _module
mistral_example = _module
mixtral_example = _module
mixtral_quant_example = _module
opt_batch_example = _module
opt_example = _module
qwen_example = _module
qwen_quant_example = _module
models = _module
baichuan2_13b = _module
configuration_baichuan = _module
generation_utils = _module
handler = _module
modeling_baichuan = _module
quantizer = _module
tokenization_baichuan = _module
baichuan2_7b = _module
modeling_baichuan_batch = _module
baichuan_13b = _module
baichuan_7b = _module
bloom = _module
modeling_bloom = _module
chatglm = _module
configuration_chatglm = _module
modeling_chatglm = _module
tokenization_chatglm = _module
chatglm3 = _module
glm = _module
configuration_glm = _module
modeling_glm = _module
modeling_glm_batch = _module
tokenization_glm = _module
gpt2 = _module
modeling_gpt2 = _module
gptj = _module
modeling_gptj = _module
internlm = _module
configuration_internlm = _module
modeling_internlm2 = _module
tokenization_internlm = _module
llama = _module
modeling_llama = _module
modeling_llama_batch = _module
modeling_llama_flash = _module
modeling_llama_fuse = _module
mistral = _module
configuration_mistral = _module
modeling_mistral = _module
mixtral = _module
configuration_mixtral = _module
modeling_mixtral = _module
opt = _module
modeling_opt = _module
modeling_opt_batch = _module
qwen = _module
configuration_qwen = _module
modeling_qwen = _module
qwen_generation_utils = _module
tokenization_qwen = _module
test_lookahead_cache = _module
test_triton_rms_norm = _module
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


import torch


import triton


import triton.language as tl


import math


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import torch.nn.functional as F


import torch.utils.checkpoint


from torch import nn


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.nn import MSELoss


import warnings


@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr, stride_x_batch, stride_x_m, stride_x_k, stride_rms_w, stride_out_batch, stride_out_m, stride_out_k, N_SIZE: 'tl.constexpr', eps: 'tl.constexpr', BLOCK_N_SIZE: 'tl.constexpr'):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        xf = x
        var += xf * xf
    var = tl.sum(var, axis=0) / N_SIZE
    std = tl.sqrt(var + eps)
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        x_new = x / std
        out = x_new * rms_w_offset
        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)

