import sys
_module = sys.modules[__name__]
del sys
main = _module
aq_engine = _module
benchmark_generate_cpu = _module
generate_benchmark = _module
matmul_benchmark = _module
matmul_benchmark_cpu = _module
convert_legacy_model_format = _module
convert_to_hf = _module
finetune = _module
aqlm = _module
inference = _module
inference_kernels = _module
cuda_kernel = _module
dequantization = _module
kernel_selector = _module
numba_kernel = _module
triton_kernel = _module
utils = _module
lmeval = _module
src = _module
aq = _module
beam_search_l2 = _module
beam_search_xtx = _module
configurable_adam = _module
datautils = _module
kmeans = _module
memory_efficient_loss = _module
modelutils = _module
pv_optimizer = _module
pv_utils = _module

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


from typing import Optional


import torch


import triton


import triton.language as tl


from torch.autograd import Function


@triton.autotune(configs=[triton.Config({'UNUSED': 1}, num_stages=num_stages, num_warps=num_warps) for num_stages in (1, 2, 3, 4, 5) for num_warps in (1, 2, 4, 8)], key=['in_features', 'out_features', 'num_codebooks', 'codebook_size', 'out_group_size', 'in_group_size', 'num_input_groups', 'num_input_groups_next_power_of_2', 'compute_in_fp32', 'has_output_scale', 'has_bias'])
@triton.jit
def _aqlm_gemv_simple(input_vec_ptr, output_vec_ptr, codes_ptr, codebooks_ptr, scales_ptr, bias_ptr, in_features: 'tl.constexpr', out_features: 'tl.constexpr', num_codebooks: 'tl.constexpr', codebook_size: 'tl.constexpr', out_group_size: 'tl.constexpr', in_group_size: 'tl.constexpr', num_input_groups: 'tl.constexpr', num_input_groups_next_power_of_2: 'tl.constexpr', compute_in_fp32: 'tl.constexpr', has_output_scale: 'tl.constexpr', has_bias: 'tl.constexpr', UNUSED: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    input_vec = tl.load(input_vec_ptr + tl.arange(0, num_input_groups_next_power_of_2)[:, None, None, None] * in_group_size + tl.arange(0, in_group_size)[None, None, None, :], mask=tl.arange(0, num_input_groups_next_power_of_2)[:, None, None, None] < num_input_groups)
    dtype = input_vec.dtype
    codes_i_ptrs = codes_ptr + pid * num_input_groups * num_codebooks + tl.arange(0, num_input_groups_next_power_of_2)[:, None] * num_codebooks + tl.arange(0, num_codebooks)[None, :]
    codes_i_mask_1d = tl.arange(0, num_input_groups_next_power_of_2) < num_input_groups
    codes_i = tl.load(codes_i_ptrs, mask=codes_i_mask_1d[:, None])
    codes_i = codes_i
    codes_i = codes_i + (codes_i < 0) * codebook_size
    codes_i += tl.arange(0, num_codebooks)[None, :] * codebook_size
    out_group_ix = tl.arange(0, out_group_size)[None, None, :, None]
    in_group_ix = tl.arange(0, in_group_size)[None, None, None, :]
    weight_i_ptrs = codebooks_ptr + codes_i[:, :, None, None] * out_group_size * in_group_size + out_group_ix * in_group_size + in_group_ix
    weights_i = tl.load(weight_i_ptrs, mask=codes_i_mask_1d[:, None, None, None], other=0)
    if compute_in_fp32:
        weights_i = weights_i
        input_vec = input_vec
    output_i = weights_i * input_vec
    if out_group_size == 1:
        output_i = tl.sum(output_i)
    else:
        output_i = tl.sum(output_i, axis=1)
        output_i = tl.sum(output_i, axis=2)
        output_i = tl.sum(output_i, axis=0)
    if has_output_scale:
        output_i *= tl.load(scales_ptr + pid)
    if has_bias:
        output_i += tl.load(bias_ptr + pid)
    if out_group_size == 1:
        tl.store(output_vec_ptr + pid, output_i)
    else:
        tl.store(output_vec_ptr + pid * out_group_size + tl.arange(0, out_group_size), output_i)

