import sys
_module = sys.modules[__name__]
del sys
qllm = _module
args_config = _module
auto_datasets = _module
auto_model_quantization = _module
custom = _module
m_mpt = _module
run = _module
modeling = _module
base = _module
config = _module
q_layers = _module
compress_weight = _module
custom_autotune = _module
ext_package_checker = _module
quant_linear_awq = _module
quant_linear_gptq = _module
quant_linear_hqq = _module
quant_linear_marlin = _module
quant_linear_onnxruntime = _module
quant_linear_triton = _module
triton_norm = _module
plugin = _module
chatcli = _module
chatio = _module
conversation = _module
generation = _module
inference = _module
perplexity_utils = _module
quantization = _module
_awq_quantizer = _module
_gptq_quantizer = _module
_hqq_quantizer = _module
config_builder = _module
gptq = _module
method = _module
quant_awq = _module
quant_frame_base = _module
quant_gptq = _module
quant_hqq = _module
sequential_layes_awq_config = _module
sequential_layes_gptq_config = _module
utils = _module
comm_utils = _module
datautils = _module
logger = _module
modelutils = _module
onnx = _module
exporter = _module
merge_encoder_decoder = _module
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


import math


import time


from typing import Dict


import triton


import torch


import torch.nn as nn


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch import nn


import triton.language as tl


@triton.jit
def rms_norm_fwd_fused(X, Y, W, stride, N, eps, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0)
        x = tl.where(cols < N, x, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0)
        x_hat = x * rstd
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)

