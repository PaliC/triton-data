import sys
_module = sys.modules[__name__]
del sys
accelerated_scan = _module
ref = _module
triton = _module
warp = _module
bench = _module
test_eq = _module

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


from typing import Literal


@triton.jit
def unpack64(merged):
    tl.static_assert(merged.dtype == tl.uint64)
    b = (merged & 4294967295).to(tl.uint32)
    a = (merged >> 32).to(tl.uint32)
    return a, b


@triton.jit
def pack64(a, b):
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a = a.to(dtype=tl.uint32, bitcast=True)
    a = a << 32
    b = b.to(dtype=tl.uint32, bitcast=True)
    return a | b


@triton.jit()
def first_order_op(l, r):
    """
    See https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf Section 1.4.1
    """
    xl, fl = unpack64(l)
    xr, fr = unpack64(r)
    x = xl * fr + xr
    f = fl * fr
    return pack64(x, f)


@triton.jit
def forward_scan(gates, tokens, outputs, SEQUENCE_LENGTH: 'tl.constexpr'):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    strides = tl.arange(0, SEQUENCE_LENGTH) + sequence_id * SEQUENCE_LENGTH
    tokens_ = tl.load(tokens + strides)
    gates_ = tl.load(gates + strides)
    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + strides, output_tokens_)


@triton.jit
def backward_scan(gates, tokens, outputs, SEQUENCE_LENGTH: 'tl.constexpr'):
    sequence_id = tl.num_programs(axis=1) * tl.program_id(axis=0) + tl.program_id(axis=1)
    forward_strides = tl.arange(0, SEQUENCE_LENGTH) + sequence_id * SEQUENCE_LENGTH
    reverse_strides = tl.num_programs(axis=0) * tl.num_programs(axis=1) * SEQUENCE_LENGTH - 1 - forward_strides
    tokens_ = tl.load(tokens + reverse_strides)
    gates_ = tl.load(gates + reverse_strides)
    tuples = pack64(tokens_, gates_)
    output_tuples_ = tl.associative_scan(tuples, axis=0, combine_fn=first_order_op)
    output_tokens_, output_gates_ = unpack64(output_tuples_)
    tl.store(outputs + reverse_strides, output_tokens_)

