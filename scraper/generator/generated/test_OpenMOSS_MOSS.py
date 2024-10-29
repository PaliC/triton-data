import sys
_module = sys.modules[__name__]
del sys
finetune_moss = _module
configuration_moss = _module
custom_autotune = _module
modeling_moss = _module
quantization = _module
tokenization_moss = _module
models_jittor = _module
generation = _module
load = _module
model = _module
utils = _module
moss_api_demo = _module
moss_cli_demo = _module
moss_cli_demo_jittor = _module
moss_inference = _module
moss_web_demo_gradio = _module
moss_web_demo_streamlit = _module

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


import numpy as np


import torch


import torch.nn as nn


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


import triton.language as tl


class Autotuner(triton.KernelInterface):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, prune_configs_by: 'Dict'=None, nearest_power_of_two: 'bool'=False):
        """
		:param prune_configs_by: a dict of functions that are used to prune configs, fields:
			'perf_model': performance model used to predicate running time with different configs, returns running time
			'top_k': number of configs to bench
			'prune_num_stages_by'(optional): a function used to prune num_stages. It take configs:List[Config] as its input, and returns pruned configs.
			'nearest_power_of_two'(optional): whether to round key arguments to the nearest power of two when caching tuning results
		"""
        if not configs:
            self.configs = [triton.Config({}, num_warps=4, num_stages=2)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.nearest_power_of_two = nearest_power_of_two
        self.cache = {}
        self.hook = lambda args: 0
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]

            def _hook(args):
                for i in self.reset_idx:
                    args[i].zero_()
            self.hook = _hook
        self.arg_names = arg_names
        if prune_configs_by:
            perf_model, top_k = prune_configs_by['perf_model'], prune_configs_by['top_k']
            if 'early_config_prune' in prune_configs_by:
                early_config_prune = prune_configs_by['early_config_prune']
        else:
            perf_model, top_k, early_config_prune = None, None, None
        self.perf_model, self.configs_top_k = perf_model, top_k
        self.early_config_prune = early_config_prune
        self.fn = fn

    def _bench(self, *args, config, **meta):
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}. Make sure that you don't re-define auto-tuned symbols.")
        current = dict(meta, **config.kwargs)

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(self.nargs)
            self.hook(args)
            self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, **current)
        try:
            return triton.testing.do_bench(kernel_call, rep=40)
        except triton.compiler.OutOfResources:
            return float('inf')

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            key = tuple(args[i] for i in self.key_idx)
            if self.nearest_power_of_two:
                key = tuple([(2 ** int(math.log2(x) + 0.5)) for x in key])
            if key not in self.cache:
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.hook(args)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if config.pre_hook is not None:
            config.pre_hook(self.nargs)
        return self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, **kwargs, **config.kwargs)

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {config: self.perf_model(**self.nargs, **kwargs, **config.kwargs, num_stages=config.num_stages, num_warps=config.num_warps) for config in pruned_configs}
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        for config in self.prune_configs(kwargs):
            self.fn.warmup(*args, num_warps=config.num_warps, num_stages=config.num_stages, **kwargs, **config.kwargs)
        self.nargs = None


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, nearest_power_of_two=False):
    """
	Decorator for auto-tuning a :code:`triton.jit`'d function.
	.. highlight:: python
	.. code-block:: python
		@triton.autotune(configs=[
			triton.Config(meta={'BLOCK_SIZE': 128}, num_warps=4),
			triton.Config(meta={'BLOCK_SIZE': 1024}, num_warps=8),
			],
			key=['x_size'] # the two above configs will be evaluated anytime
							# the value of x_size changes
		)
		@triton.jit
		def kernel(x_ptr, x_size, **META):
			BLOCK_SIZE = META['BLOCK_SIZE']
	:note: When all the configurations are evaluated, the kernel will run multiple time.
			This means that whatever value the kernel updates will be updated multiple times.
			To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
			reset the value of the provided tensor to `zero` before running any configuration.
	:param configs: a list of :code:`triton.Config` objects
	:type configs: list[triton.Config]
	:param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
	:type key: list[str]
	:param prune_configs_by: a dict of functions that are used to prune configs, fields:
		'perf_model': performance model used to predicate running time with different configs, returns running time
		'top_k': number of configs to bench
		'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It take configs:List[Config] as its input, and returns pruned configs.
	:param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
	:type reset_to_zero: list[str]
	"""

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, prune_configs_by, nearest_power_of_two)
    return decorator


@autotune(configs=[triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)], key=['M', 'N'], nearest_power_of_two=True)
@triton.jit
def matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales, stride_zeros, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32 // bits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_mask = offs_am[:, None] < M
    b_ptrs = b_ptr + (offs_k[:, None] // infearure_per_bits * stride_bk + offs_bn[None, :] * stride_bn)
    g_ptrs = g_ptr + offs_k
    scales_ptrs = scales_ptr + offs_bn[None, :]
    zeros_ptrs = zeros_ptr + offs_bn[None, :] // infearure_per_bits
    shifter = offs_k % infearure_per_bits * bits
    zeros_shifter = offs_bn % infearure_per_bits * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        g_idx = tl.load(g_ptrs)
        scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)
        zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)
        zeros = zeros >> zeros_shifter[None, :] & maxq
        zeros = zeros + 1
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)
        b = b >> shifter[:, None] & maxq
        b = (b - zeros) * scales
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K // infearure_per_bits * stride_bk
        g_ptrs += BLOCK_SIZE_K
    c = accumulator
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@autotune(configs=[triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)], key=['M', 'K'], nearest_power_of_two=True)
@triton.jit
def trans_matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales, stride_zeros, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, N) float16
    B is of shape (K//8, N) int32
    C is of shape (M, K) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N) float16
    g_ptr is of shape (K) int32
    """
    infearure_per_bits = 32 // bits
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_k = pid % num_pid_in_group // group_size_m
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)
    a_mask = offs_am[:, None] < M
    b_ptrs = b_ptr + (offs_bk[:, None] // infearure_per_bits * stride_bk + offs_n[None, :] * stride_bn)
    g_ptrs = g_ptr + offs_bk
    g_idx = tl.load(g_ptrs)
    scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
    zeros_ptrs = zeros_ptr + offs_n[None, :] // infearure_per_bits + g_idx[:, None] * stride_zeros
    shifter = offs_bk % infearure_per_bits * bits
    zeros_shifter = offs_n % infearure_per_bits * bits
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k in range(0, num_pid_n):
        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)
        zeros = zeros >> zeros_shifter[None, :] & maxq
        zeros = zeros + 1
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)
        b = b >> shifter[:, None] & maxq
        b = (b - zeros) * scales
        b = tl.trans(b)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_N
        b_ptrs += BLOCK_SIZE_N
        scales_ptrs += BLOCK_SIZE_N
        zeros_ptrs += BLOCK_SIZE_N // infearure_per_bits
    c = accumulator
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)

