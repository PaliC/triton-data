import sys
_module = sys.modules[__name__]
del sys
github_utils = _module
gitutils = _module
label_utils = _module
trymerge = _module
trymerge_explainer = _module
bench_galore_fused_kernels = _module
benchmark_aq = _module
benchmark_fp6 = _module
benchmark_gpu_sparsity = _module
benchmark_hqq = _module
benchmark_low_bit_adam = _module
benchmark_semi_sparse_training = _module
benchmark_uintx = _module
bench_utils = _module
dora_bench = _module
bench_linear_float8 = _module
bench_matmul = _module
bench_multi_gpu = _module
bench_padding = _module
float8_roofline = _module
profile_linear_float8 = _module
utils = _module
fused_benchmark_utils = _module
intmm = _module
print_config_shapes = _module
benchmark_int8mm = _module
pretrain_llama2 = _module
conf = _module
custom_directives = _module
template_tutorial = _module
amg_example = _module
server = _module
smoke_test = _module
convert_hf_checkpoint = _module
create_weight_map = _module
download = _module
hf_eval = _module
setup = _module
test_dora_fusion = _module
test_dora_layer = _module
test_affine_quantized = _module
test_affine_quantized_float = _module
test_affine_quantized_tensor_parallel = _module
test_bitnet = _module
test_bitpacking = _module
test_floatx = _module
test_nf4 = _module
test_uint2 = _module
test_uint4 = _module
test_uintx = _module
test_base = _module
test_compile = _module
test_dtensor = _module
test_fsdp = _module
test_fsdp2 = _module
test_fsdp_compile = _module
test_numerics_integration = _module
memory_analysis_utils = _module
model_configs = _module
profile_memory_usage = _module
profiling_utils = _module
test_hqq_affine = _module
test_triton_mm = _module
test_triton_qkv_fused = _module
test_integration = _module
galore_test_utils = _module
test_autotuner = _module
test_fused_kernels = _module
test_galore_downproj = _module
test_device_spec = _module
test_performance_counter = _module
test_custom_cast = _module
test_mx_linear = _module
test_mx_tensor = _module
test_autoround = _module
test_awq = _module
test_bitpacking_gen = _module
test_low_bit_optim = _module
test_parametrization = _module
test_quantized_training = _module
test_scheduler = _module
test_smoothquant = _module
test_sparse_api = _module
test_sparsifier = _module
test_sparsity_utils = _module
test_spinquant = _module
test_splitk = _module
test_structured_sparsifier = _module
test_galore_quant = _module
test_mixed_precision = _module
test_observer = _module
test_qat = _module
test_quant_api = _module
test_quant_primitives = _module
smoke_tests = _module
test_fast_sparse_training = _module
test_marlin = _module
test_wanda = _module
test_ao_models = _module
test_ops = _module
test_utils = _module
test_gptq_mt = _module
torchao = _module
_executorch_ops = _module
_models = _module
_eval = _module
llama = _module
eval = _module
generate = _module
model = _module
perf_profile = _module
tokenizer = _module
data = _module
eval_combo = _module
metrics = _module
sam2 = _module
automatic_mask_generator = _module
build_sam = _module
modeling = _module
backbones = _module
hieradet = _module
image_encoder = _module
memory_attention = _module
memory_encoder = _module
position_encoding = _module
sam = _module
mask_decoder = _module
prompt_encoder = _module
transformer = _module
sam2_base = _module
sam2_utils = _module
sam2_image_predictor = _module
sam2_video_predictor = _module
amg = _module
misc = _module
transforms = _module
dtypes = _module
affine_quantized_tensor = _module
floatx = _module
nf4tensor = _module
uint4 = _module
uintx = _module
bitpacking = _module
experimental = _module
gen_metal_shader_lib = _module
test_lowbit = _module
quant_api = _module
test_embedding_xbit_quantizer = _module
test_linear_8bit_act_xbit_weight_quantizer = _module
float8 = _module
config = _module
distributed_utils = _module
float8_linear = _module
float8_linear_utils = _module
float8_ops = _module
float8_python_api = _module
float8_scaling_utils = _module
float8_tensor = _module
float8_tensor_parallel = _module
float8_utils = _module
fsdp_utils = _module
inference = _module
roofline_utils = _module
kernel = _module
autotuner = _module
intmm_triton = _module
ops = _module
profiler = _module
device_spec = _module
performance_counter = _module
prototype = _module
autoround = _module
autoround_llm = _module
core = _module
eval_autoround = _module
multi_tensor = _module
awq = _module
api = _module
example = _module
common = _module
profiling_tools = _module
custom_fp_utils = _module
dora = _module
dora_layer = _module
dora_profile = _module
kernels = _module
common = _module
custom_autotune = _module
matmul = _module
smallk = _module
bitnet = _module
uint2 = _module
uintgen = _module
galore = _module
adam_downproj_fused = _module
adam_step = _module
custom_autotune = _module
matmul = _module
quant = _module
optim = _module
galore_torch = _module
hqq = _module
hqq_tinygemm_linear = _module
kernels = _module
mixed_mm = _module
low_bit_optim = _module
adam = _module
cpu_offload = _module
quant_utils = _module
subclass_4bit = _module
subclass_8bit = _module
subclass_fp8 = _module
mx_formats = _module
bench_qdq = _module
config = _module
constants = _module
custom_cast = _module
fp_format_spec = _module
mx_linear = _module
mx_ops = _module
mx_tensor = _module
quantized_training = _module
int8 = _module
int8_mixed_precision = _module
int8_mm = _module
smoothquant = _module
sparsity = _module
FPGM_pruner = _module
pruner = _module
base_structured_sparsifier = _module
lstm_saliency_pruner = _module
match_utils = _module
parametrization = _module
prune_functions = _module
saliency_pruner = _module
scheduler = _module
base_scheduler = _module
cubic_scheduler = _module
lambda_scheduler = _module
sparsifier = _module
base_sparsifier = _module
nearly_diagonal_sparsifier = _module
weight_norm_sparsifier = _module
superblock = _module
benchmark = _module
blocksparse = _module
evaluate = _module
supermask = _module
train = _module
spinquant = _module
_hadamard_matrices = _module
hadamard_utils = _module
splitk = _module
splitk_gemm = _module
GPTQ = _module
GPTQ_MT = _module
quantization = _module
autoquant = _module
dynamic_quant = _module
granularity = _module
linear_activation_quantized_tensor = _module
linear_activation_scale = _module
linear_activation_weight_observer = _module
observer = _module
mixed_precision = _module
BO_acc_modelsize = _module
BO_acc_throughput = _module
scripts = _module
fit = _module
hessian_grad = _module
hessian_vhp = _module
mp_quant_eval = _module
naive_intNwo = _module
qat = _module
_module_swap_api = _module
affine_fake_quantized_tensor = _module
embedding = _module
fake_quantizer = _module
linear = _module
quant_primitives = _module
smoothquant = _module
subclass = _module
unified = _module
utils = _module
weight_only = _module
weight_tensor_linear_activation_quantization = _module
marlin = _module
sparse_api = _module
training = _module
autograd = _module
pointwise_ops = _module
wanda = _module
testing = _module
fsdp2_utils = _module
add_an_op = _module
awq_like = _module
gptq_like = _module
static_quant = _module
developer_api_guide = _module
my_dtype_tensor_subclass = _module
my_trainable_tensor_subclass = _module
print_op_and_shapes = _module
tensor_parallel = _module
huggingface_24sparse_example = _module
bfloat16_code = _module
quant_code = _module
run_vit_b = _module
run_vit_b_quant = _module

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


import pandas as pd


import torch


from triton.testing import do_bench


import copy


import random


from typing import Callable


from typing import Optional


import torch.nn as nn


import torch.nn.functional as F


from torch.profiler import profile


from torch.profiler import ProfilerActivity


from torch.profiler import record_function


import collections


import re


import triton


import numpy as np


import matplotlib.pyplot as plt


import torch.utils.benchmark as benchmark


from torch._inductor import config as inductorconfig


import itertools


import logging


import time


from typing import List


import torch._dynamo.config


import torch._inductor.config


from torch.utils._triton import has_triton


from torch._inductor.utils import run_and_get_code


from torch._dynamo import config


from torch.ao.quantization import MinMaxObserver


from torch.ao.quantization import QConfigMapping


from torch.ao.quantization.quantize_fx import convert_to_reference_fx


from torch.ao.quantization.quantize_fx import prepare_fx


import torch.distributed as dist


from torch import nn


from torch.distributed._composable.fsdp import MixedPrecisionPolicy


from torch.distributed._composable.fsdp import fully_shard


from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


from torch.testing._internal.common_fsdp import FSDPTest


from torch.testing._internal.common_utils import TestCase


from torch.testing._internal.common_utils import instantiate_parametrized_tests


from torch.testing._internal.common_utils import parametrize


from torch.testing._internal.common_utils import run_tests


from torch.testing._internal.distributed._tensor.common_dtensor import ModelArgs


from torch.testing._internal.distributed._tensor.common_dtensor import Transformer


import math


import triton.language as tl


from typing import Dict


from typing import Union


import types


from functools import partial


import torch.autograd.profiler_util


from torch.autograd.profiler import record_function


from torch.cuda.nvtx import range as nvtx_range


from enum import Enum


from enum import unique


from triton.ops.matmul import early_config_prune


from triton.ops.matmul import estimate_matmul_time


from triton.ops.matmul import get_configs_io_bound


from triton.ops.matmul import get_higher_dtype


from triton.runtime import Config


from triton.runtime.cache import default_cache_dir


from triton.runtime.errors import OutOfResources


from triton.runtime.jit import KernelInterface


from triton.runtime import driver


from triton.ops.matmul import init_to_zero


from triton.ops.matmul_perf_model import early_config_prune


from triton.ops.matmul_perf_model import estimate_matmul_time


from triton.language.math import sqrt


from triton.runtime.autotuner import heuristics


from triton import Config


from triton import cdiv


from typing import Any


from typing import NamedTuple


from typing import Tuple


import torch.utils._pytree as pytree


from torch import Tensor


from torch.utils._python_dispatch import TorchDispatchMode


import torch.nn.utils.parametrize as parametrize


from math import inf


from math import nan


from torch._inductor.hooks import run_intermediate_hooks


from torch._inductor.utils import maybe_profile


from torch._inductor.codegen.memory_planning import _align as align


from torch import device


from torch import empty_strided


from torch._inductor.async_compile import AsyncCompile


from torch._inductor.select_algorithm import extern_kernels


from torch._inductor.codegen.multi_kernel import MultiKernelCall


from torch._C import _cuda_getCurrentRawStream as get_raw_stream


import torch._inductor.kernel.mm_common


@triton.jit
def matmul_kernel_with_block_pointers(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % GROUP_M
    pid_n = pid % num_pid_in_group // GROUP_M
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        accumulator += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c = accumulator
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


@triton.jit
def scaled_matmul_kernel_with_block_pointers(a_ptr, b_ptr, c_ptr, s1_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_s1m, stride_s1n, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr', EVEN_K: 'tl.constexpr', ACC_TYPE: 'tl.constexpr'=tl.int32):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = a_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = b_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)
    xindex = idx_n + N * idx_m
    tmp0 = tl.load(s1_ptr + tl.broadcast_to(idx_m, mask.shape), mask, eviction_policy='evict_last')
    tl.store(c_ptr + tl.broadcast_to(xindex, mask.shape), acc * tmp0, mask)


@triton.jit
def _matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', SPLIT_K: 'tl.constexpr', EVEN_K: 'tl.constexpr', GROUP_M: 'tl.constexpr', epilogue_alpha=None, epilogue_beta=None, epilogue_source=None, acc_dtype: 'tl.constexpr'=tl.float32, allow_tf32: 'tl.constexpr'=True, fp8_fast_accum: 'tl.constexpr'=True, AB_DTYPE: 'tl.constexpr'=None, EPILOGUE: 'tl.constexpr'=False):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=_0)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=_0)
        if AB_DTYPE is not None:
            a = a
            b = b
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        else:
            acc += tl.dot(a, b, out_dtype=acc_dtype, allow_tf32=allow_tf32)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if EPILOGUE:
        if epilogue_alpha is not None:
            acc = epilogue_alpha * acc
        if epilogue_source is not None:
            epilogue_src = tl.load(epilogue_source + rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            if epilogue_beta is not None:
                epilogue_src = epilogue_src * epilogue_beta
            acc = acc + epilogue_src
    acc = acc
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


_AUTOTUNE_TOPK = 10


class Autotuner(KernelInterface):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, prune_configs_by: 'Dict'=None, warmup=25, rep=100):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=2, num_ctas=1)]
        else:
            self.configs = configs
        self.key_idx = [arg_names.index(k) for k in key]
        self.cache = {}
        self.arg_names = arg_names
        self.reset_idx = []
        if reset_to_zero is not None:
            self.reset_idx = [arg_names.index(k) for k in reset_to_zero]
        self.restore_idx = []
        if restore_value is not None:
            self.restore_idx = [arg_names.index(k) for k in restore_value]
        self.pre_hook = lambda args, reset_only=False: 0
        self.post_hook = lambda args: 0
        if len(self.reset_idx) > 0 or len(self.restore_idx) > 0:

            def _pre_hook(args, reset_only=False):
                for i in self.reset_idx:
                    args[i].zero_()
                if not reset_only:
                    self.restore_copies = [args[i].clone() for i in self.restore_idx]
            self.pre_hook = _pre_hook
        if len(self.restore_idx) > 0:

            def _post_hook(args):
                for i, j in enumerate(self.restore_idx):
                    args[j].copy_(self.restore_copies[i])
                self.restore_copies = []
            self.post_hook = _post_hook
        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get('perf_model', self.perf_model)
            self.configs_top_k = prune_configs_by.get('top_k', self.configs_top_k)
            self.early_config_prune = prune_configs_by.get('early_config_prune', self.early_config_prune)
        self.fn = fn
        self.num_warmups = warmup
        self.num_reps = rep
        self.kernel_name = self._find_kernel_name()

    def _find_kernel_name(self):
        try:
            kernel_name = self.fn.__name__
        except AttributeError:
            try:
                kernel_name = self.fn.fn.__name__
            except:
                kernel_name = self.fn.__name__
        return kernel_name

    def _get_key_combination(self, args, as_str=True, sep=' '):
        key_vals = [f'{self.arg_names[i]}={args[i]}' for i in self.key_idx]
        return f'{sep}'.join(key_vals) if as_str else key_vals

    def _bench(self, *args, config, **meta):
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}. Make sure that you don't re-define auto-tuned symbols.")
        current = dict(meta, **config.kwargs)
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(args)
            self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas, **current)
            self.post_hook(args)
        try:
            return do_bench(kernel_call, warmup=self.num_warmups, rep=self.num_reps, quantiles=(0.5, 0.2, 0.8))
        except OutOfResources:
            return [float('inf'), float('inf'), float('inf')]

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
            key = [_args[i] for i in self.key_idx]
            for arg in _args:
                if hasattr(arg, 'dtype'):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                logger.debug('Cache miss!\n')
                logger.info(f'\n==== Autotune ====\nRunning autotune for {self.kernel_name} for {len(self.configs)} total configs for key combination {self._get_key_combination(args)}...')
                pruned_configs = self.prune_configs(kwargs)
                logger.info(f'\nNum configs after pruning {len(pruned_configs)}')
                bench_start = time.time()
                timings = {}
                for config in pruned_configs:
                    timings[config] = self._bench(*args, config=config, **kwargs)
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.pre_hook(args, reset_only=True)
                self.configs_timings = timings
                sorted_timings = dict(sorted(timings.items(), key=lambda x: np.mean(x[1])))
                _key_suffix = self._get_key_combination(args, sep='-')
                autotune_file = f'autotune_{self.kernel_name}_{_key_suffix}.log'
                autotune_log_path = os.path.join(default_cache_dir(), autotune_file)
                logger.info(f'\nFinished autotune, writing log to {autotune_log_path}')
                with open(f'{autotune_log_path}', 'w') as f:
                    f.write(f' ==== Autotune Results ====\nKernel name: {self.kernel_name}\nArgs: {self.arg_names}\nKeys: {self._get_key_combination(args)}\n')
                    f.write(f'\nPruned configs:\n')
                    for cfg in pruned_configs:
                        f.write(f'{cfg}\n')
                    f.write(f'Timings:\n')
                    for cfg, timing in sorted_timings.items():
                        f.write(f'{cfg} {timing} \n')
                    f.write(f'Best config: {self.cache[key]}\n')
            config = self.cache[key]
            logger.debug(f'\nAutotune: Cache hit! Running best config...')
        else:
            config = self.configs[0]
        self.best_config = config
        logger.info(f'\nAutotune Best Config: {config}\n')
        full_nargs = {**self.nargs, **kwargs, **self.best_config.kwargs}
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)
        ret = self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, num_ctas=config.num_ctas, **kwargs, **config.kwargs)
        self.nargs = None
        return ret

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {config: self.perf_model(**self.nargs, **kwargs, **config.kwargs, num_stages=config.num_stages, num_warps=config.num_warps, num_ctas=config.num_ctas) for config in pruned_configs}
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(*args, num_warps=config.num_warps, num_ctas=config.num_ctas, num_stages=config.num_stages, **kwargs, **config.kwargs))
        self.nargs = None
        return ret


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, warmup=25, rep=100):
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
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.
    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param warmup: Warmup time (in ms) to pass to benchmarking, defaults to 25.
    :type warmup: int
    :param rep: Repetition time (in ms) to pass to benchmarking, defaults to 100.
    :type rep: int
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, prune_configs_by, warmup, rep)
    return decorator


def get_compute_bound_configs():
    configs = [Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8), Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2), Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8), Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4), Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2)]
    return configs


def get_small_k_configs():
    configs = get_compute_bound_configs() + get_configs_io_bound()
    KEYS_TO_REMOVE = ['BLOCK_K', 'SPLIT_K']
    for cfg in configs:
        for key in KEYS_TO_REMOVE:
            del cfg.kwargs[key]
    return configs


def small_k_early_config_prune(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    dtsize = named_args['A'].element_size()
    dtype = named_args['A'].dtype
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = kw['BLOCK_M'], kw['BLOCK_N'], named_args['K'], config.num_stages
        max_shared_memory = driver.active.utils.get_device_properties(device)['max_shared_mem']
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = kw['BLOCK_M'], kw['BLOCK_N'], named_args['K'], config.num_warps, config.num_stages
        key = BLOCK_M, BLOCK_N, BLOCK_K, num_warps
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]
    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = k
        if capability[0] >= 8:
            mmas = BLOCK_M * BLOCK_N * BLOCK_K / (16 * 8 * 16)
            mma_cycles = mmas / min(4, num_warps) * 8
            ldgsts_latency = 300
            optimal_num_stages = ldgsts_latency / mma_cycles
            nearest = heapq.nsmallest(2, v, key=lambda x: 10 + abs(x[1] - optimal_num_stages) if x[1] - optimal_num_stages < 0 else x[1] - optimal_num_stages)
            for n in nearest:
                pruned_configs.append(n[0])
        else:
            random_config = v[0][0]
            random_config.num_stages = 2
            pruned_configs.append(random_config)
    return pruned_configs


@autotune(get_small_k_configs(), key=['M', 'N', 'K'], prune_configs_by={'early_config_prune': small_k_early_config_prune, 'perf_model': estimate_matmul_time, 'top_k': _AUTOTUNE_TOPK})
@triton.jit
def _mm_small_k_kernel(A, B, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, acc_dtype: 'tl.constexpr', input_precision: 'tl.constexpr', fp8_fast_accum: 'tl.constexpr', BLOCK_K: 'tl.constexpr', AB_DTYPE: 'tl.constexpr', BLOCK_M: 'tl.constexpr'=256, BLOCK_N: 'tl.constexpr'=64, C=None, stride_cm=None, stride_cn=None, Norm2=None, Source=None, stride_sourcem=None, stride_sourcen=None, Magnitude=None, ADD_SOURCE: 'tl.constexpr'=False, EPILOGUE_NORM: 'tl.constexpr'=False, EPILOGUE_MAGNITUDE: 'tl.constexpr'=False, STORE_ACC: 'tl.constexpr'=False):
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    a = tl.load(A)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    rn = tl.arange(0, BLOCK_N)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    if STORE_ACC:
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    if ADD_SOURCE:
        Source = Source + (rm[:, None] * stride_sourcem + rn[None, :] * stride_sourcen)
    if EPILOGUE_NORM:
        norm_vec = tl.zeros((BLOCK_M,), dtype=acc_dtype)
    if EPILOGUE_MAGNITUDE:
        Magnitude = Magnitude + ram
    mask_m = rm < M
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        b = tl.load(B)
        if AB_DTYPE is not None:
            a = a
            b = b
        if fp8_fast_accum:
            acc = tl.dot(a, b, acc, out_dtype=acc_dtype, input_precision=input_precision)
        else:
            acc = tl.dot(a, b, out_dtype=acc_dtype, input_precision=input_precision)
        if ADD_SOURCE:
            mask_n = (n * BLOCK_N + rn < N)[None, :]
            source = tl.load(Source, mask=mask_m[:, None] & mask_n)
            acc += source
            Source += BLOCK_N * stride_sourcen
        if EPILOGUE_NORM:
            norm_vec += tl.sum(acc * acc, axis=1)
        if STORE_ACC:
            mask_n = (n * BLOCK_N + rn < N)[None, :]
            tl.store(C, acc, mask=mask_m[:, None] & mask_n)
            C += BLOCK_N * stride_cn
        B += BLOCK_N * stride_bn
    if EPILOGUE_NORM:
        Norm2 = Norm2 + rm
        norm_vec = tl.rsqrt(norm_vec)
        if EPILOGUE_MAGNITUDE:
            magnitude = tl.load(Magnitude, mask=mask_m)
            norm_vec *= magnitude
        tl.store(Norm2, norm_vec, mask=mask_m)


def get_adam_heuristics():
    return {'USE_MASK': lambda args: args['numels'] % args['BLOCK_SIZE'] != 0}


def get_configs_for_adam(num_warps=[2, 4, 8], block_sizes=[512, 1024, 2048]):
    configs = []
    for w in num_warps:
        for bs in block_sizes:
            configs.append(Config({'BLOCK_SIZE': bs}, num_warps=w))
    return configs


@triton.jit
def _dequant_kernel(q_idx_ptr, absmax_ptr, qmap_ptr, dq_ptr, stride_qm, stride_qn, M, N, GROUP_SIZE: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rm[:, None] * stride_qm + rn[None, :] * stride_qn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.static_print(offsets)
    group_offsets = offsets // GROUP_SIZE
    tl.static_print('group_offsets', group_offsets)
    q_idx = tl.load(q_idx_ptr + offsets, mask=mask)
    tl.static_print(q_idx)
    q_vals = tl.load(qmap_ptr + q_idx)
    absmax = tl.load(absmax_ptr + group_offsets, mask=group_offsets < M * N // GROUP_SIZE)
    dq = q_vals * absmax
    tl.store(dq_ptr + offsets, dq, mask=mask)


@triton.heuristics(values={'USE_MASK': lambda args: args['numels'] % args['BLOCK_SIZE'] != 0, 'NUM_GROUPS': lambda args: triton.cdiv(args['numels'], args['BLOCK_SIZE'])})
@triton.jit
def _quantize_blockwise_kernel(t_ptr, cutoffs_ptr, q_ptr, absmax_ptr, norm_ptr, numels, BLOCK_SIZE: 'tl.constexpr', NUM_BUCKETS: 'tl.constexpr', USE_MASK: 'tl.constexpr', NUM_GROUPS: 'tl.constexpr', RETURN_NORM: 'tl.constexpr'=False):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = None
    absmax_mask = None
    if USE_MASK:
        mask = offsets < numels
        absmax_mask = pid < NUM_GROUPS
    t = tl.load(t_ptr + offsets, mask=mask)
    absmax = tl.max(tl.abs(t), axis=0)
    normalized = t / absmax
    cutoffs = tl.load(cutoffs_ptr + tl.arange(0, NUM_BUCKETS))
    q = tl.reshape(normalized, (BLOCK_SIZE, 1)) > cutoffs
    q = q
    q = tl.sum(q, axis=1)
    tl.store(q_ptr + offsets, q, mask=mask)
    tl.store(absmax_ptr + pid, absmax, mask=absmax_mask)
    if RETURN_NORM:
        tl.store(norm_ptr + offsets, normalized, mask=mask)


@triton.jit
def _mixed_mm_kernel(A, B, scales_ptr, zeros_ptr, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scale_k, stride_scale_n, IS_BFLOAT16: 'tl.constexpr', QGROUP_SIZE: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', SPLIT_K: 'tl.constexpr', EVEN_K: 'tl.constexpr', TRANSPOSED: 'tl.constexpr'=False, GROUP_M: 'tl.constexpr'=8, acc_dtype: 'tl.constexpr'=tl.float32, input_precision: 'tl.constexpr'='ieee', fp8_fast_accum: 'tl.constexpr'=False, DEBUG: 'tl.constexpr'=False):
    """Mixed matmul kernel

    A has shape (M, K) and is float16, bfloat16, or float32

    B is i4 / s4 and has shape (K // 2, N) and is packed as uint8 / int8. See `packed_2xint4` for details.

    Scales and zeros are of shape (NUM_GROUPS, N) and are same dtype as A, where NUM_GROUPS = (K // QGROUP_SIZE)
    QGROUP_SIZE should be a multiple of BLOCK_K such that a vector of scales / zeros is loaded and broadcasted to block shape
    per mainloop iteration.

    In the transposed case, A is M x N and B is K x N, and we reduce along "N":
    - TLDR: we are loading rows of A and B blocks at a time, dequantizing and transposing each block of B to achieve the overall
    effect of a transposed matmul. This is necessary to perform a transposed matmul without unpacking and repacking the B matrix.
        - Indexing remains the same for A (the reduction dim (BLK_K / K) corresponds to axis 1 of A -- "N" above)
            - We load a BLK_M x BLK_K block of A
        - Indexing for B is now flipped: N <-> K
            - We load BLK_N x BLK_K block of B (remembering that the reduction dimension is axis 1 of B)
            - We dequantize and transpose to BLK_K x BLK_N
            - scale / zero indexing also change, since we are now iterating along the non-grouping dim within the mac loop and along
            the grouping dim across blocks.
        - Each mac loop calculates BLK_M x BLK_N -> M x "N"(= K)
        - Within the mac loop for each block, we iterate along axis=1 for **both** A and B since axis = 1 is now the reduction dim for B.

    NOTE: Assumes that the quantization grouping was done along the K dimension originally (i.e., QGROUP_SIZE consecutive elements
    of original weight matrix in the K dimension were grouped together when calculating min / max scaling factors).
    """
    if not TRANSPOSED:
        tl.static_assert(QGROUP_SIZE % BLOCK_K == 0)
    else:
        tl.static_assert(QGROUP_SIZE % BLOCK_N == 0)
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    if not DEBUG:
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm
    rak = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    if not TRANSPOSED:
        rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            rbn = rn
        rbk = pid_z * BLOCK_K // 2 + tl.arange(0, BLOCK_K // 2)
    else:
        rn = (pid_n * BLOCK_N // 2 + tl.arange(0, BLOCK_N // 2)) % N
        if not DEBUG:
            rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N // 2), BLOCK_N // 2)
        else:
            rbn = rn
        rbk = rak
    A = A + (ram[:, None] * stride_am + rak[None, :] * stride_ak)
    if not TRANSPOSED:
        B = B + (rbk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    else:
        B = B + (rbn[:, None] * stride_bk + rbk[None, :] * stride_bn)
    if not TRANSPOSED:
        offsets_scale_n = pid_n * stride_scale_n * BLOCK_N + tl.arange(0, BLOCK_N) * stride_scale_n
    else:
        scale_offset_k = pid_n * BLOCK_N * stride_scale_k // QGROUP_SIZE
        offsets_scale_n = tl.arange(0, BLOCK_K) * stride_scale_n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            qb = tl.load(B)
        else:
            k_remaining_a = K - k * (BLOCK_K * SPLIT_K)
            if not TRANSPOSED:
                k_remaining_b = K - k * (BLOCK_K * SPLIT_K) // 2
            else:
                k_remaining_b = K - k * (BLOCK_K * SPLIT_K)
            _0 = tl.zeros((1, 1), dtype=C.dtype.element_ty)
            a = tl.load(A, mask=rak[None, :] < k_remaining_a, other=_0)
            qb = tl.load(B, mask=rbk[:, None] < k_remaining_b, other=_0)
        if not TRANSPOSED:
            scale_offset_k = k * BLOCK_K * SPLIT_K * stride_scale_k // QGROUP_SIZE
        else:
            offsets_scale_n = k * stride_scale_n * BLOCK_K + tl.arange(0, BLOCK_K) * stride_scale_n
        scales = tl.load(scales_ptr + offsets_scale_n + scale_offset_k)
        zeros = tl.load(zeros_ptr + offsets_scale_n + scale_offset_k)
        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        qb_lo = qb << _4_i8 >> _4_i8
        qb_hi = qb >> _4_i8
        if IS_BFLOAT16:
            dq_b = tl.join(qb_lo.to(tl.float16), qb_hi.to(tl.float16)).permute(0, 2, 1)
        else:
            dq_b = tl.join(qb_lo, qb_hi).permute(0, 2, 1)
        if not TRANSPOSED:
            dq_b = dq_b.reshape(BLOCK_K, BLOCK_N)
        else:
            dq_b = dq_b.reshape(BLOCK_N, BLOCK_K)
        zeros = zeros[None, :]
        scales = scales[None, :]
        dq_b = (dq_b - zeros) * scales
        if TRANSPOSED:
            dq_b = tl.trans(dq_b)
        if fp8_fast_accum:
            acc = tl.dot(a, dq_b, acc, out_dtype=acc_dtype, input_precision=input_precision)
        else:
            acc += tl.dot(a, dq_b, out_dtype=acc_dtype, input_precision=input_precision)
        A += BLOCK_K * SPLIT_K * stride_ak
        if not TRANSPOSED:
            B += BLOCK_K * SPLIT_K * stride_bk // 2
        else:
            B += BLOCK_K * SPLIT_K * stride_bn
    acc = acc
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


@triton.jit
def grouped_launch(pid, m, n, block_m: 'tl.constexpr', block_n: 'tl.constexpr', group_m: 'tl.constexpr'):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + pid % group_size
    pid_n = pid % width // group_size
    return pid_m, pid_n


@triton.jit
def gemm_split_k_kernel(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, scale_a, scale_b, m, n, k, block_m: 'tl.constexpr', block_n: 'tl.constexpr', block_k: 'tl.constexpr', split_k: 'tl.constexpr', group_m: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k * split_k)
    pid_m, pid_n = grouped_launch(pid, m, n, block_m, block_n, group_m)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_k * block_k + tl.arange(0, block_k)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_ in range(0, grid_k):
        k_remaining = k - k_ * (block_k * split_k)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += block_k * split_k * stride_bk
    acc = scale_a * scale_b * acc
    acc
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m < m)[:, None] & (offs_n < n)[None, :]
    tl.atomic_add(c_ptrs, acc, mask=mask)

