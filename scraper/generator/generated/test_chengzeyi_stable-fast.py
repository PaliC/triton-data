import sys
_module = sys.modules[__name__]
del sys
optimize_sd15_with_controlnet_and_ip_adapter = _module
reproduce_vae_segfault = _module
optimize_instant_id_pipeline = _module
optimize_lcm_lora = _module
optimize_lcm_pipeline = _module
optimize_stable_diffusion_pipeline = _module
optimize_stable_video_diffusion_pipeline = _module
optimize_train_text_to_image_lora = _module
setup = _module
sfast = _module
compilers = _module
diffusion_pipeline_compiler = _module
stable_diffusion_pipeline_compiler = _module
cuda = _module
graphs = _module
dynamo = _module
backends = _module
registry = _module
sfast_jit = _module
hooks = _module
module_jit_hook = _module
jit = _module
overrides = _module
passes = _module
triton_passes = _module
trace_helper = _module
utils = _module
libs = _module
diffusers = _module
image_processor = _module
xformers_attention = _module
xformers = _module
profile = _module
auto_profiler = _module
cprofile = _module
pretty_profile = _module
triton = _module
modules = _module
diffusers = _module
native = _module
patch = _module
ops = _module
activation = _module
conv = _module
copy = _module
group_norm = _module
layer_norm = _module
utils = _module
torch_ops = _module
aot_printer = _module
compute_precision = _module
copy_func = _module
custom_python_operator = _module
env = _module
flat_tensors = _module
gpu_device = _module
memory_format = _module
term_image = _module
climage = _module
image_to_ansi = _module
imgcat = _module
kdtree = _module
torch_dispatch = _module
tests = _module
test_stable_diffusion_pipeline_compiler = _module
conftest = _module
test_graphs = _module
test_trace_helper = _module
operators = _module
test_cudnn_convolution = _module
test_cutlass_dual_linear = _module
test_cutlass_qlinear = _module
test_torch_ops = _module
test_torch_dispatch = _module

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


import numpy as np


import inspect


import time


import logging


import math


import random


import torch.nn.functional as F


import torch.utils.checkpoint


from torchvision import transforms


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CUDNN_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import functools


import triton.language as tl


import torch.nn as nn


from torch._prims_common import suggest_memory_format


import triton


from itertools import product


import copy


@triton.jit
def identity(x):
    return x


@triton.jit
def silu(x):
    return x * tl.sigmoid(x.to(tl.float32))


@triton.jit
def relu(x):
    return tl.max(x, 0.0)


@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def early_config_prune(configs, named_args):
    from triton.runtime import driver
    from triton.compiler.compiler import get_architecture_descriptor
    device = torch.cuda.current_device()
    cc = get_architecture_descriptor(None)
    dtsize = named_args['x'].element_size()
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = kw['BLOCK_M'], kw['BLOCK_N'], kw['BLOCK_K'], config.num_stages
        max_shared_memory = driver.utils.get_device_properties(device)['max_shared_mem']
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs
    configs_map = {}
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = kw['BLOCK_M'], kw['BLOCK_N'], kw['BLOCK_K'], config.num_warps, config.num_stages
        key = BLOCK_M, BLOCK_N, BLOCK_K, num_warps
        if key in configs_map:
            configs_map[key].append((config, num_stages))
        else:
            configs_map[key] = [(config, num_stages)]
    pruned_configs = []
    for k, v in configs_map.items():
        BLOCK_M, BLOCK_N, BLOCK_K, num_warps = k
        if cc >= 80:
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


def estimate_conv_time(num_warps, num_stages, x, BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W, BLOCK_M, BLOCK_K, BLOCK_N, debug=False, **kwargs):
    """return estimated running time in ms
    = max(compute, loading) + store"""
    import triton
    import triton._C.libtriton.triton as _triton
    from triton.ops.matmul_perf_model import get_dram_gbps as get_dram_gbps
    from triton.ops.matmul_perf_model import get_tflops as get_tflops
    backend = _triton.runtime.backend.CUDA
    device = torch.cuda.current_device()
    dtype = x.dtype
    dtsize = x.element_size()
    M = BATCH * OUT_H * OUT_W
    N = KERNEL_N
    K = KERNEL_H * KERNEL_W * IN_C
    num_cta_m = triton.cdiv(M, BLOCK_M)
    num_cta_n = triton.cdiv(N, BLOCK_N)
    num_cta_k = 1
    num_ctas = num_cta_m * num_cta_n * num_cta_k
    M, N = max(M, BLOCK_M), max(N, BLOCK_N)
    total_ops = 2 * M * N * K / (1024 * 1024 * 1024)
    tput = get_tflops(backend, device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput
    num_sm = _triton.runtime.num_sm(backend, device)
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)
    dram_bw = get_dram_gbps(backend, device) * (active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05)
    l2_bw = dram_bw * 4
    load_a_dram = M * K * dtsize * (1 + 0.2 * (num_cta_n - 1))
    load_a_l2 = M * K * dtsize * 0.8 * (num_cta_n - 1)
    load_b_dram = N * K * dtsize * (1 + 0.2 * (num_cta_m - 1))
    load_b_l2 = N * K * dtsize * 0.8 * (num_cta_m - 1)
    total_dram = (load_a_dram + load_b_dram) / (1024 * 1024)
    total_l2 = (load_a_l2 + load_b_l2) / (1024 * 1024)
    load_ms = total_dram / dram_bw + total_l2 / l2_bw
    store_bw = dram_bw * 0.6
    store_c_dram = M * N * dtsize / (1024 * 1024)
    store_ms = store_c_dram / store_bw
    total_time_ms = max(compute_ms, load_ms) + store_ms
    if debug:
        None
    return total_time_ms


def conv_heuristics():
    configs = [triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=2), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=2), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=2)]
    key = ['BATCH', 'IN_C', 'IN_H', 'IN_W', 'KERNEL_N', 'KERNEL_H', 'KERNEL_W', 'OUT_H', 'OUT_W', 'stride_h', 'stride_w', 'padding_h', 'padding_w', 'dilation_h', 'dilation_w', 'output_padding_h', 'output_padding_w', 'groups']
    prune_configs_by = {'early_config_prune': early_config_prune, 'perf_model': estimate_conv_time, 'top_k': 10}
    return triton.autotune(configs, key, prune_configs_by=prune_configs_by)


@conv_heuristics()
@triton.jit
def _kernel_delta_x_hwc(x, w, bias, y, stride_xn, stride_xc, stride_xh, stride_xw, stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc, stride_yh, stride_yw, delta_xh_ptr, delta_xw_ptr, delta_xc_ptr, BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w, groups, ACC_TYPE: 'tl.constexpr', CONV1X1_NHWC: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_H: 'tl.constexpr', WITH_BIAS: 'tl.constexpr'):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)
    CRS = IN_C * KERNEL_H * KERNEL_W
    if not CONV1X1_NHWC:
        delta_xh_ptrs = delta_xh_ptr + off_x_crs
        delta_xw_ptrs = delta_xw_ptr + off_x_crs
        delta_xc_ptrs = delta_xc_ptr + off_x_crs
        delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
        delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
        off_x_crs_unpacked = delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
        delta_xh = 0
        delta_xw = 0
    mask_x = (off_x_n < BATCH)[:, None] & (off_x_crs < CRS)[None, :] & (off_x_h[:, None] + delta_xh[None, :] >= 0) & (off_x_h[:, None] + delta_xh[None, :] < IN_H) & (off_x_w[:, None] + delta_xw[None, :] >= 0) & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):
        acc += tl.dot(matrix_x, matrix_w, out_dtype=ACC_TYPE)
        w_ptrs += BLOCK_K
        off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
        if not CONV1X1_NHWC:
            delta_xh_ptrs += BLOCK_K
            delta_xw_ptrs += BLOCK_K
            delta_xc_ptrs += BLOCK_K
            delta_xh = tl.load(delta_xh_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xw = tl.load(delta_xw_ptrs, mask=off_x_crs < CRS, other=0)
            delta_xc = tl.load(delta_xc_ptrs, mask=off_x_crs < CRS, other=0)
            off_x_crs_unpacked = delta_xh * stride_xh + delta_xw * stride_xw + delta_xc * stride_xc
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            x_ptrs += BLOCK_K
        mask_x = (off_x_n < BATCH)[:, None] & (off_x_crs < CRS)[None, :] & (off_x_h[:, None] + delta_xh[None, :] >= 0) & (off_x_h[:, None] + delta_xh[None, :] < IN_H) & (off_x_w[:, None] + delta_xw[None, :] >= 0) & (off_x_w[:, None] + delta_xw[None, :] < IN_W)
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
    if WITH_BIAS:
        acc += tl.load(bias + off_y_k)[None, :]
    acc = acc
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w
    y_ptrs = y + off_y_n[:, None] * stride_yn + off_y_h[:, None] * stride_yh + off_y_w[:, None] * stride_yw + off_y_k[None, :] * stride_yc
    mask_y = (off_y_n < BATCH)[:, None] & (off_y_h < OUT_H + output_padding_h)[:, None] & (off_y_w < OUT_W + output_padding_w)[:, None] & (off_y_k < KERNEL_N)[None, :]
    tl.store(y_ptrs, acc, mask=mask_y)
    return


@conv_heuristics()
@triton.jit
def _kernel_delta_x(x, w, bias, y, stride_xn, stride_xc, stride_xh, stride_xw, stride_wn, stride_wc, stride_wh, stride_ww, stride_yn, stride_yc, stride_yh, stride_yw, delta_x_ptr, BATCH, IN_C, IN_H, IN_W, KERNEL_N, KERNEL_H, KERNEL_W, OUT_H, OUT_W, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, output_padding_h, output_padding_w, groups, ACC_TYPE: 'tl.constexpr', CONV1X1_NHWC: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_H: 'tl.constexpr', WITH_BIAS: 'tl.constexpr'):
    """
    each program instance computes a [BLOCK_BATCH, BLOCK_N, BLOCK_H, BLOCK_W] block of y
    """
    pid_nhw = tl.program_id(0)
    pid_k = tl.program_id(1)
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w
    off_x_n = off_y_n
    off_x_h = off_y_h * stride_h - padding_h
    off_x_w = off_y_w * stride_w - padding_w
    off_x_nhw = off_x_n * stride_xn + off_x_h * stride_xh + off_x_w * stride_xw
    off_x_crs = tl.arange(0, BLOCK_K)
    CRS = IN_C * KERNEL_H * KERNEL_W
    if not CONV1X1_NHWC:
        delta_x_ptrs = delta_x_ptr + off_x_crs
        off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS)
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
    else:
        x_ptrs = x + off_x_nhw[:, None] + off_x_crs[None, :]
    mask_x = ((off_x_n < BATCH) & (off_x_h >= 0) & (off_x_h < IN_H) & (off_x_w >= 0) & (off_x_w < IN_W))[:, None] & (off_x_crs < CRS)[None, :]
    off_w_crs = tl.arange(0, BLOCK_K)
    off_w_k = off_y_k
    w_ptrs = w + off_w_crs[:, None] + off_w_k[None, :] * stride_wn
    mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
    matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
    matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for crs in range(0, CRS, BLOCK_K):
        acc += tl.dot(matrix_x, matrix_w, out_dtype=ACC_TYPE)
        w_ptrs += BLOCK_K
        if not CONV1X1_NHWC:
            delta_x_ptrs += BLOCK_K
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            off_x_crs_unpacked = tl.load(delta_x_ptrs, mask=off_x_crs < CRS, other=0)
            x_ptrs = x + off_x_nhw[:, None] + off_x_crs_unpacked[None, :]
        else:
            off_x_crs = crs + BLOCK_K + tl.arange(0, BLOCK_K)
            x_ptrs += BLOCK_K
        mask_x = ((off_x_n < BATCH) & (off_x_h >= 0) & (off_x_h < IN_H) & (off_x_w >= 0) & (off_x_w < IN_W))[:, None] & (off_x_crs < CRS)[None, :]
        mask_w = (off_x_crs < CRS)[:, None] & (off_w_k < KERNEL_N)[None, :]
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
    if WITH_BIAS:
        acc += tl.load(bias + off_y_k)[None, :]
    acc = acc
    off_y_k = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    off_y_nhw = pid_nhw * BLOCK_M + tl.arange(0, BLOCK_M)
    off_y_n = off_y_nhw // (OUT_H * OUT_W)
    off_y_hw = off_y_nhw % (OUT_H * OUT_W)
    off_y_h = off_y_hw // OUT_W + output_padding_h
    off_y_w = off_y_hw % OUT_W + output_padding_w
    y_ptrs = y + off_y_n[:, None] * stride_yn + off_y_h[:, None] * stride_yh + off_y_w[:, None] * stride_yw + off_y_k[None, :] * stride_yc
    mask_y = (off_y_n < BATCH)[:, None] & (off_y_h < OUT_H + output_padding_h)[:, None] & (off_y_w < OUT_W + output_padding_w)[:, None] & (off_y_k < KERNEL_N)[None, :]
    tl.store(y_ptrs, acc, mask=mask_y)
    return


@eval("""triton.heuristics({
    'BLOCK_M': lambda kwargs: min(4096, triton.next_power_of_2(kwargs['size_inp_0'])),
    'BATCH_STRIDE_INP_IS_1': lambda kwargs: kwargs['batch_stride_inp'] == 1,
    'STRIDE_INP_0_IS_1': lambda kwargs: kwargs['stride_inp_0'] == 1,
    'BATCH_STRIDE_OUT_IS_1': lambda kwargs: kwargs['batch_stride_out'] == 1,
    'STRIDE_OUT_0_IS_1': lambda kwargs: kwargs['stride_out_0'] == 1,
})""")
@eval("""triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, kwargs['BLOCK_M'] // 32)),
})""")
@triton.jit
def copy_2d_kernel(output_ptr, input_ptr, bs, size_inp_0, batch_stride_inp, stride_inp_0, batch_stride_out, stride_out_0, BATCH_STRIDE_INP_IS_1: 'tl.constexpr', STRIDE_INP_0_IS_1: 'tl.constexpr', BATCH_STRIDE_OUT_IS_1: 'tl.constexpr', STRIDE_OUT_0_IS_1: 'tl.constexpr', BLOCK_M: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    grid_m = tl.cdiv(size_inp_0, BLOCK_M)
    pid_m = pid
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    A = input_ptr + (1 if BATCH_STRIDE_INP_IS_1 else batch_stride_inp) * pid_batch + rm * (1 if STRIDE_INP_0_IS_1 else stride_inp_0)
    B = output_ptr + (1 if BATCH_STRIDE_OUT_IS_1 else batch_stride_out) * pid_batch + rm * (1 if STRIDE_OUT_0_IS_1 else stride_out_0)
    mask = rm < size_inp_0
    a = tl.load(A, mask=mask)
    tl.store(B, a, mask=mask)


@eval("""triton.heuristics({
    'BLOCK_M': lambda kwargs: min(64, triton.next_power_of_2(kwargs['size_inp_0'])),
    'BLOCK_N': lambda kwargs: min(64, triton.next_power_of_2(kwargs['size_inp_1'])),
    'BATCH_STRIDE_INP_IS_1': lambda kwargs: kwargs['batch_stride_inp'] == 1,
    'STRIDE_INP_0_IS_1': lambda kwargs: kwargs['stride_inp_0'] == 1,
    'STRIDE_INP_1_IS_1': lambda kwargs: kwargs['stride_inp_1'] == 1,
    'BATCH_STRIDE_OUT_IS_1': lambda kwargs: kwargs['batch_stride_out'] == 1,
    'STRIDE_OUT_0_IS_1': lambda kwargs: kwargs['stride_out_0'] == 1,
    'STRIDE_OUT_1_IS_1': lambda kwargs: kwargs['stride_out_1'] == 1,
})""")
@eval("""triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, kwargs['BLOCK_M'] * kwargs['BLOCK_N'] // 32)),
})""")
@triton.jit
def copy_3d_kernel(output_ptr, input_ptr, bs, size_inp_0, size_inp_1, batch_stride_inp, stride_inp_0, stride_inp_1, batch_stride_out, stride_out_0, stride_out_1, BATCH_STRIDE_INP_IS_1: 'tl.constexpr', STRIDE_INP_0_IS_1: 'tl.constexpr', STRIDE_INP_1_IS_1: 'tl.constexpr', BATCH_STRIDE_OUT_IS_1: 'tl.constexpr', STRIDE_OUT_0_IS_1: 'tl.constexpr', STRIDE_OUT_1_IS_1: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    grid_m = tl.cdiv(size_inp_0, BLOCK_M)
    grid_n = tl.cdiv(size_inp_1, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid - pid_m * grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = input_ptr + (1 if BATCH_STRIDE_INP_IS_1 else batch_stride_inp) * pid_batch + (rm[:, None] * (1 if STRIDE_INP_0_IS_1 else stride_inp_0) + rn[None, :] * (1 if STRIDE_INP_1_IS_1 else stride_inp_1))
    B = output_ptr + (1 if BATCH_STRIDE_OUT_IS_1 else batch_stride_out) * pid_batch + (rm[:, None] * (1 if STRIDE_OUT_0_IS_1 else stride_out_0) + rn[None, :] * (1 if STRIDE_OUT_1_IS_1 else stride_out_1))
    mask = (rm < size_inp_0)[:, None] & (rn < size_inp_1)[None, :]
    a = tl.load(A, mask=mask)
    tl.store(B, a, mask=mask)


@eval("""triton.heuristics({
    'BLOCK_M': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_0'])),
    'BLOCK_N': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_1'])),
    'BLOCK_K': lambda kwargs: min(32, triton.next_power_of_2(kwargs['size_inp_2'])),
    'BATCH_STRIDE_INP_IS_1': lambda kwargs: kwargs['batch_stride_inp'] == 1,
    'STRIDE_INP_0_IS_1': lambda kwargs: kwargs['stride_inp_0'] == 1,
    'STRIDE_INP_1_IS_1': lambda kwargs: kwargs['stride_inp_1'] == 1,
    'STRIDE_INP_2_IS_1': lambda kwargs: kwargs['stride_inp_2'] == 1,
    'BATCH_STRIDE_OUT_IS_1': lambda kwargs: kwargs['batch_stride_out'] == 1,
    'STRIDE_OUT_0_IS_1': lambda kwargs: kwargs['stride_out_0'] == 1,
    'STRIDE_OUT_1_IS_1': lambda kwargs: kwargs['stride_out_1'] == 1,
    'STRIDE_OUT_2_IS_1': lambda kwargs: kwargs['stride_out_2'] == 1,
})""")
@eval("""triton.heuristics({
    'num_warps': lambda kwargs: max(1, min(16, kwargs['BLOCK_M'] * kwargs['BLOCK_N'] * kwargs['BLOCK_K'] // 32)),
})""")
@triton.jit
def copy_4d_kernel(output_ptr, input_ptr, bs, size_inp_0, size_inp_1, size_inp_2, batch_stride_inp, stride_inp_0, stride_inp_1, stride_inp_2, batch_stride_out, stride_out_0, stride_out_1, stride_out_2, BATCH_STRIDE_INP_IS_1: 'tl.constexpr', STRIDE_INP_0_IS_1: 'tl.constexpr', STRIDE_INP_1_IS_1: 'tl.constexpr', STRIDE_INP_2_IS_1: 'tl.constexpr', BATCH_STRIDE_OUT_IS_1: 'tl.constexpr', STRIDE_OUT_0_IS_1: 'tl.constexpr', STRIDE_OUT_1_IS_1: 'tl.constexpr', STRIDE_OUT_2_IS_1: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr'):
    pid = tl.program_id(0)
    pid_batch = tl.program_id(1)
    grid_m = tl.cdiv(size_inp_0, BLOCK_M)
    grid_n = tl.cdiv(size_inp_1, BLOCK_N)
    grid_k = tl.cdiv(size_inp_2, BLOCK_K)
    pid_m = pid // (grid_n * grid_k)
    pid_nk = pid - pid_m * (grid_n * grid_k)
    pid_n = pid_nk // grid_k
    pid_k = pid_nk - pid_n * grid_k
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    A = input_ptr + (1 if BATCH_STRIDE_INP_IS_1 else batch_stride_inp) * pid_batch + (rm[:, None, None] * (1 if STRIDE_INP_0_IS_1 else stride_inp_0) + rn[None, :, None] * (1 if STRIDE_INP_1_IS_1 else stride_inp_1) + rk[None, None, :] * (1 if STRIDE_INP_2_IS_1 else stride_inp_2))
    B = output_ptr + (1 if BATCH_STRIDE_OUT_IS_1 else batch_stride_out) * pid_batch + (rm[:, None, None] * (1 if STRIDE_OUT_0_IS_1 else stride_out_0) + rn[None, :, None] * (1 if STRIDE_OUT_1_IS_1 else stride_out_1) + rk[None, None, :] * (1 if STRIDE_OUT_2_IS_1 else stride_out_2))
    mask = (rm < size_inp_0)[:, None, None] & (rn < size_inp_1)[None, :, None] & (rk < size_inp_2)[None, None, :]
    a = tl.load(A, mask=mask)
    tl.store(B, a, mask=mask)


@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = mean_2 - mean_1
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return mean_1 + delta * w2_over_w, m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w, new_weight


@eval("""triton.heuristics({
    'ROW_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['C'] // kwargs['groups']),
    'BLOCK_SIZE':
    lambda kwargs: max(
        1, min(triton.next_power_of_2(kwargs['HxW']),
               4096 // (triton.next_power_of_2(kwargs['C'] // kwargs['groups']))
               )),
})""")
@eval("""triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
    'C_G': lambda kwargs: kwargs['C'] // kwargs['groups'],
})""")
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel(input_ptr, N, C, HxW, groups, eps, mean_ptr, rstd_ptr, C_G, ROW_SIZE: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)
    offset = pid_batch * C * HxW + group * C_G
    X = input_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    row = tl.arange(0, ROW_SIZE)
    for off in range(0, HxW, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        m2_ = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
        mask = (r < HxW)[:, None] & (row[None, :] < C_G)
        weight_ = mask
        x = tl.load(X + (r * C)[:, None] + row[None, :], mask=mask)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_, weight_)
    _mean = tl.view(_mean, (BLOCK_SIZE * ROW_SIZE,))
    _m2 = tl.view(_m2, (BLOCK_SIZE * ROW_SIZE,))
    _weight = tl.view(_weight, (BLOCK_SIZE * ROW_SIZE,))
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1.0 / tl.sqrt(var + eps)
    offset = pid_batch * groups + group
    tl.store(mean_ptr + offset, mean)
    tl.store(rstd_ptr + offset, rstd)


@eval("""triton.heuristics({
    'ROW_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['C'] // kwargs['groups']),
    'BLOCK_SIZE':
    lambda kwargs: max(
        1, min(triton.next_power_of_2(kwargs['cluster_size']),
               4096 // (triton.next_power_of_2(kwargs['C'] // kwargs['groups']))
               )),
})""")
@eval("""triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['ROW_SIZE'] * kwargs['BLOCK_SIZE'] // 128)),
    'C_G': lambda kwargs: kwargs['C'] // kwargs['groups'],
})""")
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel_stage_1(input_ptr, N, C, HxW, groups, cluster_size, cluster_num, cluster_mean_ptr, cluster_m2_ptr, cluster_weight_ptr, C_G, ROW_SIZE: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    group = tl.program_id(0)
    cluster = tl.program_id(1)
    pid_batch = tl.program_id(2)
    offset = pid_batch * C * HxW + group * C_G
    X = input_ptr + offset
    _mean = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _m2 = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    _weight = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
    row = tl.arange(0, ROW_SIZE)
    start = cluster * cluster_size
    end = start + cluster_size
    end = min(end, HxW)
    for off in range(start, end, BLOCK_SIZE):
        r = off + tl.arange(0, BLOCK_SIZE)
        m2_ = tl.zeros((BLOCK_SIZE, ROW_SIZE), dtype=tl.float32)
        mask = (r < end)[:, None] & (row[None, :] < C_G)
        weight_ = mask
        x = tl.load(X + (r * C)[:, None] + row[None, :], mask=mask)
        _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_, weight_)
    _mean = tl.view(_mean, (BLOCK_SIZE * ROW_SIZE,))
    _m2 = tl.view(_m2, (BLOCK_SIZE * ROW_SIZE,))
    _weight = tl.view(_weight, (BLOCK_SIZE * ROW_SIZE,))
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    offset = pid_batch * groups * cluster_num + group * cluster_num + cluster
    tl.store(cluster_mean_ptr + offset, mean)
    tl.store(cluster_m2_ptr + offset, m2)
    tl.store(cluster_weight_ptr + offset, weight)


@eval("""triton.heuristics({
    'BLOCK_SIZE':
    lambda kwargs: triton.next_power_of_2(kwargs['cluster_num']),
})""")
@eval("""triton.heuristics({
    'num_warps':
    lambda kwargs: max(1, min(16, kwargs['BLOCK_SIZE'] // 128)),
})""")
@triton.jit
def group_norm_4d_channels_last_forward_collect_stats_kernel_stage_2(cluster_mean_ptr, cluster_m2_ptr, cluster_weight_ptr, N, groups, cluster_num, eps, mean_ptr, rstd_ptr, BLOCK_SIZE: 'tl.constexpr'):
    group = tl.program_id(0)
    pid_batch = tl.program_id(1)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < cluster_num
    offset = pid_batch * groups * cluster_num + group * cluster_num + block
    cluster_mean = tl.load(cluster_mean_ptr + offset, mask=mask)
    cluster_m2 = tl.load(cluster_m2_ptr + offset, mask=mask)
    cluster_weight = tl.load(cluster_weight_ptr + offset, mask=mask)
    mean, m2, weight = tl.reduce((cluster_mean, cluster_m2, cluster_weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1.0 / tl.sqrt(var + eps)
    offset = pid_batch * groups + group
    tl.store(mean_ptr + offset, mean)
    tl.store(rstd_ptr + offset, rstd)


@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride: 'tl.constexpr', N: 'tl.constexpr', eps, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N)
        m2_ = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        weight_ = cols < N
        _mean, _m2, _weight = x, m2_, weight_
    else:
        _mean = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        _m2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        _weight = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N)
            m2_ = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            weight_ = cols < N
            if off == 0:
                _mean, _m2, _weight = x, m2_, weight_
            else:
                _mean, _m2, _weight = welford_combine(_mean, _m2, _weight, x, m2_, weight_)
    mean, m2, weight = tl.reduce((_mean, _m2, _weight), 0, welford_combine)
    var = m2 / weight
    rstd = 1 / tl.sqrt(var + eps)
    mean = mean
    rstd = rstd
    if Mean is not None:
        tl.store(Mean + row, mean)
    if Rstd is not None:
        tl.store(Rstd + row, rstd)
    if BLOCK_SIZE >= N:
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        if W is None:
            w = tl.full((BLOCK_SIZE,), 1.0, dtype=x.dtype)
        else:
            w = tl.load(W + cols, mask=mask)
        if B is None:
            b = tl.zeros((BLOCK_SIZE,), dtype=x.dtype)
        else:
            b = tl.load(B + cols, mask=mask)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)
    else:
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            if W is None:
                w = tl.full((BLOCK_SIZE,), 1.0, dtype=x.dtype)
            else:
                w = tl.load(W + cols, mask=mask)
            if B is None:
                b = tl.zeros((BLOCK_SIZE,), dtype=x.dtype)
            else:
                b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            tl.store(Y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, B, Mean, Rstd, Lock, stride: 'tl.constexpr', N: 'tl.constexpr', eps, GROUP_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    x = tl.load(X + cols, mask=mask, other=0)
    dy = tl.load(DY + cols, mask=mask, other=0)
    w = tl.load(W + cols, mask=mask)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    tl.store(DX + cols, dx, mask=mask)
    partial_dw = dy * xhat
    partial_db = dy
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr'):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

