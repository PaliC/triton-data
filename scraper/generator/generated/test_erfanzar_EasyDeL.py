import sys
_module = sys.modules[__name__]
del sys
conf = _module
easydel = _module
cli = _module
serve = _module
engine = _module
websocket_engine = _module
train = _module
clm = _module
dpo = _module
sft = _module
utils = _module
etils = _module
auto_tx = _module
easystate = _module
errors = _module
partition_module = _module
generation = _module
flax_utils = _module
logits_process = _module
inference = _module
generation_pipeline = _module
pipeline = _module
server = _module
client = _module
vinference = _module
api_id_generator = _module
api_models = _module
api_server = _module
api_server_test = _module
metrics = _module
kernels = _module
cpu_ops = _module
jax_mha_flash_attention_2 = _module
jax_ring_attention = _module
flash_attention_2 = _module
gemm = _module
gpu_ops = _module
pallas_gemm = _module
pallas_mha_flash_attention_2 = _module
triton_gemm = _module
triton_gqa_flash_attention_2 = _module
triton_mha_flash_attention_2 = _module
ring_attention = _module
rms_norm = _module
tpu_ops = _module
pallas_ring_attention = _module
modules = _module
_blockwise_attention = _module
_vanilla_attention = _module
arctic = _module
arctic_configuration = _module
modelling_arctic_flax = _module
attention_module = _module
auto_models = _module
chatglm = _module
chatglm_configuration = _module
modelling_chatglm_flax = _module
cohere = _module
cohere_configuration = _module
modelling_cohere_flax = _module
common = _module
dbrx = _module
dbrx_configuration = _module
modelling_dbrx_flax = _module
deepseek_v2 = _module
deepseek_configuration = _module
modeling_deepseek_flax = _module
exaone = _module
exaone_configuration = _module
modeling_exaone_flax = _module
falcon = _module
falcon_configuration = _module
modeling_falcon_flax = _module
flax_modeling_utils = _module
gemma = _module
gemma_configuration = _module
modeling_gemma_flax = _module
gemma2 = _module
gemma2_configuration = _module
modeling_gemma2_flax = _module
gpt2 = _module
gpt2_configuration = _module
modeling_gpt2_flax = _module
gpt_j = _module
gpt_j_configuration = _module
modeling_gpt_j_flax = _module
gpt_neo_x = _module
gpt_neo_x_configuration = _module
modeling_gpt_neo_x_flax = _module
grok_1 = _module
grok_1_configuration = _module
modeling_grok_1_flax = _module
internlm2 = _module
internlm2_configuration = _module
modeling_internlm2_flax = _module
jetmoe = _module
jetmoe_configuration = _module
modelling_jetmoe_flax = _module
llama = _module
llama_configuration = _module
modeling_llama_flax = _module
modeling_vision_llama_flax = _module
vision_llama_configuration = _module
lucid_transformer = _module
lt_configuration = _module
modelling_lt_flax = _module
mamba = _module
mamba_configuration = _module
modelling_mamba_flax = _module
mamba2 = _module
mamba2_configuration = _module
modelling_mamba2_flax = _module
mistral = _module
mistral_configuration = _module
modelling_mistral_flax = _module
vision_mistral_configuration = _module
mixtral = _module
mixtral_configuration = _module
modelling_mixtral_flax = _module
modeling_flax_outputs = _module
modeling_utils = _module
mosaic_mpt = _module
modelling_mpt_flax = _module
mosaic_configuration = _module
olmo = _module
modelling_olmo_flax = _module
olmo_configuration = _module
openelm = _module
modelling_openelm_flax = _module
openelm_configuration = _module
opt = _module
modelling_opt_flax = _module
opt_configuration = _module
palm = _module
modelling_palm_flax = _module
palm_configuration = _module
phi = _module
modelling_phi_flax = _module
phi_configuration = _module
phi3 = _module
modelling_phi3_flax = _module
phi3_configuration = _module
phimoe = _module
modeling_phimoe_flax = _module
phimoe_configuration = _module
qwen1 = _module
modeling_qwen1_flax = _module
qwen1_configuration = _module
qwen2 = _module
modeling_qwen_flax = _module
qwen_configuration = _module
qwen2_moe = _module
configuration_qwen2_moe = _module
modeling_qwen2_moe_flax = _module
roberta = _module
modelling_roberta_flax = _module
roberta_configuration = _module
rwkv = _module
modelling_rwkv_flax = _module
rwkv_configuration = _module
stablelm = _module
modelling_stablelm_flax = _module
stablelm_configuration = _module
t5 = _module
modelling_t5_flax = _module
t5_configuration = _module
whisper = _module
modelling_whisper_flax = _module
whisper_configuration = _module
xerxes = _module
modeling_xerxes_flax = _module
xerxes_configuration = _module
zamba2 = _module
modeling_zamba2_flax = _module
zamba2_configuration = _module
smi = _module
trainers = _module
base_trainer = _module
causal_language_model_trainer = _module
functions = _module
modeling_output = _module
direct_preference_optimization_trainer = _module
dpo_config = _module
dpo_trainer = _module
func_utils = _module
concatenators = _module
creators = _module
log_probs = _module
loss_funcs = _module
modelling_output = _module
odds_ratio_preference_optimization_trainer = _module
fwd_bwd_functions = _module
orpo_trainer = _module
packer = _module
prompt_utils = _module
sequence_classification_trainer = _module
supervised_fine_tuning_trainer = _module
stf_trainer = _module
training_configurations = _module
vision_causal_language_model_trainer = _module
transform = _module
parameters_transformation = _module
analyze_memory = _module
compiling_utils = _module
helpers = _module
lazy_import = _module
quantizers = _module
generate_documentations = _module
caching_test = _module
dataset_test = _module
easy_api_test = _module
falcon_11b_c = _module
generation_pipeline_test = _module
import_test = _module
api_engine_test = _module
client_test = _module
inference_benchmark = _module
partitioning_test = _module
shard_loading_test = _module
test_models = _module
clm_test = _module
dpo_test = _module
lora_test = _module
orpo_test = _module
pruning_clm_test = _module
pruning_sft_test = _module
scf_test = _module
sft_test = _module
vinference_api = _module
vinference_test = _module

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


from enum import Enum


from typing import Optional


from typing import Tuple


from typing import Union


from typing import Literal


from functools import partial


import math


import triton


from triton import language as tl


import functools


import numpy as np


import time


import warnings


from functools import lru_cache


from typing import Any


from typing import Dict


from typing import List


import re


from typing import Callable


from typing import Mapping


from typing import Sequence


from typing import Type


from copy import deepcopy


import torch


import copy


def _get_cuda_autotune_config():
    return [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2)]


def _get_hip_autotune_config():
    return [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=4, num_stages=0), triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2}, num_warps=8, num_stages=0), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=8, num_stages=0), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3}, num_warps=4, num_stages=0), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8}, num_warps=4, num_stages=0)]


def _is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == 'cuda'


def _get_autotune_config():
    try:
        if _is_cuda():
            return _get_cuda_autotune_config()
        else:
            return _get_hip_autotune_config()
    except:
        return _get_cuda_autotune_config()


@triton.autotune(configs=_get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def _triton_gemm(a_ptr, b_ptr, c_ptr, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, M, N, K, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_data = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_data = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a_data, b_data)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = acc
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(configs=_get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def _gemm_activation_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr', activation: 'tl.constexpr'):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_data = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_data = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(a_data, b_data)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    acc = activation(acc)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.heuristics({'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0})
@triton.jit
def _fwd_gqa_attn_kernel_block_ptr(Q, K, V, B, softmax_scale: 'tl.constexpr', stride_qb, stride_qh, stride_qg, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bg, stride_bm, stride_bn, stride_ob, stride_oh, stride_og, stride_om, stride_lb, stride_lh, stride_lg, headdim: 'tl.constexpr', num_kv_heads: 'tl.constexpr', num_groups: 'tl.constexpr', seqlen_q, seqlen_k, O, L, HAVE_BIAS: 'tl.constexpr', BIAS_SINGLE_HEAD: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m, off_bh, off_gp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    off_h = off_bh % num_kv_heads
    off_b = off_bh // num_kv_heads
    if not EVEN_N:
        offs_n = tl.arange(0, BLOCK_N)
    Q_Block_ptr = tl.make_block_ptr(base=Q + (off_b * stride_qb + off_h * stride_qh + off_gp * stride_qg), shape=(seqlen_q, headdim), block_shape=(BLOCK_M, BLOCK_HEADDIM), strides=(stride_qm, 1), offsets=(start_m * BLOCK_M, 0), order=(0, 1))
    O_Block_ptr = tl.make_block_ptr(base=O + (off_b * stride_ob + off_h * stride_oh + off_gp * stride_og), shape=(seqlen_q, headdim), block_shape=(BLOCK_M, BLOCK_HEADDIM), strides=(stride_om, 1), offsets=(start_m * BLOCK_M, 0), order=(0, 1))
    L_Block_ptr = tl.make_block_ptr(base=L + (off_b * stride_lb + off_h * stride_lh + off_gp * stride_lg), shape=(seqlen_q,), strides=(1,), offsets=(start_m * BLOCK_M,), block_shape=(BLOCK_M,), order=(0,))
    kv_stride = off_b * stride_kb + off_h * stride_kh
    K_Block_ptr = tl.make_block_ptr(base=K + kv_stride, shape=(headdim, seqlen_k), block_shape=(BLOCK_HEADDIM, BLOCK_N), strides=(1, stride_kn), offsets=(0, 0), order=(1, 0))
    V_Block_ptr = tl.make_block_ptr(base=V + kv_stride, shape=(seqlen_k, headdim), block_shape=(BLOCK_N, BLOCK_HEADDIM), strides=(stride_vn, 1), offsets=(0, 0), order=(0, 1))
    q = tl.load(Q_Block_ptr, boundary_check=(0, 1))
    softmax_scale = softmax_scale
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h * stride_bh + off_gp * stride_bg
        B_Block_ptr = tl.make_block_ptr(base=B + (off_b * stride_bb + bias_h_pos), shape=(seqlen_q, seqlen_k), block_shape=(BLOCK_M, BLOCK_N), strides=(stride_bm, stride_bn), offsets=(start_m * BLOCK_M, 0), order=(0, 1))
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    for j in range(0, seqlen_k, BLOCK_N):
        j = tl.multiple_of(j, BLOCK_N)
        k = tl.load(K_Block_ptr, boundary_check=(0, 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) * softmax_scale
        if not EVEN_N:
            qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if HAVE_BIAS:
            b = tl.load(B_Block_ptr, boundary_check=(0, 1))
            B_Block_ptr = tl.advance(B_Block_ptr, (0, BLOCK_N))
            qk = qk + b
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        else:
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(max_i - max_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(V_Block_ptr, boundary_check=(0, 1))
        acc_o += tl.dot(p, v)
        max_i = max_ij
        lse_i = max_ij + tl.log(tl.exp(lse_i - max_ij) + l_ij)
        K_Block_ptr = tl.advance(K_Block_ptr, (0, BLOCK_N))
        V_Block_ptr = tl.advance(V_Block_ptr, (BLOCK_N, 0))
    o_scale = tl.exp(max_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(L_Block_ptr, lse_i, boundary_check=(0,))
    tl.store(O_Block_ptr, acc_o, boundary_check=(0, 1))


@triton.heuristics({'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0})
@triton.jit
def _fwd_gqa_attn_kernel_ptr_block(Q, K, V, B, softmax_scale: 'tl.constexpr', stride_qb, stride_qh, stride_qg, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bg, stride_bm, stride_bn, stride_ob, stride_oh, stride_og, stride_om, stride_lb, stride_lh, stride_lg, headdim: 'tl.constexpr', num_kv_heads: 'tl.constexpr', num_groups: 'tl.constexpr', seqlen_q, seqlen_k, O, L, HAVE_BIAS: 'tl.constexpr', BIAS_SINGLE_HEAD: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m, off_bh, off_gp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    off_h = off_bh % num_kv_heads
    off_b = off_bh // num_kv_heads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh + off_gp * stride_qg) + (offs_m[:, None] * stride_qm + offs_d[None, :])
    o_ptrs = O + (off_b * stride_ob + off_h * stride_oh + off_gp * stride_og) + (offs_m[:, None] * stride_om + offs_d[None, :])
    l_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_m + off_gp * stride_lg)
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] * stride_vn + offs_d[None, :])
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    softmax_scale = softmax_scale
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h * stride_bh + off_gp * stride_bg
        b_ptrs = B + (off_b * stride_bb + bias_h_pos) + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    for j in range(0, seqlen_k, BLOCK_N):
        j = tl.multiple_of(j, BLOCK_N)
        current_k = offs_n + j
        k = tl.load(k_ptrs + j * stride_kn, mask=(current_k[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k.T) * softmax_scale
        if not EVEN_N:
            qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if HAVE_BIAS:
            b = tl.load(b_ptrs + j, mask=(offs_m[:, None] < seqlen_q) & (current_k[None, :] < seqlen_k), other=0.0)
            qk = qk + b
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        else:
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(max_i - max_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(v_ptrs + j * stride_vn, mask=(current_k[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        acc_o += tl.dot(p, v)
        max_i = max_ij
        lse_i = max_ij + tl.log(tl.exp(lse_i - max_ij) + l_ij)
    o_scale = tl.exp(max_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(l_ptrs, lse_i, mask=offs_m < seqlen_q)
    tl.store(o_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_do_attn_kernel(O, Do, De, stride_ob: 'int', stride_om: 'int', stride_oh: 'int', stride_dob: 'int', stride_dom: 'int', stride_doh: 'int', stride_deb: 'int', stride_deh: 'int', nheads: 'int', headdim: 'int', seqlen_q: 'int', BLOCK_M: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr'):
    """Triton kernel for the backward pass of the attention mechanism with respect to the output gradient.

	Args:
		O: Output array.
		Do: Output gradient array.
		De: Delta array.
		stride_ob: Stride for the output batch dimension.
		stride_om: Stride for the output sequence dimension.
		stride_oh: Stride for the output head dimension.
		stride_dob: Stride for the output gradient batch dimension.
		stride_dom: Stride for the output gradient sequence dimension.
		stride_doh: Stride for the output gradient head dimension.
		stride_deb: Stride for the delta batch dimension.
		stride_deh: Stride for the delta head dimension.
		nheads: Number of heads.
		headdim: Head dimension.
		seqlen_q: Sequence length of the query.
		BLOCK_M: Block size for the query sequence dimension.
		BLOCK_HEADDIM: Block size for the head dimension.
	"""
    off_q = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads
    offs_m = off_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    o_ptrs = O + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    do_ptrs = Do + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :]
    o = tl.load(o_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    delta = tl.sum(o * do, axis=1)
    tl.store(De + (off_b * stride_deb + off_h * stride_deh + offs_m), delta, mask=offs_m < seqlen_q)


@triton.heuristics({'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0, 'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0, 'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM']})
@triton.jit
def _bwd_attn_kernel(Q, K, V, B, Do, L, D, softmax_scale: 'float', stride_qb: 'int', stride_qm: 'int', stride_qh: 'int', stride_kb: 'int', stride_kn: 'int', stride_kh: 'int', stride_vb: 'int', stride_vn: 'int', stride_vh: 'int', stride_bb: 'int', stride_bh: 'int', stride_bm: 'int', stride_dob: 'int', stride_dom: 'int', stride_doh: 'int', stride_dqb: 'int', stride_dqm: 'int', stride_dqh: 'int', stride_dkb: 'int', stride_dkn: 'int', stride_dkh: 'int', stride_dvb: 'int', stride_dvn: 'int', stride_dvh: 'int', stride_lb: 'int', stride_lh: 'int', seqlen_q: 'int', seqlen_k: 'int', headdim: 'int', nheads: 'int', Dq: 'chex.Array', Dk: 'chex.Array', Dv: 'chex.Array', HAVE_BIAS: 'tl.constexpr', BIAS_SINGLE_HEAD: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_M: 'tl.constexpr', EVEN_N: 'tl.constexpr', EVEN_HEADDIM: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_n, off_bh = tl.program_id(0), tl.program_id(2)
    softmax_scale = softmax_scale
    off_h = off_bh % nheads
    off_b = off_bh // nheads
    offs_qm = tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    l_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_qm)
    d_ptrs = D + (off_b * stride_lb + off_h * stride_lh + offs_qm)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = Do + (off_b * stride_dob + off_h * stride_doh) + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = Dq + (off_b * stride_dqb + off_h * stride_dqh) + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h
        b_ptrs = B + (off_b * stride_bb + bias_h_pos * stride_bh) + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(0, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.dot(q, k.T) * softmax_scale
        if not EVEN_N:
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float('-inf'))
        if HAVE_BIAS:
            bias = tl.load(b_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k), other=0.0)
            qk = qk + bias
        lse_i = tl.load(l_ptrs + start_m, mask=offs_m_curr < seqlen_q, other=0.0)
        p = tl.exp(qk - lse_i[:, None])
        do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(p.T, do)
        dp = tl.dot(do, v.T)
        Di = tl.load(d_ptrs + start_m, mask=offs_m_curr < seqlen_q, other=0.0)
        ds = p * (dp - Di[:, None]) * softmax_scale
        dk += tl.dot(ds.T, q)
        dq = tl.dot(ds, k)
        tl.atomic_add(dq_ptrs, dq, mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim))
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if HAVE_BIAS:
            b_ptrs += BLOCK_M * stride_bm
    dv_ptrs = Dv + (off_b * stride_dvb + off_h * stride_dvh) + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = Dk + (off_b * stride_dkb + off_h * stride_dkh) + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
    tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.heuristics({'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0})
@triton.jit
def _fwd_attn_kernel_block_ptr(Q, K, V, B, softmax_scale: 'tl.constexpr', stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_bn, stride_ob, stride_oh, stride_om, stride_lb, stride_lh, headdim: 'tl.constexpr', nheads: 'tl.constexpr', seqlen_q, seqlen_k, O, L, HAVE_BIAS: 'tl.constexpr', BIAS_SINGLE_HEAD: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m, off_bh = tl.program_id(0), tl.program_id(1)
    off_h = off_bh % nheads
    off_b = off_bh // nheads
    if not EVEN_N:
        offs_n = tl.arange(0, BLOCK_N)
    Q_Block_ptr = tl.make_block_ptr(base=Q + (off_b * stride_qb + off_h * stride_qh), shape=(seqlen_q, headdim), block_shape=(BLOCK_M, BLOCK_HEADDIM), strides=(stride_qm, 1), offsets=(start_m * BLOCK_M, 0), order=(0, 1))
    O_Block_ptr = tl.make_block_ptr(base=O + (off_b * stride_ob + off_h * stride_oh), shape=(seqlen_q, headdim), block_shape=(BLOCK_M, BLOCK_HEADDIM), strides=(stride_om, 1), offsets=(start_m * BLOCK_M, 0), order=(0, 1))
    L_Block_ptr = tl.make_block_ptr(base=L + (off_b * stride_lb + off_h * stride_lh), shape=(seqlen_q,), strides=(1,), offsets=(start_m * BLOCK_M,), block_shape=(BLOCK_M,), order=(0,))
    kv_stride = off_b * stride_kb + off_h * stride_kh
    K_Block_ptr = tl.make_block_ptr(base=K + kv_stride, shape=(headdim, seqlen_k), block_shape=(BLOCK_HEADDIM, BLOCK_N), strides=(1, stride_kn), offsets=(0, 0), order=(1, 0))
    V_Block_ptr = tl.make_block_ptr(base=V + kv_stride, shape=(seqlen_k, headdim), block_shape=(BLOCK_N, BLOCK_HEADDIM), strides=(stride_vn, 1), offsets=(0, 0), order=(0, 1))
    q = tl.load(Q_Block_ptr, boundary_check=(0, 1))
    softmax_scale = softmax_scale
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h
        B_Block_ptr = tl.make_block_ptr(base=B + (off_b * stride_bb + bias_h_pos * stride_bh), shape=(seqlen_q, seqlen_k), block_shape=(BLOCK_M, BLOCK_N), strides=(stride_bm, stride_bn), offsets=(start_m * BLOCK_M, 0), order=(0, 1))
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    for j in range(0, seqlen_k, BLOCK_N):
        j = tl.multiple_of(j, BLOCK_N)
        k = tl.load(K_Block_ptr, boundary_check=(0, 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k) * softmax_scale
        if not EVEN_N:
            qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if HAVE_BIAS:
            b = tl.load(B_Block_ptr, boundary_check=(0, 1))
            B_Block_ptr = tl.advance(B_Block_ptr, (0, BLOCK_N))
            qk = qk + b
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        else:
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(max_i - max_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(V_Block_ptr, boundary_check=(0, 1))
        acc_o += tl.dot(p, v)
        max_i = max_ij
        lse_i = max_ij + tl.log(tl.exp(lse_i - max_ij) + l_ij)
        K_Block_ptr = tl.advance(K_Block_ptr, (0, BLOCK_N))
        V_Block_ptr = tl.advance(V_Block_ptr, (BLOCK_N, 0))
    o_scale = tl.exp(max_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(L_Block_ptr, lse_i, boundary_check=(0,))
    tl.store(O_Block_ptr, acc_o, boundary_check=(0, 1))


@triton.heuristics({'EVEN_N': lambda args: args['seqlen_k'] % args['BLOCK_N'] == 0})
@triton.jit
def _fwd_attn_kernel_ptr_block(Q, K, V, B, softmax_scale: 'tl.constexpr', stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_bn, stride_ob, stride_oh, stride_om, stride_lb, stride_lh, headdim: 'tl.constexpr', nheads: 'tl.constexpr', seqlen_q, seqlen_k, O, L, HAVE_BIAS: 'tl.constexpr', BIAS_SINGLE_HEAD: 'tl.constexpr', BLOCK_HEADDIM: 'tl.constexpr', EVEN_N: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_m, off_bh = tl.program_id(0), tl.program_id(1)
    off_h = off_bh % nheads
    off_b = off_bh // nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = Q + (off_b * stride_qb + off_h * stride_qh) + (offs_m[:, None] * stride_qm + offs_d[None, :])
    o_ptrs = O + (off_b * stride_ob + off_h * stride_oh) + (offs_m[:, None] * stride_om + offs_d[None, :])
    l_ptrs = L + (off_b * stride_lb + off_h * stride_lh + offs_m)
    k_ptrs = K + (off_b * stride_kb + off_h * stride_kh) + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (off_b * stride_vb + off_h * stride_vh) + (offs_n[:, None] * stride_vn + offs_d[None, :])
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    softmax_scale = softmax_scale
    if HAVE_BIAS:
        bias_h_pos: 'tl.constexpr' = 0 if BIAS_SINGLE_HEAD else off_h
        b_ptrs = B + (off_b * stride_bb + bias_h_pos * stride_bh) + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    max_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    for j in range(0, seqlen_k, BLOCK_N):
        j = tl.multiple_of(j, BLOCK_N)
        current_k = offs_n + j
        k = tl.load(k_ptrs + j * stride_kn, mask=(current_k[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k.T) * softmax_scale
        if not EVEN_N:
            qk += tl.where((j + offs_n)[None, :] < seqlen_k, 0, float('-inf'))
        if HAVE_BIAS:
            b = tl.load(b_ptrs + j, mask=(offs_m[:, None] < seqlen_q) & (current_k[None, :] < seqlen_k), other=0.0)
            qk = qk + b
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        else:
            max_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - max_ij[:, None])
        l_ij = tl.sum(p, 1)
        acc_o_scale = tl.exp(max_i - max_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        v = tl.load(v_ptrs + j * stride_vn, mask=(current_k[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        acc_o += tl.dot(p, v)
        max_i = max_ij
        lse_i = max_ij + tl.log(tl.exp(lse_i - max_ij) + l_ij)
    o_scale = tl.exp(max_i - lse_i)
    acc_o = acc_o * o_scale[:, None]
    tl.store(l_ptrs, lse_i, mask=offs_m < seqlen_q)
    tl.store(o_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

