import sys
_module = sys.modules[__name__]
del sys
qa_server = _module
chat_server = _module
qabot = _module
conf = _module
format = _module
format_out = _module
grammer = _module
core = _module
dpda = _module
test0 = _module
test1 = _module
test2 = _module
test3 = _module
test4 = _module
test5 = _module
test6 = _module
impl = _module
lightllm = _module
common = _module
basemodel = _module
cuda_graph = _module
cuda_kernel = _module
fast_llm_wquant = _module
lmdeploy_wquant = _module
ppl_awquant = _module
ppl_wquant = _module
infer_struct = _module
layer_infer = _module
base_layer_infer = _module
cache_tensor_manager = _module
post_layer_infer = _module
pre_layer_infer = _module
template = _module
post_layer_infer_template = _module
pre_layer_infer_template = _module
transformer_layer_infer_cohere_template = _module
transformer_layer_infer_template = _module
transformer_layer_infer_template_awquant = _module
transformer_layer_infer_template_wquant = _module
transformer_layer_infer = _module
layer_weights = _module
base_layer_weight = _module
hf_load_utils = _module
pre_and_post_layer_weight = _module
transformer_layer_weight = _module
splitfuse_infer_struct = _module
triton_kernel = _module
apply_penalty = _module
copy_kv_index_to_req = _module
dequantize_gemm_int4 = _module
dequantize_gemm_int8 = _module
destindex_copy_kv = _module
multimodal_emb = _module
quantize_gemm_int8 = _module
splitfuse_copy_kv_index_to_req = _module
build_utils = _module
deepseek2_mem_manager = _module
infer_utils = _module
int8kv_mem_manager = _module
mem_manager = _module
mem_utils = _module
ppl_int4kv_mem_manager = _module
ppl_int8kv_mem_manager = _module
req_manager = _module
models = _module
baichuan13b = _module
model = _module
baichuan2_13b = _module
baichuan2_7b = _module
baichuan7b = _module
bloom = _module
context_flashattention_nopad = _module
layernorm = _module
token_attention_nopad_att1 = _module
token_attention_nopad_reduceV = _module
token_attention_nopad_softmax = _module
token_flashattention_nopad = _module
chatglm2 = _module
rotary_emb = _module
cohere = _module
triton_kernels = _module
layernorm = _module
rotary_emb = _module
deepseek2 = _module
_custom_ops = _module
fused_moe = _module
context_flashattention_nopad = _module
destindex_copy_kv = _module
flash_decoding = _module
flash_decoding_stage1 = _module
flash_decoding_stage2 = _module
gemma_2b = _module
transformer_layer_infer = _module
gelu_and_mul = _module
internlm = _module
internlm2 = _module
internlm2_reward = _module
internlm2_wquant = _module
internlm_wquant = _module
transformer_layer_infer = _module
internlm_xcomposer = _module
internlm_visual = _module
internvl = _module
img_process = _module
internvl_visual = _module
llama = _module
transformer_layer_infer = _module
ds_load_utils = _module
context_flashattention_nopad = _module
embedding = _module
flash_decoding_stage1 = _module
flash_decoding_stage2 = _module
gqa_decode_flashattention_nopad = _module
gqa_flash_decoding = _module
gqa_flash_decoding_stage1 = _module
gqa_flash_decoding_stage2 = _module
ppl_fp16_flash_decoding = _module
ppl_int4kv_copy_kv = _module
ppl_int4kv_flash_decoding = _module
ppl_int8kv_flash_decoding = _module
ppl_quant_copy_kv = _module
rmsnorm = _module
rotary_emb = _module
silu_and_mul = _module
splitfuse_context_flashattention_nopad = _module
token_attention_nopad_att1 = _module
token_attention_nopad_reduceV = _module
token_attention_nopad_softmax = _module
token_attention_softmax_and_reducev = _module
yarn_rotary_utils = _module
llama_awquant = _module
transformer_layer_infer = _module
llama_quik = _module
quik_awquant = _module
transformer_layer_infer = _module
qlinear = _module
llama_wquant = _module
transformer_layer_infer = _module
llava = _module
llava_visual = _module
minicpm = _module
mistral = _module
transformer_layer_infer = _module
context_flashattention_nopad = _module
init_att_sliding_window_info = _module
token_attention_nopad_att1 = _module
token_attention_nopad_reduceV = _module
token_attention_softmax_and_reducev = _module
mixtral = _module
_custom_ops = _module
phi3 = _module
context_flashattention_nopad = _module
destindex_copy_kv = _module
flash_decoding_stage1 = _module
flash_decoding_stage2 = _module
rotary_emb = _module
qwen = _module
qwen2 = _module
transformer_layer_infer = _module
qwen2_vl = _module
qwen2_visual = _module
vision_process = _module
qwen2_wquant = _module
qwen_vl = _module
qwen_visual = _module
qwen_wquant = _module
stablelm = _module
starcoder = _module
starcoder2 = _module
transformer_layer_infer = _module
starcoder_wquant = _module
yi = _module
server = _module
api_lightllm = _module
api_models = _module
api_server = _module
api_tgi = _module
build_prompt = _module
detokenization = _module
decode = _module
manager = _module
embed_cache = _module
naive_memory_cache = _module
interface = _module
utils = _module
health_monitor = _module
httpserver = _module
io_struct = _module
metrics = _module
multimodal_params = _module
req_id_generator = _module
router = _module
dynamic_prompt = _module
radix_cache = _module
shared_arr = _module
model_infer = _module
infer_batch = _module
mode_backend = _module
base_backend = _module
beamsearch = _module
post_process = _module
pre_process = _module
continues_batch = _module
impl_for_return_all_prompt_logprobs = _module
impl_for_reward_model = _module
impl_for_simple_constraint_mode = _module
impl_for_token_healing = _module
outlines_patch = _module
diverse_backend = _module
splitfuse = _module
model_rpc = _module
pause_strategy = _module
req_queue = _module
base_queue = _module
beam_impl = _module
stats = _module
token_load = _module
sampling_params = _module
tokenizer = _module
visualserver = _module
config_utils = _module
graceful_utils = _module
health_check = _module
log_utils = _module
net_utils = _module
petrel_helper = _module
profile_max_tokens = _module
start_utils = _module
setup = _module
benchmark_mcq = _module
benchmark_serving = _module
benchmark_serving_vllm = _module
gomoku_game = _module
test_demo = _module
test_bloom = _module
test_chatglm2 = _module
test_intern = _module
test_llama = _module
test_llama2 = _module
model_infer_batchs = _module
process_utils = _module
test_settings = _module
test_starcoder = _module
test_starcoder_quantized = _module
benchmark_prompt_cache = _module
test_constraint_server = _module
test_mlllm = _module
test_server = _module
quick_launch_docker = _module

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


import numpy as np


import time


import math


import torch.nn.functional as F


from typing import List


from typing import Optional


from typing import Tuple


from typing import Type


import functools


from typing import Any


from typing import Dict


import torch.functional as F


import torch.distributed as dist


from functools import partial


import collections


import uuid


from typing import AsyncGenerator


@triton.jit
def _fwd_kernel_apply_penalty(Logits, presence_penalty, freqency_penalty, repetition_penalty, p_token_ids, p_token_counts, p_cumsum_seq_len, stride_logit_b, stride_logit_s, BLOCK_P: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)
    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)
    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)
    return


@triton.jit
def _fwd_kernel_copy_kv_index_to_req(req_to_token_indexs, b_req_idx, b_split_seq_len, cumsum_split_seq_len, b_seq_len, memindex, stride_req_to_token_b, stride_req_to_token_s, BLOCK_M: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    q_split_len = tl.load(b_split_seq_len + cur_index)
    q_mem_end = tl.load(cumsum_split_seq_len + cur_index)
    q_mem_start = q_mem_end - q_split_len
    store_end = tl.load(b_seq_len + cur_index)
    store_start = store_end - q_split_len
    off_m = tl.arange(0, BLOCK_M)
    for block_start in range(0, q_split_len, BLOCK_M):
        read_index = tl.load(memindex + q_mem_start + block_start + off_m, mask=q_mem_start + block_start + off_m < q_mem_end, other=0)
        tl.store(req_to_token_indexs + cur_req_idx * stride_req_to_token_b + (block_start + store_start + off_m), read_index, mask=block_start + store_start + off_m < store_end)
    return


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8), triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8)], key=['M', 'N', 'K', 'NO_GROUPS'])
@triton.jit
def matmul4_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales_g, stride_scales_n, stride_zeros_g, stride_zeros_n, groupsize, NO_GROUPS: 'tl.constexpr', BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr'):
    """
    Compute the matrix multiplication C = A x B.
    A is of shape (M, K) float16
    B is of shape (K//8, N) int32
    C is of shape (M, N) float16
    scales is of shape (G, N) float16
    zeros is of shape (G, N//8) int32
    groupsize is an int specifying the size of groups for scales and zeros.
    G is K // groupsize.
    Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.
    WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
    WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
    WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
    """
    bits = 4
    infearure_per_bits = 8
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
    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + offs_bn // infearure_per_bits * stride_zeros_n
    shifter = offs_k % infearure_per_bits * bits
    zeros_shifter = offs_bn % infearure_per_bits * bits
    if NO_GROUPS:
        scales = tl.load(scales_ptrs)
        zeros = tl.load(zeros_ptrs)
        zeros = zeros >> zeros_shifter & 15
        zeros = zeros * scales
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs)
        if not NO_GROUPS:
            g_id = k // (groupsize // BLOCK_SIZE_K)
            ptr = scales_ptrs + g_id * stride_scales_g
            scales = tl.load(ptr)
            ptr = zeros_ptrs + g_id * stride_zeros_g
            zeros = tl.load(ptr)
            zeros = zeros >> zeros_shifter & 15
            zeros = zeros * scales
        b = b >> shifter[:, None] & 15
        b = b * scales[None, :] - zeros[None, :]
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K // infearure_per_bits * stride_bk
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.autotune(configs=[triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 1, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=4), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8), triton.Config({'SPLIT_K': 2, 'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 16}, num_stages=2, num_warps=4)], key=['M', 'N', 'K'], reset_to_zero=['c_ptr'])
@triton.jit
def matmul_kernel(a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_asm, stride_bk, stride_bn, stride_bsn, stride_cm, stride_cn, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr', SPLIT_K: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
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
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    as_ptrs = as_ptr + offs_am * stride_asm
    bs_ptrs = bs_ptr + offs_bn * stride_bsn
    a_scale = tl.load(as_ptrs, mask=offs_am < M, other=0.0)
    b_scale = tl.load(bs_ptrs, mask=offs_bn < N, other=0.0)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    c = accumulator.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4), triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8), triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2), triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2)], key=['K', 'N'])
@triton.jit
def dequantize_kernel(b_ptr, b_scale_ptr, fpb_ptr, K, N, stride_bk, stride_bn, stride_fpbk, stride_fpbn, BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0)
    n_block_idx = tl.program_id(axis=1)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = n_block_idx * BLOCK_SIZE_N + offs_n[None, :]
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)


@triton.jit
def _fwd_kernel_destindex_copy_kv(K, Dest_loc, Out, stride_k_bs, stride_k_h, stride_k_d, stride_o_bs, stride_o_h, stride_o_d, head_num, head_dim, BLOCK_DMODEL: 'tl.constexpr', BLOCK_HEAD: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    dest_index = tl.load(Dest_loc + cur_index)
    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    k = tl.load(k_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(o_ptrs, k, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    return


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(K, Dest_loc, Out, Out_scale, stride_k_bs, stride_k_h, stride_k_d, stride_o_bs, stride_o_h, stride_o_d, stride_os_bs, stride_os_h, stride_os_d, head_num, head_dim, BLOCK_DMODEL: 'tl.constexpr', BLOCK_HEAD: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    dest_index = tl.load(Dest_loc + cur_index)
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :], mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    abs_data = tl.abs(src_data)
    data_scale = (tl.max(abs_data, axis=1) / 127.0)[:, None]
    q_src_data = src_data / data_scale
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs, q_src_data, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, Req_to_tokens, B_req_idx, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, stride_req_to_tokens_b, stride_req_to_tokens_s, kv_group_num, b_prompt_cache_len, head_dim: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n), mask=start_n + offs_n < block_end_loc, other=0)
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :] < block_end_loc) & (offs_d[:, None] < head_dim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :], qk, float('-100000000.0'))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
        acc = acc * acc_scale[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None] < block_end_loc) & (offs_d[None, :] < head_dim), other=0.0)
        p = p
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < head_dim))
    return


@triton.autotune(configs=[triton.Config({}, num_stages=2, num_warps=8), triton.Config({}, num_stages=2, num_warps=4), triton.Config({}, num_stages=2, num_warps=2), triton.Config({}, num_stages=2, num_warps=1)], key=['K'])
@triton.jit
def quantize_int8_perrow_kernel(fpa_ptr, a_ptr, as_ptr, M, K, stride_fpam, stride_fpak, stride_am, stride_ak, stride_asm, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr'):
    pid_m = tl.program_id(axis=0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :] * stride_fpak
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        a_max = tl.maximum(a_max, tl.max(tl.abs(fpa), axis=1))
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
    a_scale = a_max / 127.0
    fpa_ptrs = fpa_ptr + offs_am[:, None] * stride_fpam + offs_k[None, :] * stride_fpak
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        fpa = tl.load(fpa_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        inta = fpa / a_scale[:, None]
        tl.store(a_ptrs, inta, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K)
        fpa_ptrs += BLOCK_SIZE_K * stride_fpak
        a_ptrs += BLOCK_SIZE_K * stride_ak
    as_offs = pid_m * BLOCK_SIZE_M * stride_asm + tl.arange(0, BLOCK_SIZE_M)
    tl.store(as_ptr + as_offs, a_scale)


@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, stride, N, eps, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _fwd_kernel_token_att1(Q, K, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, Att_Out, stride_req_to_tokens_b, stride_req_to_tokens_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, att_stride_h, att_stride_bs, kv_group_num, sliding_window, BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Att_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_att_seq_len = tl.load(B_Att_Seqlen + cur_batch)
    cur_batch_start_index = tl.maximum(cur_batch_seq_len - sliding_window, 0)
    cur_batch_end_index = cur_batch_seq_len
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_att_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value = att_value
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@triton.jit
def _fwd_kernel_token_att2(Prob, V, Out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, stride_req_to_tokens_b, stride_req_to_tokens_s, stride_ph, stride_pbs, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, kv_group_num, sliding_window, BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = tl.maximum(cur_batch_seq_len - sliding_window, 0)
    cur_batch_in_all_start_index = tl.load(B_Att_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_att_seq_len = tl.load(B_Att_Seqlen + cur_batch)
    v_loc_off = cur_batch_req_idx * stride_req_to_tokens_b + (cur_batch_start_index + offs_n) * stride_req_to_tokens_s
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_att_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n, mask=start_n + offs_n < cur_att_seq_len, other=0.0)
        v_loc = tl.load(Req_to_tokens + v_loc_off + start_n * stride_req_to_tokens_s, mask=start_n + offs_n + cur_batch_start_index < cur_batch_seq_len, other=0.0)
        v_value = tl.load(V + v_offs + v_loc[:, None] * stride_vbs, mask=start_n + offs_n[:, None] + cur_batch_start_index < cur_batch_seq_len, other=0.0)
        acc += tl.sum(p_value[:, None] * v_value, 0)
    acc = acc
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@triton.jit
def _fwd_kernel_token_softmax(Logics, B_Start_Loc, B_Seqlen, Prob_Out, stride_logic_h, stride_logic_bs, stride_prob_h, stride_prob_bs, BLOCK_SIZE: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    row = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_in_all_start_index + col_offsets) * stride_logic_bs, mask=col_offsets < cur_batch_seq_len, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(Prob_Out + cur_head * stride_prob_h + (cur_batch_in_all_start_index + col_offsets) * stride_prob_bs, softmax_output, mask=col_offsets < cur_batch_seq_len)
    return


@triton.jit
def _rotary_kernel(Q, K, Cos, Sin, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_cosbs, stride_cosd, stride_sinbs, stride_sind, max_total_len, HEAD_Q, HEAD_K, rot_dim, head_dim, BLOCK_HEAD: 'tl.constexpr', BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr'):
    cur_head_index = tl.program_id(0)
    cur_seq_index = tl.program_id(1)
    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_range0 = tl.arange(0, BLOCK_DMODEL)
    dim_range1 = rot_dim + tl.arange(0, BLOCK_DMODEL)
    off_q0 = cur_seq_range[:, None, None] * stride_qbs + cur_head_range[None, :, None] * stride_qh + dim_range0[None, None, :] * stride_qd
    off_q1 = cur_seq_range[:, None, None] * stride_qbs + cur_head_range[None, :, None] * stride_qh + dim_range1[None, None, :] * stride_qd
    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    q0 = tl.load(Q + off_q0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range0[None, None, :] < rot_dim), other=0.0)
    q1 = tl.load(Q + off_q1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < head_dim), other=0.0)
    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos
    tl.store(Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range0[None, None, :] < rot_dim))
    tl.store(Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < head_dim))
    off_k0 = cur_seq_range[:, None, None] * stride_kbs + cur_head_range[None, :, None] * stride_kh + dim_range0[None, None, :] * stride_kd
    off_k1 = cur_seq_range[:, None, None] * stride_kbs + cur_head_range[None, :, None] * stride_kh + dim_range1[None, None, :] * stride_kd
    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    k0 = tl.load(K + off_k0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range0[None, None, :] < rot_dim), other=0.0)
    k1 = tl.load(K + off_k1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < head_dim), other=0.0)
    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos
    tl.store(K + off_k0, out_k0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range0[None, None, :] < rot_dim))
    tl.store(K + off_k1, out_k1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < head_dim))
    return


@triton.jit
def _layer_norm_fwd_kernel(X, W, Y, stride_x_N, stride_x_hn, stride_x_hd, stride_y_N, stride_y_hn, stride_y_hd, stride_w_hn, stride_w_hd, N, eps, BLOCK_SIZE: 'tl.constexpr'):
    Seq = tl.program_id(0)
    H = tl.program_id(1)
    X += Seq * stride_x_N + H * stride_x_hn
    Y += Seq * stride_y_N + H * stride_y_hn
    W += H * stride_w_hn
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0)
        x_hat = (x - mean) * rstd
        y = x_hat * w
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def moe_align_block_size_stage1(topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel: 'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(0)
    start_idx = pid * tokens_per_thread
    off_c = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel: 'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(0)
    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel: 'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(topk_ids_ptr, sorted_token_ids_ptr, expert_ids_ptr, total_tokens_post_pad_ptr, tokens_cnts_ptr, cumsum_ptr, num_experts: 'tl.constexpr', block_size: 'tl.constexpr', numel: 'tl.constexpr', tokens_per_thread: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)
    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)
    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts
    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


@triton.jit
def fused_moe_kernel(a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr, N, K, EM, num_valid_tokens, stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: 'tl.constexpr', BLOCK_SIZE_N: 'tl.constexpr', BLOCK_SIZE_K: 'tl.constexpr', GROUP_SIZE_M: 'tl.constexpr', MUL_ROUTED_WEIGHT: 'tl.constexpr', top_k: 'tl.constexpr', compute_type: 'tl.constexpr', use_fp8: 'tl.constexpr'):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if use_fp8:
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr + off_experts)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if use_fp8:
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_fp8:
        accumulator = accumulator * a_scale * b_scale
    else:
        accumulator = accumulator
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _fwd_kernel_no_prompt_cache(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, kv_group_num, head_dim, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    q = tl.load(Q + off_q, mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=(start_n + offs_n[None, :] < cur_batch_seq_len) & (offs_d[:, None] < head_dim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, float('-inf'))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=(start_n + offs_n[:, None] < cur_batch_seq_len) & (offs_d[None, :] < head_dim), other=0.0)
        p = p
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < head_dim))
    return


@triton.jit
def _fwd_kernel_flash_decode_stage1(Q, K, V, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen, Mid_O, Mid_O_LogExpSum, stride_req_to_tokens_b, stride_req_to_tokens_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, gqa_group_size, head_dim, BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    block_n_size = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1) // BLOCK_N
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    q = tl.load(Q + off_q, mask=offs_d < head_dim, other=0.0)
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(K + off_k, mask=(offs_n_new[:, None] < cur_batch_end_index) & (offs_d[None, :] < head_dim), other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float('-inf'))
        v = tl.load(V + off_k, mask=(offs_n_new[:, None] < cur_batch_end_index) & (offs_d[None, :] < head_dim), other=0.0)
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)
        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic
    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_d
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        tl.store(Mid_O + off_mid_o, acc / sum_exp, mask=offs_d < head_dim)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, head_dim, BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os, mask=offs_d < head_dim, other=0.0)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp, mask=offs_d < head_dim)
    return


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


_kAlpha = math.sqrt(2.0 / math.pi)


@triton.jit
def gelu(x):
    """
    GeLU_ activation - Gaussian error linear unit

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1 + tanh(_kAlpha * (x + 0.044715 * x * x * x)))


@triton.jit
def _gelu_and_mul_kernel(input_ptr, stride_input_m, stride_input_n, stride_output_m, stride_output_n, size_m, size_n, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    tid = tl.program_id(0)
    input_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    output_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    pid = tl.program_id(1)
    input_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    output_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    up_offsets = input_m_offsets[:, None] * stride_input_m + (input_n_offsets[None, :] + size_n) * stride_input_n
    gate_offsets = input_m_offsets[:, None] * stride_input_m + input_n_offsets[None, :] * stride_input_n
    res_offsets = output_m_offsets[:, None] * stride_output_m + output_n_offsets[None, :] * stride_output_n
    up = tl.load(input_ptr + up_offsets, mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None], other=0.0)
    gate = tl.load(input_ptr + gate_offsets, mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None], other=0.0)
    gate = gelu(gate)
    gate = gate
    tl.store(input_ptr + res_offsets, up * gate, mask=(output_n_offsets < size_n)[None, :] * (output_m_offsets < size_m)[:, None])


@triton.jit
def embedding_kernel(weight, input_ids, out, vob_start_id, vob_end_id, stride_weight_seq, stride_out_seq, n_ctx, hiden_size: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_NN: 'tl.constexpr'):
    start_n = tl.program_id(0) * BLOCK_N
    offs_nn = start_n + tl.arange(0, BLOCK_NN)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    for start_nn in range(0, BLOCK_N, BLOCK_NN):
        start_nn = tl.multiple_of(start_nn, BLOCK_NN)
        offs_seq = start_nn + offs_nn
        n_ctx_mask = offs_seq < n_ctx
        token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=vob_end_id)
        id_mask = (token_ids >= vob_start_id) & (token_ids < vob_end_id)
        token_ids = token_ids - vob_start_id
        dim_mask = offs_d < hiden_size
        load_mask = id_mask[:, None] & dim_mask[None, :]
        store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
        vecs = tl.load(weight + token_ids[:, None] * stride_weight_seq + offs_d[None, :], mask=load_mask, other=0.0)
        tl.store(out + offs_seq[:, None] * stride_out_seq + offs_d[None, :], vecs, mask=store_mask)


@triton.jit
def _fwd_kernel_destindex_copy_quantize_int4_kv(K, Dest_loc, Out, Out_scale, stride_k_bs, stride_k_h, stride_k_g, stride_k_d, stride_o_bs, stride_o_h, stride_o_g, stride_o_d, stride_os_bs, stride_os_h, stride_os_g, group_size, BLOCK_GROUP_NUM: 'tl.constexpr', BLOCK_GROUP_DIM: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM // 2)
    dest_index = tl.load(Dest_loc + cur_index)
    src_data_0 = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2, mask=offs_g[:, None] < group_size, other=0.0)
    src_data_1 = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * 2 + 1, mask=offs_g[:, None] < group_size, other=0.0)
    abs_data_0 = tl.abs(src_data_0)
    abs_data_1 = tl.abs(src_data_1)
    data_scale = tl.maximum(tl.max(abs_data_0, axis=1), tl.max(abs_data_1, axis=1)) / 7.0
    q_src_data_0 = src_data_0 / data_scale[:, None]
    q_src_data_0 = tl.where(q_src_data_0 > 7, 7, q_src_data_0)
    q_src_data_0 = tl.where(q_src_data_0 < -7, -7, q_src_data_0)
    q_src_data_1 = src_data_1 / data_scale[:, None]
    q_src_data_1 = tl.where(q_src_data_1 > 7, 7, q_src_data_1)
    q_src_data_1 = tl.where(q_src_data_1 < -7, -7, q_src_data_1)
    low_4 = (q_src_data_0 & 128) >> 4 | q_src_data_0 & 15
    high_4 = ((q_src_data_1 & 128) >> 4 | q_src_data_1 & 15) << 4
    out_data = low_4 | high_4
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g
    tl.store(o_ptrs, out_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)
    return


@triton.jit
def _rms_norm_fwd_fused(X, Y, W, stride, N, eps, BLOCK_SIZE: 'tl.constexpr'):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0)
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


@triton.jit
def _silu_and_mul_kernel(input_ptr, stride_input_m, stride_input_n, stride_output_m, stride_output_n, size_m, size_n, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    stride_input_m = stride_input_m
    stride_output_m = stride_output_m
    tid = tl.program_id(0)
    input_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    output_m_offsets = tid * BLOCK_M + tl.arange(0, BLOCK_M)
    pid = tl.program_id(1)
    input_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    output_n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    up_offsets = input_m_offsets[:, None] * stride_input_m + (input_n_offsets[None, :] + size_n) * stride_input_n
    gate_offsets = input_m_offsets[:, None] * stride_input_m + input_n_offsets[None, :] * stride_input_n
    res_offsets = output_m_offsets[:, None] * stride_output_m + output_n_offsets[None, :] * stride_output_n
    up = tl.load(input_ptr + up_offsets, mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None], other=0.0)
    gate = tl.load(input_ptr + gate_offsets, mask=(input_n_offsets < size_n)[None, :] * (input_m_offsets < size_m)[:, None], other=0.0)
    gate = gate / (1 + tl.exp(-gate))
    gate = gate
    tl.store(input_ptr + res_offsets, up * gate, mask=(output_n_offsets < size_n)[None, :] * (output_m_offsets < size_m)[:, None])


@triton.jit
def _fwd_kernel_int8(Q, K, K_scale, V, V_scale, sm_scale, Req_to_tokens, B_req_idx, B_split_start_loc, B_split_ready_cache_len, B_seqlen, Out, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_ksbs, stride_ksh, stride_ksd, stride_vbs, stride_vh, stride_vd, stride_vsbs, stride_vsh, stride_vsd, stride_obs, stride_oh, stride_od, stride_req_to_tokens_b, stride_req_to_tokens_s, kv_group_num, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_q_split_start_loc = tl.load(B_split_start_loc + cur_batch)
    cur_batch_seq_len = tl.load(B_seqlen + cur_batch)
    cur_batch_seq_start = tl.load(B_split_ready_cache_len + cur_batch)
    cur_batch_q_split_seq_len = cur_batch_seq_len - cur_batch_seq_start
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (cur_batch_q_split_start_loc + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    off_k = cur_kv_head * stride_kh + offs_d[:, None]
    off_v = cur_kv_head * stride_vh + offs_d[None, :]
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_q_split_seq_len, other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    ks_ptrs = K_scale + cur_kv_head * stride_ksh
    vs_ptrs = V_scale + cur_kv_head * stride_vsh
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(start_m * BLOCK_M < cur_batch_q_split_seq_len, 1, 0)
    for start_n in range(0, block_mask * (cur_batch_seq_start + (start_m + 1) * BLOCK_M), BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(Req_to_tokens + cur_batch_req_idx * stride_req_to_tokens_b + start_n + offs_n, mask=start_n + offs_n < cur_batch_seq_len, other=0)
        k = tl.load(k_ptrs + kv_loc[None, :] * stride_kbs, mask=start_n + offs_n[None, :] < cur_batch_seq_len, other=0.0)
        k_scale = tl.load(ks_ptrs + kv_loc[None, :] * stride_ksbs, mask=start_n + offs_n[None, :] < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k_scale * k)
        qk *= sm_scale
        qk = tl.where(cur_batch_seq_start + offs_m[:, None] >= start_n + offs_n[None, :], qk, float('-100000000.0'))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + kv_loc[:, None] * stride_vbs, mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        v_scale = tl.load(vs_ptrs + kv_loc[:, None] * stride_vsbs, mask=(start_n + offs_n)[:, None] < cur_batch_seq_len, other=0.0)
        p = p
        acc += tl.dot(p, v * v_scale)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_q_split_start_loc + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_q_split_seq_len)
    return


@triton.jit
def _fwd_kernel_token_att1_int8(Q, K, K_scale, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, Att_Out, stride_req_to_tokens_b, stride_req_to_tokens_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_ksbs, stride_ksh, stride_ksd, att_stride_h, att_stride_bs, kv_group_num, BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        off_ks = k_loc[:, None] * stride_ksbs + cur_kv_head * stride_ksh
        k_scale = tl.load(K_scale + off_ks, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k * k_scale, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return


@triton.jit
def _fwd_kernel_token_att2_int8v(Prob, V, V_scale, Out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, stride_req_to_tokens_b, stride_req_to_tokens_s, stride_ph, stride_pbs, stride_vbs, stride_vh, stride_vd, stride_vsbs, stride_vsh, stride_vsd, stride_obs, stride_oh, stride_od, kv_group_num, BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = 0
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    v_loc_off = cur_batch_req_idx * stride_req_to_tokens_b + (cur_batch_start_index + offs_n) * stride_req_to_tokens_s
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
    v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    vs_offs = cur_kv_head * stride_vsh
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n, mask=start_n + offs_n < cur_batch_seq_len, other=0.0)
        v_loc = tl.load(Req_to_tokens + v_loc_off + start_n * stride_req_to_tokens_s, mask=start_n + offs_n < cur_batch_seq_len, other=0.0)
        v_value = tl.load(V + v_offs + v_loc[:, None] * stride_vbs, mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        vs_value = tl.load(V_scale + vs_offs + v_loc[:, None] * stride_vsbs, mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        acc += tl.sum(p_value[:, None] * v_value * vs_value, 0)
    acc = acc
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


@triton.jit
def _fwd_kernel_init_att_window_info(b_seq_len, b_att_seq_len, batch_size, sliding_window, BLOCK_SIZE: 'tl.constexpr'):
    cur_index = tl.program_id(0)
    cur_start = cur_index * BLOCK_SIZE
    offsets = cur_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size
    cur_seq_len = tl.load(b_seq_len + offsets, mask=mask)
    b_att_seq_len_data = tl.minimum(cur_seq_len, sliding_window)
    tl.store(b_att_seq_len + offsets, b_att_seq_len_data, mask=mask)
    return

