import sys
_module = sys.modules[__name__]
del sys
bench_dspy_intro = _module
agent_functions = _module
bench_other = _module
bench_sglang = _module
build_dataset = _module
gen_data = _module
download_images = _module
launch_server = _module
lora_bench = _module
data_gen = _module
lmql_funcs = _module
conf = _module
deploy = _module
anthropic_example_chat = _module
anthropic_example_complete = _module
azure_openai_example_chat = _module
gemini_example_chat = _module
gemini_example_complete = _module
gemini_example_multimodal_chat = _module
local_example_chat = _module
local_example_complete = _module
local_example_llava_next = _module
openai_example_chat = _module
openai_example_complete = _module
openrouter_example_chat = _module
together_example_chat = _module
together_example_complete = _module
chinese_regex = _module
choices_logprob = _module
cot_decoding = _module
json_decode = _module
json_logprobs = _module
srt_example_llava_v = _module
openai_chat_speculative = _module
openai_speculative = _module
parallel_sample = _module
readme_examples = _module
sgl_gen_min_tokens = _module
streaming = _module
model = _module
async_io_api = _module
custom_server = _module
input_ids = _module
offline_batch_inference = _module
http_llama3_llava_test = _module
http_llava_onevision_test = _module
http_qwen_llava_test = _module
lora = _module
openai_batch_chat = _module
openai_batch_complete = _module
openai_chat_with_response_prefill = _module
reward_model = _module
sglang = _module
api = _module
bench_latency = _module
bench_server_latency = _module
bench_serving = _module
check_env = _module
global_config = _module
lang = _module
backend = _module
anthropic = _module
base_backend = _module
litellm = _module
openai = _module
runtime_endpoint = _module
vertexai = _module
chat_template = _module
choices = _module
compiler = _module
interpreter = _module
ir = _module
tracer = _module
launch_server_llavavid = _module
configs = _module
exaone = _module
model_config = _module
qwen2vl = _module
constrained = _module
base_tool_cache = _module
bnf_cache = _module
fsm_cache = _module
grammar = _module
jump_forward = _module
conversation = _module
hf_transformers_utils = _module
activation = _module
attention = _module
double_sparsity_backend = _module
flashinfer_backend = _module
triton_backend = _module
decode_attention = _module
double_sparsity_attention = _module
extend_attention = _module
prefill_attention = _module
fused_moe = _module
fused_moe = _module
layer = _module
patch = _module
layernorm = _module
linear = _module
logits_processor = _module
pooler = _module
quantization = _module
base_config = _module
radix_attention = _module
rotary_embedding = _module
sampler = _module
torchao_utils = _module
lora_config = _module
lora_manager = _module
data_parallel_controller = _module
detokenizer_manager = _module
image_processor = _module
io_struct = _module
schedule_batch = _module
schedule_policy = _module
scheduler = _module
tokenizer_manager = _module
tp_worker = _module
tp_worker_overlap_thread = _module
base_prefix_cache = _module
chunk_cache = _module
flush_cache = _module
memory_pool = _module
radix_cache = _module
mm_utils = _module
cuda_graph_runner = _module
forward_batch_info = _module
model_runner = _module
baichuan = _module
chatglm = _module
commandr = _module
dbrx = _module
deepseek = _module
deepseek_v2 = _module
gemma = _module
gemma2 = _module
gpt_bigcode = _module
grok = _module
internlm2 = _module
llama = _module
llama_classification = _module
llama_embedding = _module
llama_reward = _module
llava = _module
llavavid = _module
minicpm = _module
minicpm3 = _module
mistral = _module
mixtral = _module
mixtral_quant = _module
mllama = _module
olmo = _module
olmoe = _module
qwen = _module
qwen2 = _module
qwen2_moe = _module
qwen2_vl = _module
stablelm = _module
torch_native_llama = _module
xverse = _module
xverse_moe = _module
yivl = _module
adapter = _module
protocol = _module
penaltylib = _module
orchestrator = _module
frequency_penalty = _module
min_new_tokens = _module
presence_penalty = _module
repetition_penalty = _module
sampling_batch_info = _module
sampling_params = _module
server = _module
server_args = _module
utils = _module
few_shot_gsm8k = _module
few_shot_gsm8k_engine = _module
run_eval = _module
runners = _module
simple_eval_common = _module
simple_eval_gpqa = _module
simple_eval_humaneval = _module
simple_eval_math = _module
simple_eval_mgsm = _module
simple_eval_mmlu = _module
test_activation = _module
test_layernorm = _module
test_programs = _module
test_utils = _module
version = _module
convert_yi_vl = _module
test_flashinfer = _module
test_httpserver_classify = _module
test_httpserver_concurrent = _module
test_httpserver_decode = _module
test_httpserver_decode_stream = _module
test_httpserver_llava = _module
test_httpserver_reuse = _module
test_jump_forward = _module
test_robust = _module
fix_corrupted_json = _module
load_tokenizer = _module
analyzer = _module
lora_hf_play = _module
lora_vllm_play = _module
reference_hf = _module
run_suite = _module
test_anthropic_backend = _module
test_bind_cache = _module
test_choices = _module
test_litellm_backend = _module
test_openai_backend = _module
test_srt_backend = _module
test_tracing = _module
test_vertexai_backend = _module
compare = _module
test_embedding_models = _module
test_generation_models = _module
test_lora = _module
test_reward_models = _module
test_frequency_penalty = _module
test_min_new_tokens = _module
test_presence_penalty = _module
test_repetition_penalty = _module
test_srt_endpoint_with_penalizers = _module
test_bench_latency = _module
test_bench_serving = _module
test_cache_report = _module
test_chunked_prefill = _module
test_create_kvindices = _module
test_data_parallelism = _module
test_double_sparsity = _module
test_embedding_openai_server = _module
test_eval_accuracy_large = _module
test_eval_accuracy_large_chunked_prefill = _module
test_eval_accuracy_large_mixed_chunked_prefill = _module
test_eval_accuracy_mini = _module
test_json_constrained = _module
test_large_max_new_tokens = _module
test_matched_stop = _module
test_mla = _module
test_mla_fp8 = _module
test_models_from_modelscope = _module
test_moe_eval_accuracy_large = _module
test_nightly_gsm8k_eval = _module
test_openai_server = _module
test_overlap_schedule = _module
test_pytorch_sampling_backend = _module
test_retract_decode = _module
test_server_args = _module
test_skip_tokenizer_init = _module
test_srt_endpoint = _module
test_srt_engine = _module
test_torch_compile = _module
test_torchao = _module
test_triton_attention_backend = _module
test_triton_attention_kernels = _module
test_update_weights = _module
test_vision_openai_server = _module

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


import random


import time


import warnings


from typing import Any


from typing import AsyncGenerator


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


from collections import OrderedDict


from collections import defaultdict


import torch


from typing import TYPE_CHECKING


import torch.nn as nn


from enum import Enum


from enum import auto


import triton


import triton.language as tl


import functools


from typing import Callable


import logging


from functools import lru_cache


from typing import Type


from typing import AsyncIterator


import torch.distributed as dist


from torch import nn


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch.profiler import record_function


from triton.runtime.cache import FileCacheManager


from triton.runtime.cache import default_cache_dir


from triton.runtime.cache import default_dump_dir


from triton.runtime.cache import default_override_dir


import itertools


from types import SimpleNamespace


@triton.jit
def create_flashinfer_kv_indices_triton(req_to_token_ptr, req_pool_indices_ptr, page_kernel_lens_ptr, kv_indptr, kv_start_idx, kv_indices_ptr, max_context_len: 'tl.constexpr'):
    BLOCK_SIZE: 'tl.constexpr' = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)
    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid)
    req_to_token_ptr += req_pool_index * max_context_len
    kv_indices_ptr += kv_indices_offset
    ld_offset = kv_start + tl.arange(0, BLOCK_SIZE)
    st_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = ld_offset < kv_end
        data = tl.load(req_to_token_ptr + ld_offset, mask=mask)
        tl.store(kv_indices_ptr + st_offset, data, mask=mask)
        ld_offset += BLOCK_SIZE
        st_offset += BLOCK_SIZE


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(Q, K_Buffer, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, Att_Out, stride_req_to_tokens_b, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, att_stride_h, kv_group_num: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', logit_cap: 'tl.constexpr', Lk: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    reduce_dtype = Att_Out.dtype.element_ty
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        offs_buf_k = k_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :]
        k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n_new[:, None] < cur_batch_end_index) & (offs_d[None, :] < Lk), other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n)
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)


@triton.jit
def _fwd_kernel_stage2(logits, V_Buffer, Out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, stride_logic_h, stride_buf_vbs, stride_buf_vh, stride_obs, stride_oh, stride_req_to_token_b, kv_group_num: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', Lv: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v
    e_max = float('-inf')
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(Req_to_tokens + cur_batch_req_idx * stride_req_to_token_b + (start_n + offs_n), mask=start_n + offs_n < cur_batch_seq_len, other=0)
        qk = tl.load(logits + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n), mask=start_n + offs_n < cur_batch_seq_len, other=float('-inf'))
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_buf_vbs, mask=offs_d[None, :] < Lv)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max
    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_d < Lv)


@triton.jit
def _fwd_grouped_kernel_stage1(Q, K_Buffer, sm_scale, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, Att_Out, stride_req_to_tokens_b, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, att_stride_h, kv_group_num: 'tl.constexpr', q_head_num: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_DPE: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_H: 'tl.constexpr', logit_cap: 'tl.constexpr', Lk: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    start_n = tl.program_id(2)
    reduce_dtype = Att_Out.dtype.element_ty
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: 'tl.constexpr' = BLOCK_H
    else:
        VALID_BLOCK_H: 'tl.constexpr' = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_stard_index = start_n * BLOCK_N
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + offs_q + start_mark, mask=mask_h[:, None] & (offs_d[None, :] < Lk))
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        offs_buf_k = k_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
        k = tl.load(K_Buffer + offs_buf_k, mask=(offs_n_new[None, :] < cur_batch_end_index) & (offs_d[:, None] < Lk), other=0.0)
        qk = tl.dot(q, k)
        if BLOCK_DPE > 0:
            qpe = tl.load(Q + off_qpe + start_mark, mask=mask_h[:, None])
            offs_buf_kpe = k_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None]
            kpe = tl.load(K_Buffer + offs_buf_kpe, mask=offs_n_new[None, :] < cur_batch_end_index, other=0.0)
            qk += tl.dot(qpe, kpe)
        qk *= sm_scale
        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)
        offs_o = cur_head[:, None] * att_stride_h + (cur_batch_in_all_start_index + offs_n[None, :])
        tl.store(Att_Out + offs_o, qk, mask=mask_h[:, None] & (offs_n_new[None, :] < cur_batch_end_index))


@triton.jit
def _fwd_grouped_kernel_stage2(logits, V_Buffer, Out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, stride_logic_h, stride_buf_vbs, stride_buf_vh, stride_obs, stride_oh, stride_req_to_token_b, kv_group_num: 'tl.constexpr', q_head_num: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_H: 'tl.constexpr', Lv: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: 'tl.constexpr' = BLOCK_H
    else:
        VALID_BLOCK_H: 'tl.constexpr' = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_buf_v = cur_kv_head * stride_buf_vh + offs_d[None, :]
    v_ptrs = V_Buffer + offs_buf_v
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float('inf')
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(Req_to_tokens + cur_batch_req_idx * stride_req_to_token_b + (start_n + offs_n), mask=start_n + offs_n < cur_batch_seq_len, other=0)
        offs_qk = cur_head[:, None] * stride_logic_h + (cur_batch_start_loc + start_n + offs_n[None, :])
        qk = tl.load(logits + offs_qk, mask=mask_h[:, None] & (start_n + offs_n[None, :] < cur_batch_seq_len), other=float('-inf'))
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        e_sum = e_sum * old_scale + tl.sum(p, 1)
        v = tl.load(v_ptrs + v_index[:, None] * stride_buf_vbs, mask=offs_d[None, :] < Lv)
        p = p
        acc = acc * old_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max
    acc = acc / e_sum[:, None]
    off_o = cur_batch * stride_obs + cur_head[:, None] * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=mask_h[:, None] & (offs_d[None, :] < Lv))


@triton.jit
def _fwd_kernel_flash_decode_stage1(Q, K, V, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen, Mid_O, Mid_O_LogExpSum, stride_req_to_tokens_b, stride_req_to_tokens_s, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, gqa_group_size, BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
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
    q = tl.load(Q + off_q)
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float('-inf'))
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
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
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr'):
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
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@triton.jit
def _sparse_fwd_kernel_flash_decode_stage1(Q_Label, K_Label_Buffer, sm_scale, Req_to_tokens, B_Seqlen, Att_Out, stride_req_to_tokens_b, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, att_stride_h, att_stride_b, kv_group_num: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', logit_cap: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = 0
    cur_batch_end_index = cur_batch_seq_len
    min_val = -float('inf')
    att_value = tl.full([BLOCK_N], min_val, dtype=tl.float32)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    block_index = start_n * BLOCK_N
    block_mask = tl.where(block_index < cur_batch_seq_len, 1, 0)
    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q_Label + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch + offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        offs_buf_k = k_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :]
        k = tl.load(K_Label_Buffer + offs_buf_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        if logit_cap > 0:
            att_value = logit_cap * tanh(att_value / logit_cap)
    att_value = tl.where(offs_n < cur_batch_end_index, att_value, min_val)
    off_o = cur_head * att_stride_h + (cur_batch * att_stride_b + offs_n)
    tl.store(Att_Out + off_o, att_value)


@triton.jit
def _sparse_fwd_kernel_flash_decode_stage2(Q, K, V, sm_scale, Req_to_tokens, Topk_token_indices, Mid_O, Mid_O_LogExpSum, Heavy_token_num, stride_req_to_tokens_b, stride_topk_token_indices_h, stride_topk_token_indices_b, stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_o_eb, stride_mid_o_eh, gqa_group_size, BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(Heavy_token_num, cur_batch_start_index + BLOCK_SEQ)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    block_n_size = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1) // BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + off_q)
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(cur_batch_start_index, cur_batch_end_index, BLOCK_N):
        offs_n_new = start_n + offs_n
        topk_token_indices = tl.load(Topk_token_indices + stride_topk_token_indices_h * cur_head + stride_topk_token_indices_b * cur_batch + offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        k_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch + topk_token_indices, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float('-inf'))
        v = tl.load(V + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)
        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v, axis=0)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic
    need_store = 1
    for _ in range(0, need_store, 1):
        off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_d
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        tl.store(Mid_O + off_mid_o, acc / sum_exp)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@triton.jit
def _sparse_fwd_kernel_flash_decode_stage3(Mid_O, Mid_O_LogExpSum, O, seq_len, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_o_eb, stride_mid_o_eh, stride_obs, stride_oh, BLOCK_SEQ: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr'):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    block_n_size = tl.where(seq_len <= 0, 0, seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, stride_qbs, stride_qh, stride_kbs, stride_kh, stride_vbs, stride_vh, stride_obs, stride_oh, kv_group_num: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr', Lk: 'tl.constexpr'):
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
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :]
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
    mask_d = offs_d < Lk
    q = tl.load(Q + off_q, mask=(offs_m[:, None] < cur_batch_seq_len) & mask_d[None, :], other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    end_n = cur_batch_seq_len if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, cur_batch_seq_len)
    for start_n in range(0, block_mask * end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=(start_n + offs_n[None, :] < cur_batch_seq_len) & mask_d[:, None], other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        if IS_CAUSAL:
            qk += tl.where((start_n + offs_n[None, :] < cur_batch_seq_len) & (offs_m[:, None] >= start_n + offs_n[None, :]), 0, float('-inf'))
        else:
            qk += tl.where(start_n + offs_n[None, :] < cur_batch_seq_len, 0, float('-inf'))
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
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=(start_n + offs_n[:, None] < cur_batch_seq_len) & mask_d[None, :], other=0.0)
        p = p
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & mask_d[None, :])


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

