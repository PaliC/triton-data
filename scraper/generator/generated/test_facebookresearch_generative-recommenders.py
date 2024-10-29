import sys
_module = sys.modules[__name__]
del sys
dataset = _module
eval = _module
item_features = _module
preprocessor = _module
reco_dataset = _module
indexing = _module
candidate_index = _module
mips_top_k = _module
utils = _module
modeling = _module
initialization = _module
ndp_module = _module
sequential = _module
autoregressive_losses = _module
embedding_modules = _module
encoder_utils = _module
features = _module
hstu = _module
input_features_preprocessors = _module
output_postprocessors = _module
sasrec = _module
dot_product = _module
mol = _module
similarity_module = _module
similarity_utils = _module
data_loader = _module
train = _module
main = _module
triton_addmm = _module
triton_ragged_hstu_attention = _module
preprocess_public_data = _module
run_fractal_expansion = _module
ragged_hstu_attention_test = _module

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


from typing import List


import torch


import triton


import triton.language as tl


from typing import Optional


from typing import Tuple


ENABLE_FULL_TURNING_SPACE = False


def get_mm_configs() ->List[triton.Config]:
    if torch.version.hip:
        if ENABLE_FULL_TURNING_SPACE:
            block_m_range = [32, 64, 128, 256]
            block_n_range = [32, 64, 128, 256]
            block_k_range = [32, 64]
            group_m_range = [4, 8]
            matrix_instr_nonkdim_range = [16]
            waves_per_eu_range = [0]
            kpack_range = [1, 2]
            num_warps_range = [4, 8]
            num_stage_range = [0]
        else:
            block_m_range = [256]
            block_n_range = [256]
            block_k_range = [32]
            group_m_range = [8]
            matrix_instr_nonkdim_range = [16]
            waves_per_eu_range = [0]
            kpack_range = [2]
            num_warps_range = [8]
            num_stage_range = [0]
        return [triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'GROUP_M': group_m, 'matrix_instr_nonkdim': matrix_instr_nonkdim, 'waves_per_eu': waves_per_eu, 'kpack': kpack}, num_stages=num_stages, num_warps=num_warps) for block_m in block_m_range for block_n in block_n_range for block_k in block_k_range for group_m in group_m_range for matrix_instr_nonkdim in matrix_instr_nonkdim_range for waves_per_eu in waves_per_eu_range for kpack in kpack_range for num_stages in num_stage_range for num_warps in num_warps_range]
    else:
        return [triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2)]


@triton.autotune(configs=get_mm_configs(), key=['N', 'K'])
@triton.jit
def _addmm_fwd(x_ptr, w_ptr, y_ptr, z_ptr, M, N, K, stride_xm, stride_xk, stride_wk, stride_wn, stride_ym, stride_yn, stride_zm, stride_zn, BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', BLOCK_K: 'tl.constexpr', GROUP_M: 'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BROADCAST_Y: 'tl.constexpr'):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + pid % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        y_ptr += pid_n * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m * BLOCK_M * stride_ym
        y_ptr += pid_n * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = accumulator + y.to(tl.float32)
    z_ptr += pid_m * BLOCK_M * stride_zm
    z_ptr += pid_n * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


@triton.jit
def _ragged_hstu_attn_fwd_one_block(start_n, seq_len, offs_m, offs_n, mask_m, mask_n, q, K_block_ptr, V_block_ptr, n_targets, ts_1_ptrs, ts_0, TW, PW, alpha, MAX_SEQ_LEN, num_buckets, max_pos_ind, max_attn_len, time_bucket_incr, time_bucket_div, time_delta, bias_ptrs, attn_scale, contextual_seq_len, INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN: 'tl.constexpr', ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS: 'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS: 'tl.constexpr', HAS_ATTN_SCALE: 'tl.constexpr', HAS_MAX_ATTN_LEN: 'tl.constexpr', HAS_CONTEXTUAL_SEQ_LEN: 'tl.constexpr', IS_DELTA_Q: 'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr'):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option='zero')
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == 'lower_triangular':
            offs_m = tl.where(offs_m < seq_len - n_targets, offs_m, seq_len - n_targets)
            offs_n = tl.where(offs_n < seq_len - n_targets, offs_n, seq_len - n_targets)
        elif INVALID_MASK_TYPE == 'upper_triangular':
            offs_m = tl.where(offs_m > n_targets - 1, offs_m, n_targets - 1)
            offs_n = tl.where(offs_n > n_targets - 1, offs_n, n_targets - 1)
    offs_n_minus_m = offs_n[None, :] - offs_m[:, None]
    if HAS_MAX_ATTN_LEN:
        if INVALID_MASK_TYPE == 'lower_triangular':
            invalid_mask = invalid_mask or offs_n_minus_m < 0 and offs_n_minus_m >= -max_attn_len
        elif INVALID_MASK_TYPE == 'upper_triangular':
            invalid_mask = invalid_mask or offs_n_minus_m > 0 and offs_n_minus_m <= max_attn_len
    elif INVALID_MASK_TYPE == 'lower_triangular':
        invalid_mask = invalid_mask or offs_n_minus_m < 0
    elif INVALID_MASK_TYPE == 'upper_triangular':
        invalid_mask = invalid_mask or offs_n_minus_m > 0
    if HAS_CONTEXTUAL_SEQ_LEN:
        if INVALID_MASK_TYPE == 'lower_triangular':
            row_filter = offs_m < contextual_seq_len
            if HAS_MULTIPLE_TARGETS:
                col_filter = offs_n < seq_len - n_targets
            else:
                col_filter = offs_n < seq_len
            invalid_mask = invalid_mask or row_filter[:, None] and col_filter[None, :]
    if ATTN_BIAS_TYPE == 'fused':
        attn_bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                ts_1 = tl.load(ts_1_ptrs + start_n, mask=mask_n)
            else:
                ts_1 = tl.load(ts_1_ptrs + start_n + 1, mask=mask_n)
            ts = ts_0[:, None] - ts_1[None, :]
            ts = ts + time_delta
            ts = tl.where(ts > 1e-06, ts, 1e-06)
            ts = ts * (1.0 / time_bucket_incr)
            if BUCKET_FN == 'log':
                ts = tl.log(ts)
            elif BUCKET_FN == 'sqrt':
                ts = tl.sqrt(ts)
            ts = ts * (1.0 / time_bucket_div)
            ts = ts
            ts = tl.where(ts > 0, ts, 0)
            ts = tl.where(ts < num_buckets, ts, num_buckets)
            ts_w = tl.load(TW + ts, mask=mask_m[:, None] and mask_n[None, :])
            attn_bias = attn_bias + ts_w
        if USE_POS_BIAS:
            if HAS_MAX_POS_IND:
                offs_pos_w = offs_n_minus_m + max_pos_ind - 1
                offs_pos_w = tl.where(offs_pos_w > 0, offs_pos_w, 0)
                offs_pos_w = tl.where(offs_pos_w < 2 * max_pos_ind - 2, offs_pos_w, 2 * max_pos_ind - 2)
            else:
                offs_pos_w = offs_n_minus_m + MAX_SEQ_LEN - 1
            pos_w = tl.load(PW + offs_pos_w, mask=mask_m[:, None] and mask_n[None, :])
            attn_bias = attn_bias + pos_w
        qk = qk + attn_bias
    elif ATTN_BIAS_TYPE == 'separate':
        attn_bias = tl.load(bias_ptrs + start_n, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        qk = qk + attn_bias
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    if HAS_ATTN_SCALE:
        silu = silu * attn_scale[:, None]
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option='zero')
    silu = silu
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _ragged_hstu_attn_fwd_compute(Q, K, V, seq_offsets, TS, TW, PW, Bias, seq2_offsets, delta_x_offsets, num_targets, Scale, Out, stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh, stride_sz, stride_sm, stride_ts, stride_om, stride_oh, alpha, Z, H, MAX_SEQ_LEN, DimQ, DimV, DeltaSize, num_buckets, max_pos_ind, time_bucket_incr, time_bucket_div, time_delta, contextual_seq_len, off_z, off_h, pid, INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN: 'tl.constexpr', ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS: 'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS: 'tl.constexpr', HAS_ATTN_SCALE: 'tl.constexpr', IS_DELTA_Q: 'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BLOCK_D_Q: 'tl.constexpr', BLOCK_D_V: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', max_attn_len: 'tl.constexpr', HAS_MAX_ATTN_LEN: 'tl.constexpr', HAS_CONTEXTUAL_SEQ_LEN: 'tl.constexpr'):
    seq_start = tl.load(seq_offsets + off_z)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = seq_end - seq_start
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        delta_start = tl.load(delta_x_offsets + off_z * DeltaSize)
        start_m = start_m_delta + delta_start - seq_start
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z)
        else:
            n_targets = None
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        if IS_DELTA_Q:
            Q_block_ptr = tl.make_block_ptr(base=Q + off_h * stride_qh + off_z * DeltaSize * stride_qm, shape=(DeltaSize, BLOCK_D_Q), strides=(stride_qm, 1), offsets=(start_m_delta, 0), block_shape=(BLOCK_M, BLOCK_D_Q), order=(1, 0))
        else:
            Q_block_ptr = tl.make_block_ptr(base=Q + off_h * stride_qh + seq_start * stride_qm, shape=(seq_len, BLOCK_D_Q), strides=(stride_qm, 1), offsets=(start_m, 0), block_shape=(BLOCK_M, BLOCK_D_Q), order=(1, 0))
        K_block_ptr = tl.make_block_ptr(base=K + off_h * stride_kh + seq_start * stride_kn, shape=(BLOCK_D_Q, seq_len), strides=(1, stride_kn), offsets=(0, 0), block_shape=(BLOCK_D_Q, BLOCK_N), order=(0, 1))
        V_block_ptr = tl.make_block_ptr(base=V + off_h * stride_vh + seq_start * stride_vn, shape=(seq_len, BLOCK_D_V), strides=(stride_vn, 1), offsets=(0, 0), block_shape=(BLOCK_N, BLOCK_D_V), order=(1, 0))
        mask_m = offs_m < seq_len
        if ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS:
            ts_0_ptrs = TS + off_z * stride_ts + offs_m
            ts_1_ptrs = TS + off_z * stride_ts + offs_n
            if CAUSAL:
                ts_0 = tl.load(ts_0_ptrs + 1, mask=mask_m)
            else:
                ts_0 = tl.load(ts_0_ptrs, mask=mask_m)
        elif ATTN_BIAS_TYPE == 'separate':
            seq2_start = tl.load(seq2_offsets + off_z)
            bias_start = seq2_start * H + off_h * seq_len * seq_len
            off_bias = offs_m[:, None] * seq_len + offs_n[None, :]
            bias_ptrs = Bias + bias_start + off_bias
        if HAS_ATTN_SCALE:
            scale_ptrs = Scale + off_z * stride_sz
            attn_scale = tl.load(scale_ptrs + offs_m * stride_sm, mask=offs_m < seq_len)
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option='zero')
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if INVALID_MASK_TYPE == 'lower_triangular':
            if HAS_MULTIPLE_TARGETS:
                if HAS_MAX_ATTN_LEN:
                    start_m_index = seq_len - n_targets if start_m > seq_len - n_targets else start_m
                    low = start_m_index - max_attn_len
                    low = low if low > 0 else 0
                else:
                    low = 0
                uih_end = (seq_len - n_targets + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                if uih_end < start_m:
                    high = seq_len - n_targets
                else:
                    high = start_m + BLOCK_M
                if HAS_CONTEXTUAL_SEQ_LEN:
                    if start_m < contextual_seq_len:
                        high = seq_len - n_targets
            else:
                if HAS_MAX_ATTN_LEN:
                    low = start_m - max_attn_len
                    low = low if low > 0 else 0
                else:
                    low = 0
                high = start_m + BLOCK_M
                if HAS_CONTEXTUAL_SEQ_LEN:
                    if start_m < contextual_seq_len:
                        high = seq_len
        elif INVALID_MASK_TYPE == 'upper_triangular':
            low = start_m
            high = seq_len
        if low > 0:
            K_block_ptr = tl.advance(K_block_ptr, (0, low))
            V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        for start_n in range(low, high, BLOCK_N):
            cur_offs_n = offs_n + start_n
            mask_n = cur_offs_n < seq_len
            acc += _ragged_hstu_attn_fwd_one_block(start_n=start_n, seq_len=seq_len, offs_m=offs_m, offs_n=cur_offs_n, mask_m=mask_m, mask_n=mask_n, q=q, K_block_ptr=K_block_ptr, V_block_ptr=V_block_ptr, n_targets=n_targets if HAS_MULTIPLE_TARGETS else None, ts_1_ptrs=ts_1_ptrs if ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS else None, ts_0=ts_0 if ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS else None, TW=TW, PW=PW, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN, num_buckets=num_buckets, max_pos_ind=max_pos_ind, max_attn_len=max_attn_len, time_bucket_incr=time_bucket_incr, time_bucket_div=time_bucket_div, time_delta=time_delta, bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == 'separate' else None, attn_scale=attn_scale if HAS_ATTN_SCALE else None, contextual_seq_len=contextual_seq_len, INVALID_MASK_TYPE=INVALID_MASK_TYPE, CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=USE_POS_BIAS, HAS_MAX_POS_IND=HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, HAS_ATTN_SCALE=HAS_ATTN_SCALE, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN, HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, IS_DELTA_Q=IS_DELTA_Q, ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if HAS_MULTIPLE_TARGETS and INVALID_MASK_TYPE == 'lower_triangular':
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = low_delta - uih_end
                K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(low_delta, high_delta, BLOCK_N, num_stages=0):
                    cur_offs_n = offs_n + start_delta
                    mask_n = cur_offs_n < seq_len
                    acc += _ragged_hstu_attn_fwd_one_block(start_n=start_delta, seq_len=seq_len, offs_m=offs_m, offs_n=cur_offs_n, mask_m=mask_m, mask_n=mask_n, q=q, K_block_ptr=K_block_ptr, V_block_ptr=V_block_ptr, n_targets=n_targets if HAS_MULTIPLE_TARGETS else None, ts_1_ptrs=ts_1_ptrs if ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS else None, ts_0=ts_0 if ATTN_BIAS_TYPE == 'fused' and USE_TIME_BIAS else None, TW=TW, PW=PW, alpha=alpha, MAX_SEQ_LEN=MAX_SEQ_LEN, num_buckets=num_buckets, max_pos_ind=max_pos_ind, max_attn_len=max_attn_len, time_bucket_incr=time_bucket_incr, time_bucket_div=time_bucket_div, time_delta=time_delta, bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == 'separate' else None, attn_scale=attn_scale if HAS_ATTN_SCALE else None, contextual_seq_len=contextual_seq_len, INVALID_MASK_TYPE=INVALID_MASK_TYPE, CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=USE_POS_BIAS, HAS_MAX_POS_IND=HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, HAS_ATTN_SCALE=HAS_ATTN_SCALE, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN, HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, IS_DELTA_Q=IS_DELTA_Q, ALLOW_TF32=ALLOW_TF32, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = (off_z * DeltaSize + offs_m_delta[:, None]) * stride_om + off_h * stride_oh + offs_v_d[None, :]
            out_ptrs = Out + off_o
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = (seq_start + offs_m[:, None]) * stride_om + off_h * stride_oh + offs_v_d[None, :]
            out_ptrs = Out + off_o
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


def _get_fw_configs() ->List[triton.Config]:
    configs = []
    if torch.version.hip:
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64]:
                for num_stages in [0, 1]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2]:
                                configs.append(triton.Config({'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'matrix_instr_nonkdim': matrix_instr_nonkdim, 'waves_per_eu': waves_per_eu}, num_stages=num_stages, num_warps=num_warps))
    else:
        configs = [triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_stages=2, num_warps=2), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=2), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=2), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=4, num_warps=8), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=2), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=2), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=2), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=2), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4), triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8)]
    return configs


@triton.autotune(configs=_get_fw_configs(), key=['Z', 'H', 'AUTOTUNE_MAX_SEQ_LEN', 'DimQ', 'DimV', 'BUCKET_FN', 'ATTN_BIAS_TYPE', 'DeltaSize', 'IS_DELTA_Q'])
@triton.jit
def _ragged_hstu_attn_fwd(Q, K, V, sort_by_length_indices, seq_offsets, TS, TW, PW, Bias, seq2_offsets, delta_x_offsets, num_targets, Scale, Out, stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh, stride_sz, stride_sm, stride_ts, stride_om, stride_oh, alpha, Z, H, MAX_SEQ_LEN, AUTOTUNE_MAX_SEQ_LEN, DimQ, DimV, DeltaSize, num_buckets, max_pos_ind, time_bucket_incr, time_bucket_div, time_delta, contextual_seq_len, INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN: 'tl.constexpr', ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS: 'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS: 'tl.constexpr', HAS_ATTN_SCALE: 'tl.constexpr', IS_DELTA_Q: 'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BLOCK_D_Q: 'tl.constexpr', BLOCK_D_V: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', max_attn_len: 'tl.constexpr', HAS_MAX_ATTN_LEN: 'tl.constexpr', HAS_CONTEXTUAL_SEQ_LEN: 'tl.constexpr', HAS_SORT_BY_LENGTH_INDICES: 'tl.constexpr'):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    _ragged_hstu_attn_fwd_compute(Q=Q, K=K, V=V, seq_offsets=seq_offsets, TS=TS, TW=TW, PW=PW, Bias=Bias, seq2_offsets=seq2_offsets, delta_x_offsets=delta_x_offsets, num_targets=num_targets, Scale=Scale, Out=Out, stride_qm=stride_qm, stride_qh=stride_qh, stride_kn=stride_kn, stride_kh=stride_kh, stride_vn=stride_vn, stride_vh=stride_vh, stride_sz=stride_sz, stride_sm=stride_sm, stride_ts=stride_ts, stride_om=stride_om, stride_oh=stride_oh, alpha=alpha, Z=Z, H=H, MAX_SEQ_LEN=MAX_SEQ_LEN, DimQ=DimQ, DimV=DimV, DeltaSize=DeltaSize, num_buckets=num_buckets, max_pos_ind=max_pos_ind, time_bucket_incr=time_bucket_incr, time_bucket_div=time_bucket_div, time_delta=time_delta, contextual_seq_len=contextual_seq_len, off_z=off_z, off_h=off_h, pid=pid, INVALID_MASK_TYPE=INVALID_MASK_TYPE, CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=USE_POS_BIAS, HAS_MAX_POS_IND=HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, HAS_ATTN_SCALE=HAS_ATTN_SCALE, IS_DELTA_Q=IS_DELTA_Q, ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=BLOCK_D_V, max_attn_len=max_attn_len, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN, HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)


@triton.autotune(configs=_get_fw_configs(), key=['Z', 'H', 'AUTOTUNE_MAX_SEQ_LEN', 'DimQ', 'DimV', 'BUCKET_FN', 'ATTN_BIAS_TYPE', 'DeltaSize', 'IS_DELTA_Q'])
@triton.jit
def _ragged_hstu_attn_fwd_persistent(Q, K, V, sort_by_length_indices, seq_offsets, TS, TW, PW, Bias, seq2_offsets, delta_x_offsets, num_targets, Scale, Out, stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh, stride_sz, stride_sm, stride_ts, stride_om, stride_oh, alpha, Z, H, MAX_SEQ_LEN, AUTOTUNE_MAX_SEQ_LEN, DimQ, DimV, DeltaSize, num_buckets, max_pos_ind, time_bucket_incr, time_bucket_div, time_delta, contextual_seq_len, INVALID_MASK_TYPE: 'tl.constexpr', CAUSAL: 'tl.constexpr', BUCKET_FN: 'tl.constexpr', ATTN_BIAS_TYPE: 'tl.constexpr', USE_TIME_BIAS: 'tl.constexpr', USE_POS_BIAS: 'tl.constexpr', HAS_MAX_POS_IND: 'tl.constexpr', HAS_MULTIPLE_TARGETS: 'tl.constexpr', HAS_ATTN_SCALE: 'tl.constexpr', IS_DELTA_Q: 'tl.constexpr', ALLOW_TF32: 'tl.constexpr', BLOCK_D_Q: 'tl.constexpr', BLOCK_D_V: 'tl.constexpr', BLOCK_M: 'tl.constexpr', BLOCK_N: 'tl.constexpr', max_attn_len: 'tl.constexpr', HAS_MAX_ATTN_LEN: 'tl.constexpr', HAS_CONTEXTUAL_SEQ_LEN: 'tl.constexpr', HAS_SORT_BY_LENGTH_INDICES: 'tl.constexpr'):
    n_tile_num = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_tiles = n_tile_num * Z * H
    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1
    tile_idx = prog_id
    for _ in range(0, tiles_per_sm):
        pid = (total_tiles - tile_idx - 1) // (Z * H)
        off_hz = (total_tiles - tile_idx - 1) % (Z * H)
        off_z = off_hz // H
        off_h = off_hz % H
        _ragged_hstu_attn_fwd_compute(Q=Q, K=K, V=V, seq_offsets=seq_offsets, TS=TS, TW=TW, PW=PW, Bias=Bias, seq2_offsets=seq2_offsets, delta_x_offsets=delta_x_offsets, num_targets=num_targets, Scale=Scale, Out=Out, stride_qm=stride_qm, stride_qh=stride_qh, stride_kn=stride_kn, stride_kh=stride_kh, stride_vn=stride_vn, stride_vh=stride_vh, stride_sz=stride_sz, stride_sm=stride_sm, stride_ts=stride_ts, stride_om=stride_om, stride_oh=stride_oh, alpha=alpha, Z=Z, H=H, MAX_SEQ_LEN=MAX_SEQ_LEN, DimQ=DimQ, DimV=DimV, DeltaSize=DeltaSize, num_buckets=num_buckets, max_pos_ind=max_pos_ind, time_bucket_incr=time_bucket_incr, time_bucket_div=time_bucket_div, time_delta=time_delta, contextual_seq_len=contextual_seq_len, off_z=off_z, off_h=off_h, pid=pid, INVALID_MASK_TYPE=INVALID_MASK_TYPE, CAUSAL=CAUSAL, BUCKET_FN=BUCKET_FN, ATTN_BIAS_TYPE=ATTN_BIAS_TYPE, USE_TIME_BIAS=USE_TIME_BIAS, USE_POS_BIAS=USE_POS_BIAS, HAS_MAX_POS_IND=HAS_MAX_POS_IND, HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS, HAS_ATTN_SCALE=HAS_ATTN_SCALE, IS_DELTA_Q=IS_DELTA_Q, ALLOW_TF32=ALLOW_TF32, BLOCK_D_Q=BLOCK_D_Q, BLOCK_D_V=BLOCK_D_V, max_attn_len=max_attn_len, HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN, HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
        tile_idx += num_progs

