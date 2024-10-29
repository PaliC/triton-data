import sys
_module = sys.modules[__name__]
del sys
configs = _module
alignment_data_to_fasta = _module
add_non_unique_to_alignment_db = _module
create_alignment_db = _module
create_alignment_db_sharded = _module
unify_alignment_db_indices = _module
expand_alignment_duplicates = _module
fasta_to_clusterfile = _module
generate_chain_data_cache = _module
generate_mmcif_cache = _module
utils = _module
setup = _module
src = _module
common = _module
ligand_constants = _module
protein = _module
residue_constants = _module
config = _module
data = _module
components = _module
protein_datamodule = _module
protein_dataset = _module
data_modules = _module
data_pipeline = _module
data_transforms = _module
data_transforms_multimer = _module
errors = _module
feature_pipeline = _module
feature_processing_multimer = _module
input_pipeline = _module
input_pipeline_multimer = _module
mmcif_parsing = _module
msa_identifiers = _module
msa_pairing = _module
parsers = _module
templates = _module
hhblits = _module
hhsearch = _module
hmmbuild = _module
hmmsearch = _module
jackhmmer = _module
kalign = _module
nhmmer = _module
parse_msa_files = _module
diffusion = _module
augmentation = _module
noise = _module
models = _module
atom_attention = _module
atom_attention_naive = _module
attention_pair_bias = _module
dropout = _module
msa_kernel = _module
outer_product_mean = _module
primitives = _module
relative_position_encoding = _module
transition = _module
triangular_attention = _module
triangular_multiplicative_update = _module
diffusion_conditioning = _module
diffusion_module = _module
diffusion_transformer = _module
embedders = _module
heads = _module
model = _module
model_wrapper = _module
msa_module = _module
pairformer = _module
template = _module
train_alphafold = _module
block_utils = _module
checkpointing = _module
chunk_utils = _module
exponential_moving_average = _module
geometry = _module
alignment = _module
rigid_matrix_vector = _module
rotation_matrix = _module
vector = _module
instantiators = _module
logging_utils = _module
loss = _module
lr_schedulers = _module
pylogger = _module
rich_utils = _module
rigid_utils = _module
superimposition = _module
tensor_utils = _module
validation_metrics = _module
tests = _module
conftest = _module
helpers = _module
package_available = _module
run_if = _module
run_sh_command = _module
test_alignment = _module
test_atom_attention = _module
test_atom_attention_naive = _module
test_attention_pair_bias = _module
test_augmentation = _module
test_conditioning = _module
test_configs = _module
test_datamodules = _module
test_diffusion_module = _module
test_diffusion_transformer = _module
test_embedders = _module
test_eval = _module
test_loss = _module
test_msa_module = _module
test_pairformer = _module
test_primitives = _module
test_relpos = _module
test_sweeps = _module
test_train = _module
test_transition = _module
test_validation_metrics = _module

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


import torch.nn as nn


import torch.nn.functional as F


import math


import triton


import triton.language as tl


@triton.jit
def MSAFwdFused(v_si_ptr, b_ij_ptr, g_si_ptr, output_ptr, vw_ptr, logsumexp_ptr, C_hidden, N_head, C_LEN_POW2: 'tl.constexpr', RES_LEN_POW2: 'tl.constexpr', SEQ_LEN: 'tl.constexpr', RES_LEN: 'tl.constexpr', BLOCK_SIZE_ROW: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr', BLOCK_SIZE_SEQ: 'tl.constexpr'):
    pid_z = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    z_off = pid_z
    h_off = pid_h
    i_off = pid_i * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    offs_c = tl.arange(0, C_LEN_POW2)
    log2_e = 1.44269504089
    prev_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    new_row_max = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    l = tl.full((BLOCK_SIZE_ROW, 1), 0.0, dtype=tl.float32)
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = z_off * RES_LEN * RES_LEN * N_head + offs_i[:, None] * RES_LEN * N_head + offs_j[None, :] * N_head + h_off
        ij_mask = (offs_i < RES_LEN)[:, None] & (offs_j < RES_LEN)[None, :]
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        new_row_max = tl.maximum(tl.max(b, axis=1, keep_dims=True), prev_row_max)
        w = tl.exp2(log2_e * (b - new_row_max))
        l *= tl.exp2(log2_e * (prev_row_max - new_row_max))
        l += tl.sum(w, axis=1, keep_dims=True)
        for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
            for ch in range(0, C_hidden, 1):
                offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
                si_off = z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + offs_s[None, :] * RES_LEN * N_head * C_hidden + offs_i[:, None] * N_head * C_hidden + h_off * C_hidden + ch
                sj_off = z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + offs_s[None, :] * RES_LEN * N_head * C_hidden + offs_j[:, None] * N_head * C_hidden + h_off * C_hidden + ch
                si_mask = (offs_s < SEQ_LEN)[None, :] & (offs_i < RES_LEN)[:, None]
                sj_mask = (offs_s < SEQ_LEN)[None, :] & (offs_j < RES_LEN)[:, None]
                v = tl.load(v_si_ptr + sj_off, sj_mask, 0)
                vw = tl.load(output_ptr + si_off, si_mask, 0)
                vw = vw * tl.exp2(log2_e * (prev_row_max - new_row_max))
                vw = tl.dot(w, v, acc=vw)
                tl.store(output_ptr + si_off, vw, si_mask)
        prev_row_max = new_row_max
    for s in range(0, SEQ_LEN, BLOCK_SIZE_SEQ):
        for ch in range(0, C_hidden, 1):
            offs_s = s + tl.arange(0, BLOCK_SIZE_SEQ)
            si_off = z_off * SEQ_LEN * RES_LEN * N_head * C_hidden + offs_s[None, :] * RES_LEN * N_head * C_hidden + offs_i[:, None] * N_head * C_hidden + h_off * C_hidden + ch
            si_mask = (offs_s < SEQ_LEN)[None, :] & (offs_i < RES_LEN)[:, None]
            g = tl.load(g_si_ptr + si_off, si_mask, 0)
            g = tl.sigmoid(g)
            vw = tl.load(output_ptr + si_off, si_mask, 0)
            vw = vw / l
            out = g * vw
            tl.store(output_ptr + si_off, out, si_mask)
            tl.store(vw_ptr + si_off, vw, si_mask)
    lse_off = z_off * RES_LEN * N_head + offs_i[:, None] * N_head + h_off
    lse_mask = (offs_i < RES_LEN)[:, None]
    tl.store(logsumexp_ptr + lse_off, new_row_max + tl.log(l), lse_mask)


@triton.jit
def MSABwdFused(b_ij_ptr, logsumexp_ptr, N_head, RES_LEN: 'tl.constexpr', BLOCK_SIZE_ROW: 'tl.constexpr', BLOCK_SIZE_COL: 'tl.constexpr'):
    pid_zh = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_z = pid_zh // N_head
    pid_h = pid_zh % N_head
    log2_e = 1.44269504089
    z_off = pid_z
    h_off = pid_h
    i_off = pid_i * BLOCK_SIZE_ROW
    offs_i = i_off + tl.arange(0, BLOCK_SIZE_ROW)
    lse_off = z_off * RES_LEN * N_head + offs_i[:, None] * N_head + h_off
    lse_mask = (offs_i < RES_LEN)[:, None]
    logsumexp = tl.load(logsumexp_ptr + lse_off, lse_mask, 0)
    for j in range(0, RES_LEN, BLOCK_SIZE_COL):
        offs_j = j + tl.arange(0, BLOCK_SIZE_COL)
        b_offs = z_off * RES_LEN * RES_LEN * N_head + offs_i[:, None] * RES_LEN * N_head + offs_j[None, :] * N_head + h_off
        ij_mask = (offs_i < RES_LEN)[:, None] & (offs_j < RES_LEN)[None, :]
        b = tl.load(b_ij_ptr + b_offs, ij_mask, -INF)
        b = tl.exp2(log2_e * (b - logsumexp))
        tl.store(b_ij_ptr + b_offs, b, ij_mask)

