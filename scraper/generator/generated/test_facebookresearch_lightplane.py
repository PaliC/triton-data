import sys
_module = sys.modules[__name__]
del sys
conf = _module
fit_single_scene = _module
dataloader = _module
co3d_dataset = _module
config_util = _module
dataset = _module
dataset_base = _module
llff_dataset = _module
load_llff = _module
nerf_dataset = _module
nsvf_dataset = _module
pt3d_cow_dataset = _module
util = _module
camera_util = _module
grid_util = _module
io_util = _module
metric = _module
nnfm_loss = _module
renderer_util = _module
lightplane = _module
lightplane_renderer = _module
lightplane_splatter = _module
misc_utils = _module
mlp_utils = _module
naive_renderer = _module
naive_splatter = _module
ray_utils = _module
renderer_module = _module
splatter_module = _module
triton_src = _module
const = _module
func_util = _module
fwbw_util = _module
grid_sample_util = _module
rand_util = _module
ray_util = _module
cog_util = _module
renderer_bw = _module
renderer_fw = _module
renderer_mlp_util = _module
splatter_bw = _module
splatter_fw = _module
splatter_mlp_util = _module
visualize = _module
setup = _module
tests = _module
renderer_speed_benchmark = _module
splatter_speed_benchmark = _module
test_randn = _module
test_renderer_with_autograd = _module
test_splatter_with_autograd = _module
utils = _module

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


import random


import time


import warnings


from typing import List


from typing import Optional


from typing import Tuple


import torch


import copy


from typing import Union


from enum import Enum


from logging import getLogger


import logging


from torch.utils.checkpoint import checkpoint


from typing import Any


import functools


import triton


import triton.language as tl


import itertools


@triton.jit
def d_sigmoid(dy, x):
    s = tl.sigmoid(x)
    return dy * s * (1 - s)


@triton.jit
def _softplus(x):
    z = tl.where(x >= 0, x + tl.log(1 + tl.exp(-x)), tl.log(1 + tl.exp(x)))
    return z


@triton.jit
def _d_softplus(grad, x):
    z = tl.where(x >= 0, 1 / (1 + tl.exp(-x)), 1 - 1 / (1 + tl.exp(x)))
    return grad * z


ALLOW_TF32 = False


@triton.jit
def d_linear(d_y, w, b, x):
    d_x = tl.dot(d_y, tl.trans(w), allow_tf32=ALLOW_TF32)
    d_w = tl.dot(tl.trans(d_y), x, allow_tf32=ALLOW_TF32)
    d_b = tl.sum(d_y, axis=0)
    return d_x, d_w, d_b


@triton.jit
def d_linear_relu(d_y, w, b, xwb, x):
    d_y_relu = d_y * (xwb > 0.0)
    return d_linear(d_y_relu, w, b, x)


@triton.jit
def fwbw_init(directions, origins, grid_idx, near, far, rays_encoding, inject_noise_seed, DIM_IN_COLOR: 'tl.constexpr', DIM_OUT_COLOR: 'tl.constexpr', num_samples: 'tl.constexpr', num_samples_inf: 'tl.constexpr', num_rays: 'tl.constexpr', C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    tot_num_samples = num_samples + num_samples_inf
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = offs < num_rays
    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1
    offs_features = pid * BLOCK_SIZE * DIM_OUT_COLOR + DIM_OUT_COLOR * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, DIM_OUT_COLOR)[None, :]
    offs_features_mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] < num_rays
    center_x = tl.load(origins + offs_x, mask=offs_x < num_rays * 3)
    center_y = tl.load(origins + offs_y, mask=offs_y < num_rays * 3)
    center_z = tl.load(origins + offs_z, mask=offs_z < num_rays * 3)
    ray_x = tl.load(directions + offs_x, mask=offs_x < num_rays * 3)
    ray_y = tl.load(directions + offs_y, mask=offs_y < num_rays * 3)
    ray_z = tl.load(directions + offs_z, mask=offs_z < num_rays * 3)
    near_buffer = tl.load(near + offs, mask=offs_mask)
    far_buffer = tl.load(far + offs, mask=offs_mask)
    grid_idx_buffer = tl.load(grid_idx + offs, mask=offs_mask)
    seed_buffer = tl.load(inject_noise_seed + offs, mask=offs < num_rays)
    sample_index_buffer = tl.arange(0, BLOCK_SIZE) * tot_num_samples + pid * BLOCK_SIZE * tot_num_samples + 1
    rays_encoding_buffer = tl.load(rays_encoding + pid * BLOCK_SIZE * DIM_IN_COLOR + DIM_IN_COLOR * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, DIM_IN_COLOR)[None, :], mask=offs_features_mask)
    one_scaffold = tl.full((BLOCK_SIZE,), 1.0, tl.float32)
    zero_value = tl.zeros((BLOCK_SIZE,), tl.float32)
    one_vec = tl.full((BLOCK_SIZE, C), 1.0, tl.float32)
    zero_color = tl.zeros((BLOCK_SIZE, DIM_OUT_COLOR), tl.float32)
    return tot_num_samples, pid, offs, offs_mask, offs_features, offs_features_mask, center_x, center_y, center_z, ray_x, ray_y, ray_z, near_buffer, far_buffer, grid_idx_buffer, seed_buffer, sample_index_buffer, rays_encoding_buffer, one_scaffold, zero_value, one_vec, zero_color


@triton.jit
def fwbw_splatter_init(directions, origins, grid_idx, near, far, splatting_feature, mask, num_samples: 'tl.constexpr', num_samples_inf: 'tl.constexpr', num_rays: 'tl.constexpr', grid_channel: 'tl.constexpr', feature_channel: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    tot_num_samples = num_samples + num_samples_inf
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = offs < num_rays
    offs_x = pid * BLOCK_SIZE * 3 + tl.arange(0, BLOCK_SIZE) * 3
    offs_y = offs_x + 1
    offs_z = offs_y + 1
    offs_features = pid * BLOCK_SIZE * feature_channel + feature_channel * tl.arange(0, BLOCK_SIZE)[:, None] + tl.arange(0, feature_channel)[None, :]
    offs_features_mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] < num_rays
    center_x = tl.load(origins + offs_x, mask=offs_x < num_rays * 3)
    center_y = tl.load(origins + offs_y, mask=offs_y < num_rays * 3)
    center_z = tl.load(origins + offs_z, mask=offs_z < num_rays * 3)
    ray_x = tl.load(directions + offs_x, mask=offs_x < num_rays * 3)
    ray_y = tl.load(directions + offs_y, mask=offs_y < num_rays * 3)
    ray_z = tl.load(directions + offs_z, mask=offs_z < num_rays * 3)
    near_buffer = tl.load(near + offs, mask=offs_mask)
    far_buffer = tl.load(far + offs, mask=offs_mask)
    grid_idx_buffer = tl.load(grid_idx + offs, mask=offs_mask)
    sample_index_buffer = tl.arange(0, BLOCK_SIZE) * tot_num_samples + pid * BLOCK_SIZE * tot_num_samples + 1
    feature = tl.load(splatting_feature + offs_features, mask=offs_features_mask)
    mask = tl.load(mask + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None], mask=offs_features_mask)
    mask = tl.view(mask, (BLOCK_SIZE, 1))
    return tot_num_samples, pid, offs, offs_mask, offs_features, offs_features_mask, center_x, center_y, center_z, ray_x, ray_y, ray_z, near_buffer, far_buffer, grid_idx_buffer, sample_index_buffer, feature, mask


@triton.jit
def _floor(x):
    return x - x % 1


@triton.jit
def _round(x):
    return _floor(x + 0.5)


@triton.jit
def is_in_bounds(x, y, z, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    in_bounds = (tl.abs(x) <= 1) * (tl.abs(y) <= 1) * (tl.abs(z) <= 1)
    if C == 1:
        in_bounds_mask = tl.view(in_bounds, (BLOCK_SIZE,))
    else:
        in_bounds_mask = tl.broadcast_to(in_bounds[:, None], (BLOCK_SIZE, C))
    return in_bounds_mask


@triton.jit
def _splat_3d(to_splat, grad_image, w, batch_index, ix, iy, iz, ID, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1)
    w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0))
    w = tl.view(w[:, None], (BLOCK_SIZE, 1))
    offs = tl.view((batch_index * ID * IW * IH * C + iz_ * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :], (BLOCK_SIZE, C))
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _splat_2d(to_splat, grad_image, w, batch_index, ix, iy, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1)
    w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW))
    w = tl.view(w[:, None], (BLOCK_SIZE, 1))
    offs = tl.view((batch_index * IW * IH * C + iy_ * IW * C + ix_ * C)[:, None] + Coffs[None, :], (BLOCK_SIZE, C))
    tl.atomic_add(grad_image + offs, w * to_splat)


@triton.jit
def _get_plane_grid_sample_info(gi, ix_in, iy_in, IH, IW, feature_grid_size, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    BS = tl.load(feature_grid_size + offs + 0)
    grid_numel = BS * IH * IW * C
    grid_numel = tl.sum(grid_numel, axis=0) // BLOCK_SIZE
    ix11 = (ix_in + 1) / 2 * IW - 0.5
    iy11 = (iy_in + 1) / 2 * IH - 0.5
    ix = ix11 * (IW > 1)
    iy = iy11 * (IH > 1)
    ix0 = _floor(ix)
    iy0 = _floor(iy)
    return ix, iy, ix0, iy0, grid_numel


@triton.jit
def _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0):
    return ix0, iy0, ix0, iy0 + 1, ix0 + 1, iy0, ix0 + 1, iy0 + 1, ix - ix0, iy - iy0


@triton.jit
def _plane_grid_splat_one(gi, to_splat, grad_feature_grid, feature_grid_size, batch_index, ix_in, iy_in, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    ix, iy, ix0, iy0, grid_numel = _get_plane_grid_sample_info(gi, ix_in, iy_in, IH, IW, feature_grid_size, C, BLOCK_SIZE)
    V00x, V00y, V10x, V10y, V01x, V01y, V11x, V11y, x, y = _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0)
    to_splat_now = to_splat
    _splat_2d(to_splat_now, grad_feature_grid, (1 - x) * (1 - y), batch_index, V00x, V00y, IH, IW, C, BLOCK_SIZE)
    _splat_2d(to_splat_now, grad_feature_grid, (1 - x) * y, batch_index, V10x, V10y, IH, IW, C, BLOCK_SIZE)
    _splat_2d(to_splat_now, grad_feature_grid, x * (1 - y), batch_index, V01x, V01y, IH, IW, C, BLOCK_SIZE)
    _splat_2d(to_splat_now, grad_feature_grid, x * y, batch_index, V11x, V11y, IH, IW, C, BLOCK_SIZE)
    return grid_numel


@triton.jit
def _get_voxel_grid_sample_info(gi, ix_in, iy_in, iz_in, ID, IH, IW, feature_grid_size, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    BS = tl.load(feature_grid_size + offs + 0)
    grid_numel = BS * ID * IH * IW * C
    grid_numel = tl.sum(grid_numel, axis=0) // BLOCK_SIZE
    ix11 = (ix_in + 1) / 2 * IW - 0.5
    iy11 = (iy_in + 1) / 2 * IH - 0.5
    iz11 = (iz_in + 1) / 2 * ID - 0.5
    ix = ix11 * (IW > 1)
    iy = iy11 * (IH > 1)
    iz = iz11 * (ID > 1)
    ix0 = _floor(ix)
    iy0 = _floor(iy)
    iz0 = _floor(iz)
    return ix, iy, iz, ix0, iy0, iz0, grid_numel


@triton.jit
def _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0):
    return ix0, iy0, iz0, ix0, iy0, iz0 + 1, ix0, iy0 + 1, iz0, ix0 + 1, iy0, iz0, ix0 + 1, iy0, iz0 + 1, ix0 + 1, iy0 + 1, iz0, ix0, iy0 + 1, iz0 + 1, ix0 + 1, iy0 + 1, iz0 + 1, ix - ix0, iy - iy0, iz - iz0


@triton.jit
def _voxel_grid_splat_one(gi, to_splat, grad_feature_grid, feature_grid_size, batch_index, ix_in, iy_in, iz_in, IH, IW, ID, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    ix, iy, iz, ix0, iy0, iz0, grid_numel = _get_voxel_grid_sample_info(gi, ix_in, iy_in, iz_in, ID, IH, IW, feature_grid_size, C, BLOCK_SIZE)
    V000x, V000y, V000z, V100x, V100y, V100z, V010x, V010y, V010z, V001x, V001y, V001z, V101x, V101y, V101z, V011x, V011y, V011z, V110x, V110y, V110z, V111x, V111y, V111z, x, y, z = _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * (1 - y) * (1 - z), batch_index, V000x, V000y, V000z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * (1 - y) * z, batch_index, V100x, V100y, V100z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * y * (1 - z), batch_index, V010x, V010y, V010z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * (1 - y) * (1 - z), batch_index, V001x, V001y, V001z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * (1 - y) * z, batch_index, V101x, V101y, V101z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * y * (1 - z), batch_index, V011x, V011y, V011z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, (1 - x) * y * z, batch_index, V110x, V110y, V110z, ID, IH, IW, C, BLOCK_SIZE)
    _splat_3d(to_splat, grad_feature_grid, x * y * z, batch_index, V111x, V111y, V111z, ID, IH, IW, C, BLOCK_SIZE)
    return grid_numel


@triton.jit
def _voxel_grid_splat(to_splat, grad_feature_grid, feature_grid_size, batch_index, ix_in, iy_in, iz_in, C: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    feature_grid_offs = tl.zeros((1,), dtype=tl.int32)
    for gi in range(NUM_GRIDS):
        offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        ID = tl.load(feature_grid_size + offs + 1)
        IH = tl.load(feature_grid_size + offs + 2)
        IW = tl.load(feature_grid_size + offs + 3)
        ID_ = tl.sum(ID, axis=0) // BLOCK_SIZE
        IH_ = tl.sum(IH, axis=0) // BLOCK_SIZE
        IW_ = tl.sum(IW, axis=0) // BLOCK_SIZE
        voxel_grid = (ID_ - 1) * (IH_ - 1) * (IW_ - 1)
        if mask_out_of_bounds_samples:
            in_bounds_mask = is_in_bounds(ix_in, iy_in, iz_in, C, BLOCK_SIZE)
            if C == 1:
                in_bounds_mask = in_bounds_mask[:, None]
            to_splat = to_splat * in_bounds_mask
        else:
            to_splat = to_splat
        if voxel_grid > 0:
            grid_numel = _voxel_grid_splat_one(gi, to_splat, grad_feature_grid + feature_grid_offs, feature_grid_size, batch_index, ix_in, iy_in, iz_in, IH, IW, ID, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        elif ID_ == 1:
            grid_numel = _plane_grid_splat_one(gi, to_splat, grad_feature_grid + feature_grid_offs, feature_grid_size, batch_index, ix_in, iy_in, IH, IW, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        elif IH_ == 1:
            grid_numel = _plane_grid_splat_one(gi, to_splat, grad_feature_grid + feature_grid_offs, feature_grid_size, batch_index, ix_in, iz_in, ID, IW, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        else:
            grid_numel = _plane_grid_splat_one(gi, to_splat, grad_feature_grid + feature_grid_offs, feature_grid_size, batch_index, iy_in, iz_in, ID, IH, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        feature_grid_offs += grid_numel


@triton.jit
def _sample_3d(image, w, batch_index, ix, iy, iz, ID, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1)
    iz_ = tl.minimum(tl.maximum(iz, 0.0), ID - 1)
    image_offs = image + batch_index * ID * IW * IH * C + iz_ * IW * IH * C + iy_ * IW * C + ix_ * C
    mask_w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW) * (iz < ID) * (iz >= 0))
    if C == 1:
        val = tl.view(tl.load(image_offs), (BLOCK_SIZE,))
        out = tl.view(val * mask_w, (BLOCK_SIZE,))
        return out
    else:
        val = tl.view(tl.load(image_offs[:, None] + Coffs[None, :]), (BLOCK_SIZE, C))
        mask_w_bcast = tl.view(mask_w[:, None], (BLOCK_SIZE, 1))
        return val * mask_w_bcast


@triton.jit
def _sample_2d(image, w, batch_index, ix, iy, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    Coffs = tl.arange(0, C)
    ix_ = tl.minimum(tl.maximum(ix, 0.0), IW - 1)
    iy_ = tl.minimum(tl.maximum(iy, 0.0), IH - 1)
    image_offs = image + batch_index * IW * IH * C + iy_ * IW * C + ix_ * C
    mask_w = w * ((iy >= 0) * (iy < IH) * (ix >= 0) * (ix < IW))
    if C == 1:
        val = tl.view(tl.load(image_offs), (BLOCK_SIZE,))
        out = tl.view(val * mask_w, (BLOCK_SIZE,))
        return out
    else:
        val = tl.view(tl.load(image_offs[:, None] + Coffs[None, :]), (BLOCK_SIZE, C))
        mask_w_bcast = tl.view(mask_w[:, None], (BLOCK_SIZE, 1))
        return val * mask_w_bcast


@triton.jit
def voxel_grid_sample_one_nearest(gi, feature_grid, feature_grid_size, batch_index, ix_in, iy_in, iz_in, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    ID = tl.load(feature_grid_size + offs + 1)
    IH = tl.load(feature_grid_size + offs + 2)
    IW = tl.load(feature_grid_size + offs + 3)
    ix11 = (ix_in + 1) / 2 * IW - 0.5
    iy11 = (iy_in + 1) / 2 * IH - 0.5
    iz11 = (iz_in + 1) / 2 * ID - 0.5
    ix = ix11 * (ID > 1)
    iy = iy11 * (IH > 1)
    iz = iz11 * (IW > 1)
    unit_weight = ix * 0.0 + 1.0
    ix = _round(ix)
    iy = _round(iy)
    iz = _round(iz)
    sampled = _sample_3d(feature_grid, unit_weight, batch_index, ix, iy, iz, ID, IH, IW, C, BLOCK_SIZE)
    if mask_out_of_bounds_samples:
        in_bounds_mask = is_in_bounds(ix_in, iy_in, iz_in, C, BLOCK_SIZE)
        sampled *= in_bounds_mask
    return sampled


@triton.jit
def _voxel_grid_sample_one(gi, feature_grid, feature_grid_size, batch_index, ix_in, iy_in, iz_in, ID, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    ix, iy, iz, ix0, iy0, iz0, grid_numel = _get_voxel_grid_sample_info(gi, ix_in, iy_in, iz_in, ID, IH, IW, feature_grid_size, C, BLOCK_SIZE)
    V000x, V000y, V000z, V100x, V100y, V100z, V010x, V010y, V010z, V001x, V001y, V001z, V101x, V101y, V101z, V011x, V011y, V011z, V110x, V110y, V110z, V111x, V111y, V111z, x, y, z = _get_voxel_grid_sample_locs_weights(ix, iy, iz, ix0, iy0, iz0)
    sampled = _sample_3d(feature_grid, (1 - x) * (1 - y) * (1 - z), batch_index, V000x, V000y, V000z, ID, IH, IW, C, BLOCK_SIZE) + _sample_3d(feature_grid, (1 - x) * (1 - y) * z, batch_index, V100x, V100y, V100z, ID, IH, IW, C, BLOCK_SIZE) + _sample_3d(feature_grid, (1 - x) * y * (1 - z), batch_index, V010x, V010y, V010z, ID, IH, IW, C, BLOCK_SIZE) + _sample_3d(feature_grid, x * (1 - y) * (1 - z), batch_index, V001x, V001y, V001z, ID, IH, IW, C, BLOCK_SIZE) + _sample_3d(feature_grid, x * (1 - y) * z, batch_index, V101x, V101y, V101z, ID, IH, IW, C, BLOCK_SIZE) + _sample_3d(feature_grid, x * y * (1 - z), batch_index, V011x, V011y, V011z, ID, IH, IW, C, BLOCK_SIZE) + _sample_3d(feature_grid, (1 - x) * y * z, batch_index, V110x, V110y, V110z, ID, IH, IW, C, BLOCK_SIZE) + _sample_3d(feature_grid, x * y * z, batch_index, V111x, V111y, V111z, ID, IH, IW, C, BLOCK_SIZE)
    return sampled, grid_numel


@triton.jit
def _plane_grid_sample_one(gi, feature_grid, feature_grid_size, batch_index, ix_in, iy_in, IH, IW, C: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    ix, iy, ix0, iy0, grid_numel = _get_plane_grid_sample_info(gi, ix_in, iy_in, IH, IW, feature_grid_size, C, BLOCK_SIZE)
    V00x, V00y, V10x, V10y, V01x, V01y, V11x, V11y, x, y = _get_plane_grid_sample_locs_weights(ix, iy, ix0, iy0)
    sampled = _sample_2d(feature_grid, (1 - x) * (1 - y), batch_index, V00x, V00y, IH, IW, C, BLOCK_SIZE) + _sample_2d(feature_grid, x * (1 - y), batch_index, V01x, V01y, IH, IW, C, BLOCK_SIZE) + _sample_2d(feature_grid, (1 - x) * y, batch_index, V10x, V10y, IH, IW, C, BLOCK_SIZE) + _sample_2d(feature_grid, x * y, batch_index, V11x, V11y, IH, IW, C, BLOCK_SIZE)
    return sampled, grid_numel


@triton.jit
def _voxel_grid_sample(feature_grid, feature_grid_size, batch_index, ix_in, iy_in, iz_in, C: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    out_val = tl.zeros((BLOCK_SIZE, C), dtype=tl.float32)
    feature_grid_offs = tl.zeros((1,), dtype=tl.int32)
    for gi in range(NUM_GRIDS):
        offs = gi * 5 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        ID = tl.load(feature_grid_size + offs + 1)
        IH = tl.load(feature_grid_size + offs + 2)
        IW = tl.load(feature_grid_size + offs + 3)
        ID_ = tl.sum(ID, axis=0) // BLOCK_SIZE
        IH_ = tl.sum(IH, axis=0) // BLOCK_SIZE
        IW_ = tl.sum(IW, axis=0) // BLOCK_SIZE
        voxel_grid = (ID_ - 1) * (IH_ - 1) * (IW_ - 1)
        if voxel_grid > 0:
            sampled, grid_numel = _voxel_grid_sample_one(gi, feature_grid + feature_grid_offs, feature_grid_size, batch_index, ix_in, iy_in, iz_in, ID, IH, IW, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        elif ID_ == 1:
            sampled, grid_numel = _plane_grid_sample_one(gi, feature_grid + feature_grid_offs, feature_grid_size, batch_index, ix_in, iy_in, IH, IW, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        elif IH_ == 1:
            sampled, grid_numel = _plane_grid_sample_one(gi, feature_grid + feature_grid_offs, feature_grid_size, batch_index, ix_in, iz_in, ID, IW, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        else:
            sampled, grid_numel = _plane_grid_sample_one(gi, feature_grid + feature_grid_offs, feature_grid_size, batch_index, iy_in, iz_in, ID, IH, C, BLOCK_SIZE, mask_out_of_bounds_samples)
        out_val += sampled
        feature_grid_offs += grid_numel
    if mask_out_of_bounds_samples:
        in_bounds_mask = is_in_bounds(ix_in, iy_in, iz_in, C, BLOCK_SIZE)
        out_val *= in_bounds_mask
    return out_val


@triton.jit
def sample_grid_rep(feature_grid, feature_grid_sizes, grid_idx, sample_x, sample_y, sample_z, C: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    vec = _voxel_grid_sample(feature_grid, feature_grid_sizes, grid_idx, sample_x, sample_y, sample_z, C, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)
    return vec


@triton.jit
def splat_grid_rep(feature_grid, grad_image, feature_grid_sizes, grid_idx, sample_x, sample_y, sample_z, C: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr'):
    _voxel_grid_splat(feature_grid, grad_image, feature_grid_sizes, grid_idx, sample_x, sample_y, sample_z, C, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)


INT32_PRIME = 105097564


@triton.jit
def hash(x):
    x = (x >> 16 ^ x) * 73244475
    x = (x >> 16 ^ x) * 73244475
    x = x >> 16 ^ x
    return x


MAX_INT_32_F = 2147483647.0


MAX_UINT_32_F = 4294967295.0


MAX_UINT_32_F_EPS = 3.0


@triton.jit
def int32_to_float01(x):
    x_01 = (x + MAX_INT_32_F + MAX_UINT_32_F_EPS) / (MAX_UINT_32_F + MAX_UINT_32_F_EPS)
    return x_01


@triton.jit
def pair_hash(x, h):
    h = h ^ x
    h = (h << 24) + h * 403
    return h


@triton.jit
def int_to_randn(x1, x2, seed):
    x_hash_1 = hash(x1)
    x_hash_2 = hash(x2)
    x_hash_1 = pair_hash(pair_hash(INT32_PRIME, seed), x_hash_1)
    x_hash_2 = pair_hash(pair_hash(INT32_PRIME, seed + 1), x_hash_2)
    x_01_1 = int32_to_float01(x_hash_1)
    x_01_2 = int32_to_float01(x_hash_2)
    z = tl.sqrt(-2 * tl.log(x_01_1)) * tl.cos(6.28318530718 * x_01_2)
    return z


@triton.jit
def int_to_randn_kernel(x1, x2, out, N: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', seed: 'tl.constexpr'):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < N
    x1_buffer = tl.load(x1 + offs, mask=offs_mask)
    x2_buffer = tl.load(x2 + offs, mask=offs_mask)
    seed_buffer = tl.full((BLOCK_SIZE,), seed, dtype=tl.int64)
    r = int_to_randn(x1_buffer, x2_buffer, seed_buffer)
    tl.store(out + offs, r, mask=offs_mask)


@triton.jit
def get_sample_randn(pid, step, n_rays, n_steps, BLOCK_SIZE, seed_buffer):
    offs = pid * BLOCK_SIZE * n_steps + 1
    i1 = offs + step + tl.arange(0, BLOCK_SIZE) * n_steps
    i2 = n_rays * n_steps + i1
    return int_to_randn(i1, i2, seed_buffer)


@triton.jit
def _contract_pi_one(x, n):
    x_c = tl.where(n <= 1.0, x, tl.where(tl.abs(tl.abs(x) - n) <= 1e-08, (2 - 1 / tl.abs(x)) * (x / tl.abs(x)), x / n))
    x_c = x_c * 0.5
    return x_c


@triton.jit
def contract_pi(x, y, z):
    n = tl.maximum(tl.maximum(tl.abs(x), tl.abs(y)), tl.abs(z))
    x_c = _contract_pi_one(x, n)
    y_c = _contract_pi_one(y, n)
    z_c = _contract_pi_one(z, n)
    return x_c, y_c, z_c


@triton.jit
def depth_inv_sphere(far, disparity_at_inf, n, step):
    frac_step = (step + 1) / n
    n_disp = (disparity_at_inf - 1) * frac_step + 1
    return far * (1 / n_disp)


@triton.jit
def depth_lin(near, far, n, step):
    frac_step = step / (n - 1)
    return (far - near) * frac_step + near


@triton.jit
def bw_kernel(grad_feature_grid, grad_feature_grid_sizes, directions, origins, grid_idx, near, far, splatting_feature, mask, num_samples: 'tl.constexpr', num_samples_inf: 'tl.constexpr', num_rays: 'tl.constexpr', grid_channel: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', feature_channel: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr', contract_coords: 'tl.constexpr', disparity_at_inf: 'tl.constexpr', grad_splatting_feature):
    tot_num_samples, pid, offs, offs_mask, offs_features, offs_features_mask, center_x, center_y, center_z, ray_x, ray_y, ray_z, near_buffer, far_buffer, grid_idx_buffer, sample_index_buffer, feature_buffer, mask_buffer = fwbw_splatter_init(directions, origins, grid_idx, near, far, splatting_feature, mask, num_samples, num_samples_inf, num_rays, grid_channel, feature_channel, BLOCK_SIZE)
    depth = near_buffer
    grad_splatting_feature_buffer = tl.zeros((BLOCK_SIZE, feature_channel), dtype=tl.float32)
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = depth_inv_sphere(far_buffer, disparity_at_inf, num_samples_inf, step - num_samples)
        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z
        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y, sample_z)
        grad_vec = sample_grid_rep(grad_feature_grid, grad_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y, sample_z, grid_channel, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)
        grad_vec = grad_vec * mask_buffer
        grad_splatting_feature_buffer += grad_vec
    tl.store(grad_splatting_feature + offs_features, grad_splatting_feature_buffer, mask=offs_features_mask)


@triton.jit
def fw_kernel(feature_grid, feature_grid_sizes, directions, origins, grid_idx, near, far, splatting_feature, mask, num_samples: 'tl.constexpr', num_samples_inf: 'tl.constexpr', num_rays: 'tl.constexpr', grid_channel: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', feature_channel: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr', contract_coords: 'tl.constexpr', disparity_at_inf: 'tl.constexpr'):
    tot_num_samples, pid, offs, offs_mask, offs_features, offs_features_mask, center_x, center_y, center_z, ray_x, ray_y, ray_z, near_buffer, far_buffer, grid_idx_buffer, sample_index_buffer, feature_buffer, mask_buffer = fwbw_splatter_init(directions, origins, grid_idx, near, far, splatting_feature, mask, num_samples, num_samples_inf, num_rays, grid_channel, feature_channel, BLOCK_SIZE)
    feature_buffer = feature_buffer * mask_buffer
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = depth_inv_sphere(far_buffer, disparity_at_inf, num_samples_inf, step - num_samples)
        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z
        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y, sample_z)
        splat_grid_rep(feature_buffer, feature_grid, feature_grid_sizes, grid_idx_buffer, sample_x, sample_y, sample_z, grid_channel, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)


@triton.jit
def bw_kernel_wMLP(grad_feature_grid, grad_feature_grid_sizes, feature_grid, feature_grid_sizes, input_feature_grid, input_feature_grid_sizes, directions, origins, grid_idx, near, far, splatting_feature, mask, mlp_params, DIM_HIDDEN: 'tl.constexpr', DIM_IN: 'tl.constexpr', DIM_OUT: 'tl.constexpr', num_samples: 'tl.constexpr', num_samples_inf: 'tl.constexpr', num_rays: 'tl.constexpr', grid_channel: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', feature_channel: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr', contract_coords: 'tl.constexpr', disparity_at_inf: 'tl.constexpr', grad_splatting_feature, grad_mlp_params, grad_input_feature_grid):
    tot_num_samples, pid, offs, offs_mask, offs_features, offs_features_mask, center_x, center_y, center_z, ray_x, ray_y, ray_z, near_buffer, far_buffer, grid_idx_buffer, sample_index_buffer, feature_buffer, mask_buffer = fwbw_splatter_init(directions, origins, grid_idx, near, far, splatting_feature, mask, num_samples, num_samples_inf, num_rays, grid_channel, feature_channel, BLOCK_SIZE)
    depth = near_buffer
    grad_splatting_feature_buffer = tl.zeros((BLOCK_SIZE, feature_channel), dtype=tl.float32)
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = depth_inv_sphere(far_buffer, disparity_at_inf, num_samples_inf, step - num_samples)
        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z
        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y, sample_z)
        prev_vec = sample_grid_rep(input_feature_grid, input_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y, sample_z, feature_channel, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)
        grad_vec = sample_grid_rep(grad_feature_grid, grad_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y, sample_z, grid_channel, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)
        grad_vec = grad_vec * mask_buffer
        fused_feature = feature_buffer + prev_vec
        splat_grid_rep(grad_splatting, grad_input_feature_grid, input_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y, sample_z, feature_channel, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)
        grad_splatting_feature_buffer += grad_splatting
    tl.store(grad_splatting_feature + offs_features, grad_splatting_feature_buffer, mask=offs_features_mask)
    update_mlp_params(grad_mlp_params, DIM_IN, DIM_HIDDEN)


@triton.jit
def fw_kernel_wMLP(feature_grid, feature_grid_sizes, input_feature_grid, input_feature_grid_sizes, directions, origins, grid_idx, near, far, splatting_feature, mask, mlp_params, DIM_HIDDEN: 'tl.constexpr', DIM_IN: 'tl.constexpr', DIM_OUT: 'tl.constexpr', num_samples: 'tl.constexpr', num_samples_inf: 'tl.constexpr', num_rays: 'tl.constexpr', grid_channel: 'tl.constexpr', NUM_GRIDS: 'tl.constexpr', feature_channel: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr', mask_out_of_bounds_samples: 'tl.constexpr', contract_coords: 'tl.constexpr', disparity_at_inf: 'tl.constexpr'):
    tot_num_samples, pid, offs, offs_mask, offs_features, offs_features_mask, center_x, center_y, center_z, ray_x, ray_y, ray_z, near_buffer, far_buffer, grid_idx_buffer, sample_index_buffer, feature_buffer, mask_buffer = fwbw_splatter_init(directions, origins, grid_idx, near, far, splatting_feature, mask, num_samples, num_samples_inf, num_rays, grid_channel, feature_channel, BLOCK_SIZE)
    for step in range(tot_num_samples):
        if step < num_samples:
            depth = depth_lin(near_buffer, far_buffer, num_samples, step)
        else:
            depth = depth_inv_sphere(far_buffer, disparity_at_inf, num_samples_inf, step - num_samples)
        sample_x = center_x + depth * ray_x
        sample_y = center_y + depth * ray_y
        sample_z = center_z + depth * ray_z
        if contract_coords:
            sample_x, sample_y, sample_z = contract_pi(sample_x, sample_y, sample_z)
        prev_vec = sample_grid_rep(input_feature_grid, input_feature_grid_sizes, grid_idx_buffer, sample_x, sample_y, sample_z, feature_channel, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)
        fused_feature = feature_buffer + prev_vec
        fused_feature = fused_feature * mask_buffer
        splat_grid_rep(fused_feature, feature_grid, feature_grid_sizes, grid_idx_buffer, sample_x, sample_y, sample_z, grid_channel, NUM_GRIDS, BLOCK_SIZE, mask_out_of_bounds_samples)

