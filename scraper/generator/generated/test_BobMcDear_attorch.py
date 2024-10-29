import sys
_module = sys.modules[__name__]
del sys
attorch = _module
act_kernels = _module
act_layers = _module
batch_norm_kernels = _module
batch_norm_layer = _module
conv_kernels = _module
conv_layer = _module
cross_entropy_loss_kernels = _module
cross_entropy_loss_layer = _module
dropout_kernels = _module
dropout_layer = _module
glu_kernels = _module
glu_layer = _module
layer_norm_kernels = _module
layer_norm_layer = _module
linear_kernels = _module
linear_layer = _module
math = _module
multi_head_attention_kernels = _module
multi_head_attention_layer = _module
nll_loss_kernels = _module
nll_loss_layer = _module
nn = _module
p_loss_kernels = _module
p_loss_layers = _module
rms_norm_kernels = _module
rms_norm_layer = _module
softmax_kernels = _module
softmax_layers = _module
types = _module
utils = _module
examples = _module
convnext = _module
main = _module
resnet = _module
vit = _module
utils = _module
gpt = _module
tests = _module
test_act_layers = _module
test_batch_norm_layer = _module
test_conv_layer = _module
test_cross_entropy_loss_layer = _module
test_dropout_layer = _module
test_glu_layer = _module
test_layer_norm_layer = _module
test_linear_layer = _module
test_multi_head_attention_layer = _module
test_nll_loss_layer = _module
test_p_loss_layers = _module
test_rms_norm_layer = _module
test_softmax_layers = _module

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


import triton


import triton.language as tl


import warnings


from random import randint


from typing import Optional


from typing import Tuple


import torch


from torch import Tensor


from torch import nn


from torch.amp import custom_bwd


from torch.amp import custom_fwd


from triton import cdiv


from typing import Dict


from triton import next_power_of_2


from typing import Union


from typing import List


from torch import autocast


from triton.testing import do_bench


@triton.jit
def sigmoid(input):
    """
    Applies sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by sigmoid.
    """
    return 1 / (1 + tl.exp(-input))


@triton.jit
def sigmoid_grad(input):
    """
    Calculates the gradient of sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of sigmoid.
    """
    output_sigmoid = sigmoid(input)
    return output_sigmoid * (1 - output_sigmoid)


@triton.jit
def tanh(input):
    """
    Applies tanh to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by tanh.
    """
    return 2 * sigmoid(2 * input) - 1


@triton.jit
def tanh_grad(input):
    """
    Calculates the gradient of tanh.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of tanh.
    """
    output_tanh = tanh(input)
    return 1 - output_tanh * output_tanh


@triton.jit
def relu(input):
    """
    Applies ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU.
    """
    return tl.maximum(0, input)


@triton.jit
def relu_grad(input):
    """
    Calculates the gradient of ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU.
    """
    return tl.where(input <= 0, 0, 1)


@triton.jit
def gelu(input):
    """
    Applies GELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    return cdf * input


@triton.jit
def gelu_grad(input):
    """
    Calculates the gradient of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of GELU.
    """
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    cdf_grad = 0.39894228 * tl.exp(-0.5 * input * input)
    return cdf_grad * input + cdf


@triton.jit
def silu(input):
    """
    Applies SiLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SiLU.
    """
    return input * sigmoid(input)


@triton.jit
def silu_grad(input):
    """
    Calculates the gradient of SiLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SiLU.
    """
    output_sigmoid = sigmoid(input)
    return output_sigmoid * (input * (1 - output_sigmoid) + 1)


@triton.jit
def relu6(input):
    """
    Applies ReLU6 to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by ReLU6.
    """
    return tl.minimum(relu(input), 6)


@triton.jit
def relu6_grad(input):
    """
    Calculates the gradient of ReLU6.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of ReLU6.
    """
    return tl.where((0 < input) & (input < 6), 1, 0)


@triton.jit
def hardsigmoid(input):
    """
    Applies hard sigmoid to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard sigmoid.
    """
    return tl.maximum(0, tl.minimum(1, input / 6 + 0.5))


@triton.jit
def hardsigmoid_grad(input):
    """
    Calculates the gradient of hard sigmoid.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard sigmoid.
    """
    return tl.where((-3 < input) & (input < 3), 1 / 6, 0)


@triton.jit
def hardswish(input):
    """
    Applies hard Swish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by hard Swish.
    """
    return input * relu6(input + 3) / 6


@triton.jit
def hardswish_grad(input):
    """
    Calculates the gradient of hard Swish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of hard Swish.
    """
    return (relu6(input + 3) + input * relu6_grad(input + 3)) / 6


@triton.jit
def selu(input):
    """
    Applies SELU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by SELU.
    """
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * (tl.maximum(0, input) + tl.minimum(0, alpha * (tl.exp(input) - 1)))


@triton.jit
def selu_grad(input):
    """
    Calculates the gradient of SELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of SELU.
    """
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    return scale * tl.where(input <= 0, alpha * tl.exp(input), 1)


@triton.jit
def mish(input):
    """
    Applies Mish to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Input transformed by Mish.
    """
    return input * tanh(tl.log(1 + tl.exp(input)))


@triton.jit
def mish_grad(input):
    """
    Calculates the gradient of Mish.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of Mish.
    """
    exp = tl.exp(input)
    delta = exp * (exp + 2) + 2
    return exp * (exp * (4 * input + 6 + exp * (exp + 4)) + 4 * (input + 1)) / (delta * delta)


@triton.jit
def leaky_relu(input, negative_slope):
    """
    Applies leaky ReLU to the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Input transformed by leaky ReLU.
    """
    return relu(input) + negative_slope * tl.minimum(0, input)


@triton.jit
def leaky_relu_grad(input, negative_slope):
    """
    Calculates the gradient of leaky ReLU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        negative_slope: Slope of the negative component.

    Returns:
        Gradient of leaky ReLU.
    """
    return tl.where(input <= 0, negative_slope, 1)


@triton.jit
def apply_dropout(input, drop_p, seed, offset):
    """
    Randomly zeroes elements in the input.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        offset: Offset to generate the mask for.

    Returns:
        Input with elements randomly zeroed out.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, input / (1 - drop_p))


@triton.jit
def apply_act_func(input, drop_p, seed, offset, param, act_func: 'tl.constexpr', dropout: 'tl.constexpr'):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Input transformed by the desired activation function,
        potentially with fused dropout.
    """
    if act_func == 'sigmoid':
        input = input
        output = sigmoid(input)
    elif act_func == 'tanh':
        input = input
        output = tanh(input)
    elif act_func == 'relu':
        output = relu(input)
    elif act_func == 'gelu':
        input = input
        output = gelu(input)
    elif act_func == 'silu':
        input = input
        output = silu(input)
    elif act_func == 'relu6':
        output = relu6(input)
    elif act_func == 'hardsigmoid':
        output = hardsigmoid(input)
    elif act_func == 'hardswish':
        output = hardswish(input)
    elif act_func == 'selu':
        input = input
        output = selu(input)
    elif act_func == 'mish':
        input = input
        output = mish(input)
    elif act_func == 'leaky_relu':
        output = leaky_relu(input, param)
    if dropout:
        output = apply_dropout(output, drop_p, seed, offset)
    return output


@triton.jit
def apply_dropout_grad(output_grad, drop_p, seed, offset):
    """
    Calculates the input gradient of dropout.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        offset: Offset to generate the mask for.

    Returns:
        Gradient of dropout.
    """
    random = tl.rand(seed, offset)
    return tl.where(random < drop_p, 0, output_grad / (1 - drop_p))


@triton.jit
def apply_act_func_grad(output_grad, input, drop_p, seed, offset, param, act_func: 'tl.constexpr', dropout: 'tl.constexpr'):
    """
    Calculates the gradient of an activation function.

    Args:
        output_grad: Output gradients. The output gradients must be
            loaded and cannot be a pointer.
        input: Input. The input must be loaded and cannot be a pointer.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        offset: Offset to generate the dropout mask for if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.

    Returns:
        Gradient of the desired activation function.
    """
    if act_func == 'sigmoid':
        input = input
        output = sigmoid_grad(input)
    elif act_func == 'tanh':
        input = input
        output = tanh_grad(input)
    elif act_func == 'relu':
        output = relu_grad(input)
    elif act_func == 'gelu':
        input = input
        output = gelu_grad(input)
    elif act_func == 'silu':
        input = input
        output = silu_grad(input)
    elif act_func == 'relu6':
        output = relu6_grad(input)
    elif act_func == 'hardsigmoid':
        output = hardsigmoid_grad(input)
    elif act_func == 'hardswish':
        output = hardswish_grad(input)
    elif act_func == 'selu':
        input = input
        output = selu_grad(input)
    elif act_func == 'mish':
        input = input
        output = mish_grad(input)
    elif act_func == 'leaky_relu':
        output = leaky_relu_grad(input, param)
    if dropout:
        output_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    return output_grad * output


def element_wise_kernel_configs(block_name: 'str'='BLOCK_SIZE') ->List[triton.Config]:
    """
    Returns kernel configurations for element-wise operations.

    Args:
        block_name: Name of block argument rows are distributed over.
    """
    return [triton.Config({block_name: 64}, num_warps=2), triton.Config({block_name: 128}, num_warps=2), triton.Config({block_name: 256}, num_warps=4), triton.Config({block_name: 512}, num_warps=4), triton.Config({block_name: 1024}, num_warps=4)]


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def act_func_forward_kernel(input_pointer, output_pointer, size, drop_p, seed, param, act_func: 'tl.constexpr', dropout: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    Applies an activation function to the input, optionally fusing dropout.

    Args:
        input_pointer: Pointer to the input to transform.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(output_pointer + offset, apply_act_func(input, drop_p, seed, offset, param, act_func, dropout), mask=mask)


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def act_func_backward_kernel(output_grad_pointer, input_pointer, input_grad_pointer, size, drop_p, seed, param, act_func: 'tl.constexpr', dropout: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    Calculates the input gradient of an activation function.

    Args:
        output_grad_pointer: Pointer to the activation's output gradients.
            The output gradients must be of shape [size].
        input_pointer: Pointer to the activation's input.
            The input must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element if dropout is True.
        seed: Seed for generating the dropout mask if dropout is True.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function whose gradient is calculated.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        dropout: Flag for performing dropout on the activation output.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input = tl.load(input_pointer + offset, mask=mask)
    tl.store(input_grad_pointer + offset, apply_act_func_grad(output_grad, input, drop_p, seed, offset, param, act_func, dropout), mask=mask)


def BLOCK_SIZE_SPATIAL_heuristic(args: 'Dict') ->int:
    """
    Approximates an appropriate spatial block size for batch normalization
    using a heuristic.

    Args:
        args: Arguments to batch normalization kernel.

    Returns:
        Appropriate spatial block size.
    """
    BLOCK_SIZE_BATCH = next_power_of_2(args['batch_dim'])
    BLOCK_SIZE_SPATIAL = next_power_of_2(args['spatial_dim'])
    return min(BLOCK_SIZE_SPATIAL, max(1, 2 ** 14 // BLOCK_SIZE_BATCH))


def warps_kernel_configs() ->List[triton.Config]:
    """
    Returns kernel configurations with all possible number of warps.
    """
    return [triton.Config({}, num_warps=2 ** i) for i in range(6)]


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'spatial_dim'], restore_value=['running_mean_pointer', 'running_var_pointer'])
@triton.heuristics({'BLOCK_SIZE_BATCH': lambda args: next_power_of_2(args['batch_dim']), 'BLOCK_SIZE_SPATIAL': BLOCK_SIZE_SPATIAL_heuristic})
@triton.jit
def batch_norm_forward_kernel(input_pointer, weight_pointer, bias_pointer, mean_pointer, inv_std_pointer, pre_act_add_pointer, pre_act_pointer, output_pointer, running_mean_pointer, running_var_pointer, batch_dim, spatial_dim, input_batch_stride, input_feat_stride, input_spatial_stride, pre_act_add_batch_stride, pre_act_add_feat_stride, pre_act_add_spatial_stride, pre_act_batch_stride, pre_act_feat_stride, pre_act_spatial_stride, output_batch_stride, output_feat_stride, output_spatial_stride, momentum, eps, param, affine: 'tl.constexpr', save_stats: 'tl.constexpr', track_running_stats: 'tl.constexpr', is_train: 'tl.constexpr', add_pre_act: 'tl.constexpr', act_func: 'tl.constexpr', save_pre_act: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_SPATIAL: 'tl.constexpr'):
    """
    Batch-normalizes the input, optionally adding a residual and fusing an activation function.

    Args:
        input_pointer: Pointer to the input to layer-normalize.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        weight_pointer: Pointer to optional weights for affine transform.
            The weights, if provided, must be of shape [feat_dim].
        bias_pointer: Pointer to an optional bias vector for affine transform.
            The bias vector, if provided, must be of shape [feat_dim].
        mean_pointer: Pointer to an optional container the input's mean
            is written to if save_stats is True.
            The container, if provided, must be of shape [feat_dim].
        inv_std_pointer: Pointer to an optional container the input's inverse
            standard deviation is written to if save_stats is True.
            The container, if provided, must be of shape [feat_dim].
        pre_act_add_pointer: Pointer to an optional residual added to the pre-activation result.
            The residual, if provided, must be of shape [batch_dim, feat_dim, spatial_dim].
        pre_act_pointer: Pointer to an optional container the pre-activation input
            is written to if act_func is not None and save_pre_act is True.
            The container, if provided, must be of shape [batch_dim, feat_dim, spatial_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim].
        running_mean_pointer: Pointer to an optional container the input's running
            mean is written to if track_running_stats and is_train are True.
            The container, if provided, must be of shape [feat_dim].
        running_var_pointer: Pointer to an optional container the input's running
            variance is written to if track_running_stats and is_train are True.
            The container, if provided, must be of shape [feat_dim].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        pre_act_add_batch_stride: Stride necessary to jump one element along the
            residual's batch dimension.
        pre_act_add_out_feat_stride: Stride necessary to jump one element along the
            residual's feature dimension.
        pre_act_add_spatial_stride: Stride necessary to jump one element along the
            residual's spatial dimension.
        pre_act_batch_stride: Stride necessary to jump one element along the
            pre-activation input container's batch dimension.
        pre_act_out_feat_stride: Stride necessary to jump one element along the
            pre-activation input container's feature dimension.
        pre_act_spatial_stride: Stride necessary to jump one element along the
            pre-activation input container's spatial dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        output_spatial_stride: Stride necessary to jump one element along the
            output container's spatial dimension.
        momentum: Momentum for the running mean and variance.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        param: Parameter in the case of parameterized activation functions.
        affine: Flag for performing an affine transformation on the normalized output.
        save_stats: Flag for saving the mean and standard deviation.
        track_running_stats: Flag for tracking running mean and variance if
            is_train is also True.
        is_train: Flag indicating if the model is in training mode.
        add_pre_act: Flag for adding the residual to the pre-activation result.
        act_func: Name of activation function to apply, with None for identity.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        save_pre_act: Flag for saving the pre-activation input.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    feat_pid = tl.program_id(axis=0)
    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim
    if is_train or not track_running_stats:
        count = 0
        mean = 0.0
        var = 0.0
        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
            spatial_mask = spatial_offset < spatial_dim
            curr_input_pointer = input_pointer + input_feat_stride * feat_pid + input_batch_stride * batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :]
            curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
            spatial_count = min(BLOCK_SIZE_SPATIAL, spatial_dim - block_ind * BLOCK_SIZE_SPATIAL)
            curr_count = spatial_count * batch_dim
            count += curr_count
            prev_mean = mean
            mean += (tl.sum(curr_input) - curr_count * mean) / count
            deltas = tl.where(batch_mask[:, None] & spatial_mask[None, :], (curr_input - mean) * (curr_input - prev_mean), 0.0)
            var += tl.sum(deltas)
        var /= count
        inv_std = tl.rsqrt(var + eps)
        if save_stats:
            tl.store(feat_pid + mean_pointer, mean)
            tl.store(feat_pid + inv_std_pointer, inv_std)
        if track_running_stats:
            running_mean_pointer += feat_pid
            running_var_pointer += feat_pid
            running_mean = tl.load(running_mean_pointer)
            running_var = tl.load(running_var_pointer)
            n = batch_dim * spatial_dim
            tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
            tl.store(running_var_pointer, (1 - momentum) * running_var + momentum * var * n / (n - 1))
    else:
        mean = tl.load(feat_pid + running_mean_pointer)
        inv_std = tl.rsqrt(tl.load(feat_pid + running_var_pointer) + eps)
    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        bias = tl.load(feat_pid + bias_pointer)
    else:
        weight = 1.0
        bias = 0.0
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_input_pointer = input_pointer + input_feat_stride * feat_pid + input_batch_stride * batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :]
        curr_output_pointer = output_pointer + output_feat_stride * feat_pid + output_batch_stride * batch_offset[:, None] + output_spatial_stride * spatial_offset[None, :]
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
        output = weight * (curr_input - mean) * inv_std + bias
        if add_pre_act:
            curr_pre_act_add_pointer = pre_act_add_pointer + pre_act_add_feat_stride * feat_pid + pre_act_add_batch_stride * batch_offset[:, None] + pre_act_add_spatial_stride * spatial_offset[None, :]
            curr_pre_act_add = tl.load(curr_pre_act_add_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
            output += curr_pre_act_add
        if act_func is not None:
            if save_pre_act:
                curr_pre_act_pointer = pre_act_pointer + pre_act_feat_stride * feat_pid + pre_act_batch_stride * batch_offset[:, None] + pre_act_spatial_stride * spatial_offset[None, :]
                tl.store(curr_pre_act_pointer, output, mask=batch_mask[:, None] & spatial_mask[None, :])
            output = apply_act_func(output, None, None, None, param, act_func, False)
        tl.store(curr_output_pointer, output, mask=batch_mask[:, None] & spatial_mask[None, :])


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': lambda args: next_power_of_2(args['batch_dim']), 'BLOCK_SIZE_SPATIAL': BLOCK_SIZE_SPATIAL_heuristic})
@triton.jit
def batch_norm_backward_kernel(output_grad_pointer, input_pointer, mean_pointer, inv_std_pointer, weight_pointer, input_grad_pointer, weight_grad_pointer, bias_grad_pointer, batch_dim, spatial_dim, output_grad_batch_stride, output_grad_feat_stride, output_grad_spatial_stride, input_batch_stride, input_feat_stride, input_spatial_stride, input_grad_batch_stride, input_grad_feat_stride, input_grad_spatial_stride, affine: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_SPATIAL: 'tl.constexpr'):
    """
    Calculates the input gradient of batch normalization.

    Args:
        output_grad_pointer: Pointer to layer normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim, spatial_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        mean_pointer: Pointer to the input's mean.
            The mean should be of shape [feat_dim].
        inv_std_pointer: Pointer to the input's inverse standard deviation.
            The inverse standard deviation should be of shape [feat_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim].
        weight_grad_pointer: Pointer to an optional container the weights' gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's gradients
            are written to if scale_by_weight is True.
            The container, if provided, must be of shape [feat_dim].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        output_grad_spatial_stride: Stride necessary to jump one element along the
            output gradients' spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        affine: Flag for performing an affine transformation on the normalized output.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    feat_pid = tl.program_id(axis=0)
    batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
    batch_mask = batch_offset < batch_dim
    mean = tl.load(feat_pid + mean_pointer)
    inv_std = tl.load(feat_pid + inv_std_pointer)
    term1 = 0.0
    term2 = 0.0
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_output_grad_pointer = output_grad_pointer + output_grad_feat_stride * feat_pid + output_grad_batch_stride * batch_offset[:, None] + output_grad_spatial_stride * spatial_offset[None, :]
        curr_input_pointer = input_pointer + input_feat_stride * feat_pid + input_batch_stride * batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :]
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
        term1 += tl.sum(curr_pre_lin * curr_output_grad)
        term2 += tl.sum(curr_output_grad)
    if affine:
        weight = tl.load(feat_pid + weight_pointer)
        weight_grad = 0.0
        bias_grad = 0.0
    else:
        weight = 1.0
    count = batch_dim * spatial_dim
    term1 *= weight / count
    term2 *= weight / count
    for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
        spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
        spatial_mask = spatial_offset < spatial_dim
        curr_output_grad_pointer = output_grad_pointer + output_grad_feat_stride * feat_pid + output_grad_batch_stride * batch_offset[:, None] + output_grad_spatial_stride * spatial_offset[None, :]
        curr_input_pointer = input_pointer + input_feat_stride * feat_pid + input_batch_stride * batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :]
        curr_input_grad_pointer = input_grad_pointer + input_grad_feat_stride * feat_pid + input_grad_batch_stride * batch_offset[:, None] + input_grad_spatial_stride * spatial_offset[None, :]
        curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
        curr_pre_lin = (curr_input - mean) * inv_std
        curr_output_grad = tl.load(curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
        curr_input_grad = inv_std * (weight * curr_output_grad - (term1 * curr_pre_lin + term2))
        tl.store(curr_input_grad_pointer, curr_input_grad, mask=batch_mask[:, None] & spatial_mask[None, :])
        if affine:
            weight_grad += tl.sum(curr_pre_lin * curr_output_grad)
            bias_grad += tl.sum(curr_output_grad)
    if affine:
        tl.store(feat_pid + weight_grad_pointer, weight_grad)
        tl.store(feat_pid + bias_grad_pointer, bias_grad)


def allow_tf32() ->bool:
    """
    Returns whether the current GPU architecture supports TF32.
    """
    return torch.cuda.get_device_capability()[0] >= 8


def get_n_stages(n_stages: 'int'=2) ->int:
    """
    Receives number of stages for software pipelining and returns it as-is
    if the GPU architecture is Ampere or newer and 2 otherwise.
    """
    return 2 if torch.cuda.get_device_capability()[0] < 8 else n_stages


def conv2d_forward_config(BLOCK_SIZE_BATCH_HEIGHT_WIDTH: 'int', BLOCK_SIZE_IN_FEAT: 'int', BLOCK_SIZE_OUT_FEAT: 'int', n_warps: 'int'=4, n_stages: 'int'=2) ->triton.Config:
    """
    Creates a triton.Config object for conv2d_forward_kernel
    given meta-parameters for auto-tuning.

    Args:
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH: Block size across the batch, height, and
            width dimensions.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
        n_warps: Number of warps to use for the kernel when compiled for GPUs.
        n_stages: Number of stages the compiler uses to software-pipeline.

    Returns:
        Kernel configuration.
    """
    return triton.Config({'BLOCK_SIZE_BATCH_HEIGHT_WIDTH': BLOCK_SIZE_BATCH_HEIGHT_WIDTH, 'BLOCK_SIZE_IN_FEAT': BLOCK_SIZE_IN_FEAT, 'BLOCK_SIZE_OUT_FEAT': BLOCK_SIZE_OUT_FEAT}, num_warps=n_warps, num_stages=get_n_stages(n_stages))


def BLOCK_SIZE_BATCH_heuristic(args: 'Dict') ->int:
    """
    Approximates an appropriate batch block size for softmax using a heuristic.

    Args:
        args: Arguments to softmax kernel.

    Returns:
        Appropriate batch block size.
    """
    return min(max(1, next_power_of_2(args['batch_dim'] // 2 ** 10)), 128) if args['feat_dim'] < 64 else 1


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def cross_entropy_loss_forward_kernel(input_pointer, target_pointer, weight_pointer, sum_weights_pointer, output_pointer, batch_dim, feat_dim, input_batch_stride, input_feat_stride, weighted: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Measures the mean cross entropy loss between the input and target,
    with optional reweighing of each class.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to a container the sum of the class weights is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        output_pointer: Pointer to a container the loss is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    target = tl.load(target_pointer + batch_offset, mask=batch_mask)
    pred_pointer = input_pointer + input_feat_stride * target + input_batch_stride * batch_offset
    input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :], other=-float('inf'))
    pred = tl.load(pred_pointer, mask=batch_mask)
    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask)
        loss *= weight
        tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
    else:
        loss /= batch_dim
    tl.store(output_pointer + batch_pid, tl.sum(loss))


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def cross_entropy_loss_backward_kernel(output_grad_pointer, target_pointer, input_pointer, weight_pointer, sum_weights_pointer, input_grad_pointer, batch_dim, feat_dim, input_batch_stride, input_feat_stride, input_grad_batch_stride, input_grad_feat_stride, weighted: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Calculates the input gradient of cross entropy loss.

    Args:
        output_grad_pointer: Pointer to the loss's output gradients.
            The output gradient must be a scalar.
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to the sum of the class weights if the classes were weighed.
            The sum of weights must be a scalar.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    input_grad_pointer += input_grad_batch_stride * batch_offset[:, None] + input_grad_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :], other=-float('inf'))
    input -= tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    softmax = numerator / tl.sum(numerator, axis=1)[:, None]
    output_grad = tl.load(output_grad_pointer)
    target = tl.load(target_pointer + batch_offset, mask=batch_mask)
    broadcasted_feat_offset = tl.broadcast_to(feat_offset[None, :], (BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT))
    broadcasted_target = tl.broadcast_to(target[:, None], (BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT))
    input_grad = output_grad * (softmax - (broadcasted_feat_offset == broadcasted_target))
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask)
        sum_weights = tl.load(sum_weights_pointer)
        input_grad *= weight[:, None] / sum_weights
    else:
        input_grad /= batch_dim
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] & feat_mask[None, :])


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_forward_kernel(input_pointer, output_pointer, size, drop_p, seed, BLOCK_SIZE: 'tl.constexpr'):
    """
    Randomly zeroes elements in the input.

    Args:
        input_pointer: Pointer to the input to perform dropout on.
            The input must be of shape [size].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    output = apply_dropout(input, drop_p, seed, offset)
    tl.store(output_pointer + offset, output, mask=mask)


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def dropout_backward_kernel(output_grad_pointer, input_grad_pointer, size, drop_p, seed, BLOCK_SIZE: 'tl.constexpr'):
    """
    Calculates the input gradient of dropout.

    Args:
        output_grad_pointer: Pointer to dropout's output gradients.
            The output gradients must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input.
        drop_p: Probability of dropping an element used in dropout.
        seed: Seed for generating the dropout mask.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input_grad = apply_dropout_grad(output_grad, drop_p, seed, offset)
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def glu_forward_kernel(input1_pointer, input2_pointer, output_pointer, size, param, act_func: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.

    Args:
        input1_pointer: Pointer to the first half of the input to gate.
            The first half must be contiguous and contain size elements.
        input2_pointer: Pointer to the second half of the input to gate.
            The second half must be contiguous and contain size elements.
        output_pointer: Pointer to a container the result is written to.
            The container must be contiguous and contain size elements.
        size: Number of elements in each half of the input.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input1 = tl.load(input1_pointer + offset, mask=mask)
    input2 = tl.load(input2_pointer + offset, mask=mask)
    output = input1 * apply_act_func(input2, None, None, None, param, act_func, False)
    tl.store(output_pointer + offset, output, mask=mask)


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def glu_backward_kernel(output_grad_pointer, input1_pointer, input2_pointer, input1_grad_pointer, input2_grad_pointer, size, param, act_func: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    Calculates the input gradient of the gated linear unit.

    Args:
        output_grad_pointer: Pointer to the unit's output gradients.
            The output gradients must be contiguous and contain size elements.
        input1_pointer: Pointer to the first half of the input that was gated.
            The first half must be contiguous and contain size elements.
        input2_pointer: Pointer to the second half of the input that was gated.
            The second half must be contiguous and contain size elements.
        input1_grad_pointer: Pointer to a container the first half's gradients are written to.
            The container must be contiguous and contain size elements.
        input2_grad_pointer: Pointer to a container the second half's gradients are written to.
            The container must be contiguous and contain size elements.
        size: Number of elements in each half of the input.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    input1 = tl.load(input1_pointer + offset, mask=mask)
    input2 = tl.load(input2_pointer + offset, mask=mask)
    input1_grad = output_grad * apply_act_func(input2, None, None, None, param, act_func, False)
    input2_grad = output_grad * input1 * apply_act_func_grad(1, input2, None, None, None, param, act_func, False)
    tl.store(input1_grad_pointer + offset, input1_grad, mask=mask)
    tl.store(input2_grad_pointer + offset, input2_grad, mask=mask)


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def layer_norm_forward_kernel(input_pointer, weight_pointer, bias_pointer, mean_pointer, inv_std_pointer, output_pointer, batch_dim, feat_dim, input_batch_stride, input_feat_stride, output_batch_stride, output_feat_stride, eps, scale_by_weight: 'tl.constexpr', add_bias: 'tl.constexpr', save_stats: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Layer-normalizes the input.

    Args:
        input_pointer: Pointer to the input to layer-normalize.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to optional weights for affine transform.
            The weights, if provided, must be of shape [feat_dim].
        bias_pointer: Pointer to an optional bias vector for affine transform.
            The bias vector, if provided, must be of shape [feat_dim].
        mean_pointer: Pointer to an optional container the input's mean
            is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        inv_std_pointer: Pointer to an optional container the input's inverse
            standard deviation is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        scale_by_weight: Flag for scaling the normalized output by weights.
        add_bias: Flag for adding a bias vector to the normalized output
            if scale_by_weight is True.
        save_stats: Flag for saving the mean and standard deviation.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    output_pointer += output_batch_stride * batch_offset[:, None] + output_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    mean = tl.sum(input, axis=1) / feat_dim
    diff = tl.where(feat_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / feat_dim + eps)
    if save_stats:
        tl.store(mean_pointer + batch_offset, mean, mask=batch_mask)
        tl.store(inv_std_pointer + batch_offset, inv_std, mask=batch_mask)
    output = diff * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        output *= weight
        if add_bias:
            bias = tl.load(bias_pointer + feat_offset, mask=feat_mask)
            output += bias
    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[None, :])


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def layer_norm_backward_kernel(output_grad_pointer, input_pointer, mean_pointer, inv_std_pointer, weight_pointer, input_grad_pointer, weight_grad_pointer, bias_grad_pointer, batch_dim, feat_dim, output_grad_batch_stride, output_grad_feat_stride, input_batch_stride, input_feat_stride, input_grad_batch_stride, input_grad_feat_stride, weight_grad_batch_stride, weight_grad_feat_stride, bias_grad_batch_stride, bias_grad_feat_stride, scale_by_weight: 'tl.constexpr', add_bias: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Calculates the input gradient of layer normalization.

    Args:
        output_grad_pointer: Pointer to layer normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        mean_pointer: Pointer to the input's mean.
            The mean should be of shape [batch_dim].
        inv_std_pointer: Pointer to the input's inverse standard deviation.
            The inverse standard deviation should be of shape [batch_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        weight_grad_pointer: Pointer to an optional container the weights' row-wise gradients
            are written to if scale_by_weight is True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's row-wise gradients
            are written to if scale_by_weight and add_bias are True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        weight_grad_batch_stride: Stride necessary to jump one element along the
            weight gradient container's batch dimension.
        weight_grad_feat_stride: Stride necessary to jump one element along the
            weight gradient container's feature dimension.
        bias_grad_batch_stride: Stride necessary to jump one element along the
            weight gradient container's batch dimension.
        bias_grad_feat_stride: Stride necessary to jump one element along the
            weight gradient container's feature dimension.
        scale_by_weight: Flag for scaling the normalized output by weights.
        add_bias: Flag for adding a bias vector to the normalized output
            if scale_by_weight is True.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    output_grad_pointer += output_grad_batch_stride * batch_offset[:, None] + output_grad_feat_stride * feat_offset[None, :]
    input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    input_grad_pointer += input_grad_batch_stride * batch_offset[:, None] + input_grad_feat_stride * feat_offset[None, :]
    output_grad = tl.load(output_grad_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    mean = tl.load(mean_pointer + batch_offset, mask=batch_mask)
    inv_std = tl.load(inv_std_pointer + batch_offset, mask=batch_mask)
    pre_lin = (input - mean[:, None]) * inv_std[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        weight_output_grad_prod = weight * output_grad
    else:
        weight_output_grad_prod = output_grad
    term1 = tl.sum(pre_lin * weight_output_grad_prod, axis=1) / feat_dim
    term1 = pre_lin * term1[:, None]
    term2 = tl.sum(weight_output_grad_prod, axis=1) / feat_dim
    input_grad = inv_std[:, None] * (weight_output_grad_prod - (term1 + term2[:, None]))
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] & feat_mask[None, :])
    if scale_by_weight:
        weight_grad_pointer += weight_grad_batch_stride * batch_pid + weight_grad_feat_stride * feat_offset
        tl.store(weight_grad_pointer, tl.sum(output_grad * pre_lin, axis=0), mask=feat_mask)
        if add_bias:
            bias_grad_pointer += bias_grad_batch_stride * batch_pid + bias_grad_feat_stride * feat_offset
            tl.store(bias_grad_pointer, tl.sum(output_grad, axis=0), mask=feat_mask)


def linear_forward_config(BLOCK_SIZE_BATCH: 'int', BLOCK_SIZE_IN_FEAT: 'int', BLOCK_SIZE_OUT_FEAT: 'int', GROUP_SIZE_BATCH: 'int'=8, n_warps: 'int'=4, n_stages: 'int'=2) ->triton.Config:
    """
    Creates a triton.Config object for linear_forward_kernel
    given meta-parameters for auto-tuning.

    Args:
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_IN_FEAT: Block size across the input feature dimension.
        BLOCK_SIZE_OUT_FEAT: Block size across the output feature dimension.
        GROUP_SIZE_BATCH: Group size across the batch dimension.
        n_warps: Number of warps to use for the kernel when compiled for GPUs.
        n_stages: Number of stages the compiler uses to software-pipeline.
            On GPU architectures older than Ampere, this is fixed at 2.

    Returns:
        Kernel configuration.
    """
    return triton.Config({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH, 'BLOCK_SIZE_IN_FEAT': BLOCK_SIZE_IN_FEAT, 'BLOCK_SIZE_OUT_FEAT': BLOCK_SIZE_OUT_FEAT, 'GROUP_SIZE_BATCH': GROUP_SIZE_BATCH}, num_warps=n_warps, num_stages=get_n_stages(n_stages))


@triton.jit
def accum_linear(accum, input1, input2, fp16: 'tl.constexpr', tf32: 'tl.constexpr'):
    """
    Accumulates matrix multiplications of input tensors for linear functions.

    Args:
        accum: Accumulator holding aggregation of matrix multiplications.
            The accumulator must be of shape [BLOCK_SIZE1, BLOCK_SIZE3].
        input1: First operand of matrix multiplication.
            The operand must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        input2: Second operand of matrix multiplication.
            The operand must be of shape [BLOCK_SIZE2, BLOCK_SIZE3].
        fp16: Flag for converting operands to FP16.
        tf32: Flag for performing matrix multiplication in TF32.

    Returns:
        Accumulator with the result of the new matrix multiplication added to it.
    """
    if fp16:
        input1 = input1
        input2 = input2
    return accum + tl.dot(input1, input2, allow_tf32=tf32)


@triton.jit
def glu(input1, input2, param, act_func: 'tl.constexpr'):
    """
    Applies the gated linear unit with an arbitrary activation function
    to the input.

    Args:
        input1: First half of input to gate.
            The first half must be of the same shape as the second half.
        input2: Second half of input to gate.
            The second half must be of the same shape as the first half.
        param: Parameter in the case of parameterized activation functions.
        act_func: Name of activation function to apply.
            Options are 'sigmoid', 'tanh', 'relu', 'gelu', 'silu',
            'relu6', 'hardsigmoid', 'hardswish', 'selu', 'mish', and 'leaky_relu'.
        param: Parameter in the case of parameterized activation functions.

    Args:
        Input transformed by the gated linear unit
        with an arbitrary activation function.
    """
    return input1 * apply_act_func(input2, None, None, None, param, act_func, False)


@triton.jit
def softmax(input, log: 'tl.constexpr'):
    """
    Normalizes the input using softmax along the last dimension.

    Args:
        input: Input to normalize.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        log: Flag for indicating if the log of softmax should be taken.

    Returns:
        Input normalized by softmax.
    """
    input = input
    input = input - tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]
    if log:
        output = input - tl.log(denominator)
    else:
        output = numerator / denominator
    return output


@triton.jit
def calc_mean_and_inv_std(input, last_dim, eps, last_dim_mask: 'tl.constexpr'):
    """
    Calculates the mean and inverse standard deviation of the input
    along the last dimension.

    Args:
        input: Input whose mean and inverse standard deviation are calculated.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        last_dim: Size of the last dimension of input.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        last_dim_mask: Mask for the last dimension indicating
            which elements should be included in the calculations.
            The mask must be of shape [BLOCK_SIZE2].

    Returns:
        Mean and inverse standard deviation of the input.
    """
    input = input
    mean = tl.sum(input, axis=1) / last_dim
    diff = tl.where(last_dim_mask[None, :], input - mean[:, None], 0)
    inv_std = tl.rsqrt(tl.sum(diff * diff, axis=1) / last_dim + eps)
    return mean, inv_std


@triton.jit
def update_welford(input, prev_count, prev_mean, prev_var, curr_count, mask: 'tl.constexpr'):
    """
    Updates count, mean, and variance (M2) statistics for Welford's algorithm.

    Args:
        input: Input used to update statistics.
            The input must be of the same shape as the mask.
        prev_count: Previous count statistic to update.
        prev_mean: Previous mean statistic to update.
        prev_var: Previous variance (M2) statistic to update.
        curr_count: Count of elements in current input.
        mask: Mask indicating which elements should be included in the calculations.
            The mask must be of the same shape as the input.

    Returns:
        Updated count, mean, and variance (M2) statistics
    """
    input = input
    count = prev_count + curr_count
    mean = (tl.sum(input) - curr_count * prev_mean) / count
    deltas = tl.where(mask, (input - mean) * (input - prev_mean), 0.0)
    var = prev_var + tl.sum(deltas)
    return count, mean, var


@triton.jit
def update_ema(prev_ema, new_val, momentum):
    """
    Updates exponential moving average.

    Args:
        prev_ema: Previous exponential moving average.
        new_val: Value used to update the exponential moving average.
        momentum: Momentum.

    Returns:
        Updated running statistic.
    """
    return (1 - momentum) * prev_ema + momentum * new_val


@triton.jit
def standardize(input, mean, inv_std, weight, bias):
    """
    Standardizes the input given its mean and inverse standard deviation,
    multiplies the result by weights, and adds a bias vector.

    Args:
        input: Input to standardize.
        mean: Mean of input.
        inv_std: Inverse standard deviation of input.
        weight: Weight multiplied by the standardized input.
        bias: Bias added to the result of the weight multiplication.

    Returns:
        Standardized input.
    """
    return weight * inv_std * (input - mean) + bias


@triton.jit
def calc_p_loss(input, target, size, p_loss: 'tl.constexpr', reduction: 'tl.constexpr'):
    """
    Measures the L1 or squared L2 norm of the difference between the input
    and target (i.e., mean absolute error or mean squared error).

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE].
        target: Target.
            The target must be of shape [BLOCK_SIZE].
        size: Number of elements in the input and target.
            This value is used only if reduction is 'mean'.
        p_loss: p-norm used to compute the error.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.

    Returns:
        Error.
    """
    input = input
    target = target
    diff = input - target
    if p_loss == 1:
        error = tl.abs(diff)
    elif p_loss == 2:
        error = diff * diff
    if reduction == 'none':
        output = error
    elif reduction == 'mean':
        output = tl.sum(error) / size
    elif reduction == 'sum':
        output = tl.sum(error)
    return output


@triton.jit
def nll_loss(input, size, reduction: 'tl.constexpr'):
    """
    Measures the negative log likelihood loss given log-probabilities of target class.

    Args:
        input: Input containing predicted log-probabilities corresponding to target class.
            The input can have arbitrary shape.
        size: Number of elements in the input.
            This value is used only if reduction is 'mean'.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.

    Returns:
        Loss.
    """
    input = input
    if reduction == 'none':
        output = -input
    elif reduction == 'mean':
        output = -tl.sum(input) / size
    elif reduction == 'sum':
        output = -tl.sum(input)
    return output


@triton.jit
def cross_entropy_loss(input, pred):
    """
    Measures the per-row cross entropy loss given
    input and predicted logits corresponding to target class.

    Args:
        input: Input.
            The input must be of shape [BLOCK_SIZE1, BLOCK_SIZE2].
        pred: Predicted logits corresponding to target class.
            The predictions must be of shape [BLOCK_SIZE1].

    Returns:
        Loss.
    """
    input = input
    pred = pred
    mx = tl.max(input, axis=1)
    input -= mx[:, None]
    loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx
    return loss


@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, L, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX, Z_H_N_CTX, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', IS_CAUSAL: 'tl.constexpr'):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm
    K_block_ptr = tl.make_block_ptr(base=K, shape=(BLOCK_DMODEL, Z_H_N_CTX), strides=(stride_kk, stride_kn), offsets=(0, vk_offset), block_shape=(BLOCK_DMODEL, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(base=V, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk), offsets=(vk_offset, 0), block_shape=(BLOCK_N, BLOCK_DMODEL), order=(1, 0))
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs)
    q = q * qk_scale
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, float('-inf'))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    O_block_ptr = tl.make_block_ptr(base=Out, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_om, stride_on), offsets=(vk_offset + start_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    tl.store(O_block_ptr, acc)


@triton.jit
def _bwd_preprocess(Out, DO, Delta, BLOCK_M: 'tl.constexpr', D_HEAD: 'tl.constexpr'):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :])
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :])
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO, DQ, DK, DV, L, D, Q_block_ptr, K_block_ptr, V_block_ptr, DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr, stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, Z, H, N_CTX, off_h, off_z, off_hz, start_n, num_block, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', SEQUENCE_PARALLEL: 'tl.constexpr', CAUSAL: 'tl.constexpr', MMA_V3: 'tl.constexpr'):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0
    Q_offset = (off_z * stride_qz + off_h * stride_qh) // stride_qm
    DQ_offset = off_z * stride_qz + off_h * stride_qh
    K_offset = (off_z * stride_kz + off_h * stride_kh) // stride_kn
    V_offset = (off_z * stride_vz + off_h * stride_vh) // stride_vn
    if SEQUENCE_PARALLEL:
        DQ_offset += stride_dqa * start_n
    DQ_offset = DQ_offset // stride_qm
    Q_block_ptr = tl.advance(Q_block_ptr, (lo + Q_offset, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_M + V_offset, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo + Q_offset, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo + DQ_offset, 0))
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_M + V_offset, 0))
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        q = tl.load(Q_block_ptr)
        if CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= offs_n[None, :], float(0.0), float('-inf'))
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i[:, None])
        do = tl.load(DO_block_ptr)
        dv += tl.dot(tl.trans(p), do)
        Di = tl.load(D_ptrs + offs_m_curr)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None]) * sm_scale
        dk += tl.dot(tl.trans(ds), q)
        if not SEQUENCE_PARALLEL:
            dq = tl.load(DQ_block_ptr)
            dq += tl.dot(ds, k)
            tl.store(DQ_block_ptr, dq)
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                dq = tl.dot(ds, k)
            else:
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds)))
            tl.store(DQ_block_ptr, dq)
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
    tl.store(DV_block_ptr, dv)
    tl.store(DK_block_ptr, dk)


@triton.jit
def _bwd_kernel(Q, K, V, sm_scale, Out, DO, DQ, DK, DV, L, D, stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, Z, H, N_CTX, Z_H_N_CTX, SQ_Z_H_N_CTX, BLOCK_M: 'tl.constexpr', BLOCK_DMODEL: 'tl.constexpr', BLOCK_N: 'tl.constexpr', SEQUENCE_PARALLEL: 'tl.constexpr', CAUSAL: 'tl.constexpr', MMA_V3: 'tl.constexpr'):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    Q_block_ptr = tl.make_block_ptr(base=Q, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(base=K, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_kn, stride_kk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(base=V, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    DO_block_ptr = tl.make_block_ptr(base=DO, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    if SEQUENCE_PARALLEL:
        DQ_block_ptr = tl.make_block_ptr(base=DQ, shape=(SQ_Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    else:
        DQ_block_ptr = tl.make_block_ptr(base=DQ, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    DK_block_ptr = tl.make_block_ptr(base=DK, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_kn, stride_kk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    DV_block_ptr = tl.make_block_ptr(base=DV, shape=(Z_H_N_CTX, BLOCK_DMODEL), strides=(stride_vn, stride_vk), offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_DMODEL), order=(1, 0))
    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO, DQ, DK, DV, L, D, Q_block_ptr, K_block_ptr, V_block_ptr, DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr, stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, Z, H, N_CTX, off_h, off_z, off_hz, start_n, num_block_n, BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N, SEQUENCE_PARALLEL=SEQUENCE_PARALLEL, CAUSAL=CAUSAL, MMA_V3=MMA_V3)
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(Q, K, V, sm_scale, qk_scale, Out, DO, DQ, DK, DV, L, D, Q_block_ptr, K_block_ptr, V_block_ptr, DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr, stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vn, stride_vk, Z, H, N_CTX, off_h, off_z, off_hz, start_n, num_block_n, BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_N=BLOCK_N, SEQUENCE_PARALLEL=SEQUENCE_PARALLEL, CAUSAL=CAUSAL, MMA_V3=MMA_V3)


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_forward_kernel(input_pointer, target_pointer, weight_pointer, sum_weights_pointer, output_pointer, batch_dim, spatial_dim, input_batch_stride, input_feat_stride, input_spatial_stride, target_batch_stride, target_spatial_stride, output_batch_stride, output_spatial_stride, reduction: 'tl.constexpr', weighted: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_SPATIAL: 'tl.constexpr'):
    """
    Measures the negative log likelihood loss between the input and target,
    with optional reweighing of each class.

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim, spatial_dim].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to a container the sum of the class weights is written to.
            The container must be of shape [batch_dim/BLOCK_SIZE_BATCH].
        output_pointer: Pointer to a container the loss is written to.
            The container must be of shape [batch_dim, spatial_dim] if reduction is 'none',
            and otherwise of shape [batch_dim/BLOCK_SIZE].
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_spatial_stride: Stride necessary to jump one element along the
            output container's spatial dimension.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the summed weights and
            output container, which should later be summed.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)
    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim
    target_pointer += target_batch_stride * batch_offset[:, None] + target_spatial_stride * spatial_offset[None, :]
    target = tl.load(target_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
    input_pointer += input_feat_stride * target + input_batch_stride * batch_offset[:, None] + input_spatial_stride * spatial_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
    output = -input
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask[:, None] & spatial_mask[None, :])
        output *= weight
    if reduction == 'none':
        output_pointer += output_batch_stride * batch_offset[:, None] + output_spatial_stride * spatial_offset[None, :]
        tl.store(output_pointer, output, mask=batch_mask[:, None] & spatial_mask[None, :])
    elif reduction == 'mean':
        if weighted:
            tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))
            tl.store(output_pointer + batch_pid, tl.sum(output))
        else:
            tl.store(output_pointer + batch_pid, tl.sum(output) / (batch_dim * spatial_dim))
    elif reduction == 'sum':
        tl.store(output_pointer + batch_pid, tl.sum(output))


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'spatial_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_SPATIAL': lambda args: next_power_of_2(args['spatial_dim'])})
@triton.jit
def nll_loss_backward_kernel(output_grad_pointer, target_pointer, weight_pointer, sum_weights_pointer, input_grad_pointer, batch_dim, spatial_dim, output_grad_batch_stride, output_grad_feat_stride, target_batch_stride, target_spatial_stride, input_grad_batch_stride, input_grad_feat_stride, input_grad_spatial_stride, reduction: 'tl.constexpr', weighted: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_SPATIAL: 'tl.constexpr'):
    """
    Calculates the input gradient of negative log likelihood loss.

    Args:
        output_grad_pointer: Pointer to the loss's output gradients.
            The output gradients must be of shape [batch_dim, spatial_dim]
            if reduction is 'none', and otherwise [batch_dim/BLOCK_SIZE_BATCH].
        target_pointer: Pointer to the target.
            The target must be of shape [batch_dim, spatial_dim].
        weight_pointer: Pointer to an optional class weight vector.
            The class weight vector, if provided, must be of shape [feat_dim].
        sum_weights_pointer: Pointer to the sum of the class weights if the classes were weighed.
            The sum of weights must be a scalar.
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim, spatial_dim] and zeroed.
        batch_dim: Batch dimension.
        spatial_dim: Spatial dimension.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_spatial_stride: Stride necessary to jump one element along the
            input's spatial dimension.
        target_batch_stride: Stride necessary to jump one element along the
            target's batch dimension.
        target_spatial_stride: Stride necessary to jump one element along the
            target's spatial dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        input_grad_spatial_stride: Stride necessary to jump one element along the
            input gradient container's spatial dimension.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the loss
            across all entries, and 'sum' for summing the loss across all entries.
        weighted: Flag for weighing each class.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_SPATIAL: Block size across the spatial dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    spatial_offset = tl.arange(0, BLOCK_SIZE_SPATIAL)
    batch_mask = batch_offset < batch_dim
    spatial_mask = spatial_offset < spatial_dim
    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += output_grad_batch_stride * batch_offset[:, None] + output_grad_feat_stride * spatial_offset[None, :]
        output_grad_mask = batch_mask[:, None] & spatial_mask[None, :]
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask)
    input_grad = -output_grad
    target_pointer += target_batch_stride * batch_offset[:, None] + target_spatial_stride * spatial_offset[None, :]
    target = tl.load(target_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
    if weighted:
        weight = tl.load(weight_pointer + target, mask=batch_mask[:, None] & spatial_mask[None, :])
        input_grad *= weight
        if reduction == 'mean':
            input_grad /= tl.load(sum_weights_pointer)
    elif reduction == 'mean':
        input_grad /= batch_dim * spatial_dim
    input_grad_pointer += input_grad_feat_stride * target + input_grad_batch_stride * batch_offset[:, None] + input_grad_spatial_stride * spatial_offset[None, :]
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] & spatial_mask[None, :])


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def p_loss_forward_kernel(input_pointer, target_pointer, output_pointer, size, p_loss: 'tl.constexpr', reduction: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    Measures the L1 or squared L2 norm of the difference between the input
    and target (i.e., mean absolute error or mean squared error).

    Args:
        input_pointer: Pointer to the input.
            The input must be of shape [size].
        target_pointer: Pointer to the target.
            The target must be of shape [size].
        output_pointer: Pointer to a container the error is written to.
            The container must be of shape [size] if reduction is 'none',
            and otherwise of shape [size/BLOCK_SIZE].
        size: Number of elements in the input and target.
        p_loss: p-norm used to compute the error.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
            If a reduction method is specified, the reduced result of each
            program is written to a separate index in the output container,
            which should later be summed.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    input = tl.load(input_pointer + offset, mask=mask)
    target = tl.load(target_pointer + offset, mask=mask)
    diff = input - target
    if p_loss == 1:
        error = tl.abs(diff)
    elif p_loss == 2:
        error = diff * diff
    if reduction == 'none':
        tl.store(output_pointer + offset, error, mask=mask)
    elif reduction == 'mean':
        tl.store(output_pointer + pid, tl.sum(error) / size)
    elif reduction == 'sum':
        tl.store(output_pointer + pid, tl.sum(error))


@triton.autotune(configs=element_wise_kernel_configs(), key=['size'])
@triton.jit
def p_loss_backward_kernel(output_grad_pointer, input_pointer, target_pointer, input_grad_pointer, target_grad_pointer, size, p_loss: 'tl.constexpr', reduction: 'tl.constexpr', BLOCK_SIZE: 'tl.constexpr'):
    """
    Calculates the input gradient of the mean absolute error or
    mean squared error.

    Args:
        output_grad_pointer: Pointer to the error's output gradients.
            The output gradients must be a scalar or of shape [size].
        input_pointer: Pointer to the input.
            The input must be of shape [size].
        target_pointer: Pointer to the target.
            The target must be of shape [size].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [size].
        target_grad_pointer: Pointer to a container the target's gradients are written to.
            The container must be of shape [size].
        size: Number of elements in the input and target.
        p_loss: p-norm used to compute the error whose gradient is calculated.
            Options are 1 for MAE and 2 for MSE.
        reduction: Reduction strategy for the output whose gradient is calculated.
            Options are 'none' for no reduction, 'mean' for averaging the error
            across all entries, and 'sum' for summing the error across all entries.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    output_grad_mask = None
    if reduction == 'none':
        output_grad_pointer += offset
        output_grad_mask = mask
    input = tl.load(input_pointer + offset, mask=mask)
    target = tl.load(target_pointer + offset, mask=mask)
    output_grad = tl.load(output_grad_pointer, mask=output_grad_mask)
    if p_loss == 1:
        input_grad = tl.where(target <= input, 1, -1)
    elif p_loss == 2:
        input_grad = 2 * (input - target)
    if reduction == 'mean':
        input_grad /= size
    input_grad *= output_grad
    tl.store(input_grad_pointer + offset, input_grad, mask=mask)
    tl.store(target_grad_pointer + offset, -input_grad, mask=mask)


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def rms_norm_forward_kernel(input_pointer, weight_pointer, inv_rms_pointer, output_pointer, batch_dim, feat_dim, input_batch_stride, input_feat_stride, output_batch_stride, output_feat_stride, eps, scale_by_weight: 'tl.constexpr', save_stats: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Root-mean-square-normalizes the input.

    Args:
        input_pointer: Pointer to the input to root-mean-square-normalize.
            The input must be of shape [batch_dim, feat_dim].
        weight_pointer: Pointer to optional weights for linear transform.
            The weights, if provided, must be of shape [feat_dim].
        inv_rms_pointer: Pointer to an optional container the input's inverse
            root mean square is written to if save_stats is True.
            The container, if provided, must be of shape [batch_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        eps: Epsilon added in the square root in the denominator
            to avoid division by zero.
        scale_by_weight: Flag for scaling the normalized output by weights.
        save_stats: Flag for saving the root mean square.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    output_pointer += output_batch_stride * batch_offset[:, None] + output_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    inv_rms = tl.rsqrt(tl.sum(input * input, axis=1) / feat_dim + eps)
    output = input * inv_rms[:, None]
    if save_stats:
        tl.store(inv_rms_pointer + batch_offset, inv_rms, mask=batch_mask)
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        output *= weight
    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[None, :])


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def rms_norm_backward_kernel(output_grad_pointer, input_pointer, inv_rms_pointer, weight_pointer, input_grad_pointer, weight_grad_pointer, batch_dim, feat_dim, output_grad_batch_stride, output_grad_feat_stride, input_batch_stride, input_feat_stride, input_grad_batch_stride, input_grad_feat_stride, weight_grad_batch_stride, weight_grad_feat_stride, scale_by_weight: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Calculates the input gradient of root mean square normalization.

    Args:
        output_grad_pointer: Pointer to root mean square normalization's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim].
        input_pointer: Pointer to the input.
            The input must be of shape [batch_dim, feat_dim].
        inv_rms_pointer: Pointer to the input's inverse root mean square.
            The inverse root mean square should be of shape [batch_dim].
        weight_pointer: Pointer to optional weights if affine transform occurred.
            The weights, if provided, must be of shape [feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        weight_grad_pointer: Pointer to an optional container the weights' row-wise gradients
            are written to if scale_by_weight is True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        bias_grad_pointer: Pointer to an optional container the bias vector's row-wise gradients
            are written to if scale_by_weight and add_bias are True, which should later be summed.
            The container, if provided, must be of shape [batch_dim/BLOCK_SIZE_BATCH, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        weight_grad_batch_stride: Stride necessary to jump one element along the
            weight gradient container's batch dimension.
        weight_grad_feat_stride: Stride necessary to jump one element along the
            weight gradient container's feature dimension.
        scale_by_weight: Flag for scaling the normalized output by weights.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    output_grad_pointer += output_grad_batch_stride * batch_offset[:, None] + output_grad_feat_stride * feat_offset[None, :]
    input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    input_grad_pointer += input_grad_batch_stride * batch_offset[:, None] + input_grad_feat_stride * feat_offset[None, :]
    output_grad = tl.load(output_grad_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    inv_rms = tl.load(inv_rms_pointer + batch_offset, mask=batch_mask)
    pre_lin = input * inv_rms[:, None]
    if scale_by_weight:
        weight = tl.load(weight_pointer + feat_offset, mask=feat_mask)
        weight_output_grad_prod = weight * output_grad
    else:
        weight_output_grad_prod = output_grad
    term1 = input * tl.sum(input * weight_output_grad_prod, axis=1)
    term2 = inv_rms[:, None] * inv_rms[:, None]
    input_grad = inv_rms[:, None] * (weight_output_grad_prod - term1 * term2 / feat_dim)
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] & feat_mask[None, :])
    if scale_by_weight:
        weight_grad_pointer += weight_grad_batch_stride * batch_pid + weight_grad_feat_stride * feat_offset
        tl.store(weight_grad_pointer, tl.sum(output_grad * pre_lin, axis=0), mask=feat_mask)


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_forward_kernel(input_pointer, output_pointer, batch_dim, feat_dim, input_batch_stride, input_feat_stride, output_batch_stride, output_feat_stride, log: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Normalizes the input using softmax.

    Args:
        input_pointer: Pointer to the input to normalize.
            The input must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to a container the result is written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        input_batch_stride: Stride necessary to jump one element along the
            input's batch dimension.
        input_feat_stride: Stride necessary to jump one element along the
            input's feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output container's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output container's feature dimension.
        log: Flag for indicating if the log of softmax should be taken.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
    output_pointer += output_batch_stride * batch_offset[:, None] + output_feat_stride * feat_offset[None, :]
    input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :], other=-float('inf'))
    input -= tl.max(input, axis=1)[:, None]
    numerator = tl.exp(input)
    denominator = tl.sum(numerator, axis=1)[:, None]
    if log:
        output = input - tl.log(denominator)
    else:
        output = numerator / denominator
    tl.store(output_pointer, output, mask=batch_mask[:, None] & feat_mask[None, :])


@triton.autotune(configs=warps_kernel_configs(), key=['batch_dim', 'feat_dim'])
@triton.heuristics({'BLOCK_SIZE_BATCH': BLOCK_SIZE_BATCH_heuristic, 'BLOCK_SIZE_FEAT': lambda args: next_power_of_2(args['feat_dim'])})
@triton.jit
def softmax_backward_kernel(output_grad_pointer, output_pointer, input_grad_pointer, batch_dim, feat_dim, output_grad_batch_stride, output_grad_feat_stride, output_batch_stride, output_feat_stride, input_grad_batch_stride, input_grad_feat_stride, log: 'tl.constexpr', BLOCK_SIZE_BATCH: 'tl.constexpr', BLOCK_SIZE_FEAT: 'tl.constexpr'):
    """
    Calculates the input gradient of softmax.

    Args:
        output_grad_pointer: Pointer to softmax's output gradients.
            The output gradients must be of shape [batch_dim, feat_dim].
        output_pointer: Pointer to softmax's output.
            The output must be of shape [batch_dim, feat_dim].
        input_grad_pointer: Pointer to a container the input's gradients are written to.
            The container must be of shape [batch_dim, feat_dim].
        batch_dim: Batch dimension.
        feat_dim: Dimensionality of the features.
        output_grad_batch_stride: Stride necessary to jump one element along the
            output gradients' batch dimension.
        output_grad_feat_stride: Stride necessary to jump one element along the
            output gradients' feature dimension.
        output_batch_stride: Stride necessary to jump one element along the
            output's batch dimension.
        output_feat_stride: Stride necessary to jump one element along the
            output's feature dimension.
        input_grad_batch_stride: Stride necessary to jump one element along the
            input gradient container's batch dimension.
        input_grad_feat_stride: Stride necessary to jump one element along the
            input gradient container's feature dimension.
        log: Flag indicating if log of softmax was taken.
        BLOCK_SIZE_BATCH: Block size across the batch dimension.
        BLOCK_SIZE_FEAT: Block size across the feature dimension.
    """
    batch_pid = tl.program_id(axis=0)
    batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
    feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)
    batch_mask = batch_offset < batch_dim
    feat_mask = feat_offset < feat_dim
    output_grad_pointer += output_grad_batch_stride * batch_offset[:, None] + output_grad_feat_stride * feat_offset[None, :]
    output_pointer += output_batch_stride * batch_offset[:, None] + output_feat_stride * feat_offset[None, :]
    input_grad_pointer += input_grad_batch_stride * batch_offset[:, None] + input_grad_feat_stride * feat_offset[None, :]
    output_grad = tl.load(output_grad_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    output = tl.load(output_pointer, mask=batch_mask[:, None] & feat_mask[None, :])
    if log:
        input_grad = output_grad - tl.exp(output) * tl.sum(output_grad, axis=1)[:, None]
    else:
        input_grad = output * (output_grad - tl.sum(output_grad * output, axis=1)[:, None])
    tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] & feat_mask[None, :])

