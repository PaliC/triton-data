# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_sahanp/cv/ccvxv5rnyn63al3bsrfre25l4tkjd2n5f45ftiq4nwsvxqkxdcvv.py
# Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_11 => add_172, mul_159, mul_160, sub_50
#   input_12 => relu_3
#   input_13 => convolution_6
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_73), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_75), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_77), kwargs = {})
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_79), kwargs = {})
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_172,), kwargs = {})
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lu/clunqxusjru3vq7vha7ncqkfcfdrfnfv6t3lfs2bzn5wnfzmgvcd.py
# Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_41 => clone_129, var_mean_41
# Graph fragment:
#   %clone_129 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_161,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_41 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_129, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_layer_norm_1 = async_compile.triton('triton_red_fused_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 2
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r3) + (75264*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp4, xmask)
    tl.store(out_ptr1 + (x5), tmp5, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2e/c2ee75dqzcsqlpmyc3z46s2v6bjoq34rkgcspokj3gyfib7q45sx.py
# Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_41 => clone_129, var_mean_41
# Graph fragment:
#   %clone_129 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_161,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_41 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_129, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_native_layer_norm_2 = async_compile.triton('triton_per_fused_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (1568*x1)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (784*r2) + (1568*x1)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (784*r2) + (1568*x1)), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x3), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3s/c3s42jesq7vyftbwawxqop7mjveguxfakp5ijueueoaskex7ak5t.py
# Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_41 => add_177, add_178, clone_129, mul_167, mul_168, rsqrt_41, sub_53, var_mean_41
# Graph fragment:
#   %clone_129 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%permute_161,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_41 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_129, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_129, %getitem_187), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_186, 1e-05), kwargs = {})
#   %rsqrt_41 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_177,), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %rsqrt_41), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_167, %arg18_1), kwargs = {})
#   %add_178 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_168, %arg19_1), kwargs = {})
triton_poi_fused_native_layer_norm_3 = async_compile.triton('triton_poi_fused_native_layer_norm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 784) % 192
    x0 = xindex % 784
    x2 = (xindex // 150528)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (784*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0 + (784*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 192.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pq/cpqyubxjq2aqa7wotcu4mp63xcpeglhyfjd6omosiso6gtlpqjzi.py
# Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_166, [2, 2], [2, 2], [0, 0], True), kwargs = {})
triton_poi_fused_avg_pool2d_4 = async_compile.triton('triton_poi_fused_avg_pool2d_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 14
    x3 = (xindex // 14)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (56*x3) + (784*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (56*x3) + (784*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + (2*x2) + (56*x3) + (784*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x2) + (56*x3) + (784*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (y0 + (192*x5) + (37632*y1)), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3a/c3a2ha5z245m3nyfkd34lglhq26z4lxkbavkoew37fjsd74kwt27.py
# Topologically Sorted Source Nodes: [attn_28, attn_29], Original ATen: [aten.mul, aten._softmax]
# Source node to ATen node mapping:
#   attn_28 => mul_169
#   attn_29 => amax_6, clone_131, div_6, exp_6, sub_54, sum_7
# Graph fragment:
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_169, 0.1767766952966369), kwargs = {})
#   %clone_131 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%mul_169,), kwargs = {memory_format: torch.contiguous_format})
#   %amax_6 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%clone_131, [-1], True), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_131, %amax_6), kwargs = {})
#   %exp_6 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_54,), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_6, [-1], True), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_6, %sum_7), kwargs = {})
triton_per_fused__softmax_mul_5 = async_compile.triton('triton_per_fused__softmax_mul_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_mul_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_mul_5(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 84672
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x7 = xindex
    x0 = xindex % 54
    x3 = xindex % 9
    x4 = (xindex // 9) % 6
    x5 = (xindex // 54) % 196
    x6 = (xindex // 10584)
    tmp0 = tl.load(in_ptr0 + (r2 + (9*x7)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + (9*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.1767766952966369
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp9 = tmp4 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r2 + (9*x3) + (81*x5) + (15876*x4) + (95256*x6)), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ls/clsltjr3oz23hnho2v3jvwxhh26haxlg537rt7oyhb3ypckgmp43.py
# Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   linear_88 => mm_27
# Graph fragment:
#   %mm_27 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_253, %permute_162), kwargs = {})
triton_poi_fused_mm_6 = async_compile.triton('triton_poi_fused_mm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((784*x1) + (150528*(x0 // 784)) + (x0 % 784)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zh/czh22v2553yhwlvsu2heu4xrbdfytzyeyryzyb74x5l3ffb7dajo.py
# Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_8 => clone_133
# Graph fragment:
#   %clone_133 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_18,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 9
    x2 = (xindex // 288) % 196
    x0 = xindex % 32
    x3 = (xindex // 56448) % 6
    x4 = (xindex // 338688)
    x6 = xindex
    tmp0 = (-1) + (2*(x2 // 14)) + (x1 // 3)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*(x2 % 14)) + (x1 % 3)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-5568) + x0 + (32*x3) + (192*(x1 % 3)) + (384*(x2 % 14)) + (5376*(x1 // 3)) + (10752*(x2 // 14)) + (150528*x4)), tmp10 & xmask, other=0.0)
    tl.store(out_ptr0 + (x6), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qa/cqawtgtd43r62ogxtlfzv465thlihq67wfdzvde5vp4qd67kc3ui.py
# Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.col2im]
# Source node to ATen node mapping:
#   x_224 => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 30, 30], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_col2im_8 = async_compile.triton('triton_poi_fused_col2im_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_col2im_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1382400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lj/cljon74nbkm5oa56wuzatumpfjp3zsh6wnxupyvrmx5awv6duxmq.py
# Topologically Sorted Source Nodes: [x_224, x_223], Original ATen: [aten.col2im, aten.clone]
# Source node to ATen node mapping:
#   x_223 => clone_134
#   x_224 => full_default, index_put_4
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([8, 192, 30, 30], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %clone_134 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_170,), kwargs = {memory_format: torch.contiguous_format})
#   %index_put_4 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [None, None, %unsqueeze_105, %add_182], %permute_171, True), kwargs = {})
triton_poi_fused_clone_col2im_9 = async_compile.triton('triton_poi_fused_clone_col2im_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_col2im_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_col2im_9(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 196
    x3 = (xindex // 196)
    y0 = yindex % 32
    y1 = (yindex // 32)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x3) + (288*x2) + (56448*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr1 + (x5 + (1792*y4)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/25/c25vrnphbmup2hf6cp5vpqppwejixllldonwewqwy2k4gmmt7z7o.py
# Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.col2im]
# Source node to ATen node mapping:
#   x_224 => add_181
# Graph fragment:
#   %add_181 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_102, %unsqueeze_103), kwargs = {})
triton_poi_fused_col2im_10 = async_compile.triton('triton_poi_fused_col2im_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[64], 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_col2im_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_col2im_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 42
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    tmp0 = x1 + (2*x0)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/64/c64p4zuf2dthed65eurhovtbsm5n3acez4wlqpiptslmfii6qqfm.py
# Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_225 => clone_135
# Graph fragment:
#   %clone_135 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_172,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_11 = async_compile.triton('triton_poi_fused_clone_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 28) % 28
    y0 = yindex % 28
    x3 = xindex
    y2 = (yindex // 784)
    y5 = yindex
    tmp0 = 1 + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 30, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + y0
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (31 + y0 + (30*y1) + (900*x3) + (172800*y2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tl.store(out_ptr0 + (x3 + (192*y5)), tmp11, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zm/czmqadbxbktsam7icezgse2w3dsmop5fjvnj2iw4huoh7v63z3hz.py
# Topologically Sorted Source Nodes: [x_225, x_227, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_42 => clone_137, var_mean_42
#   x_225 => add_183
#   x_227 => add_184
# Graph fragment:
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_266, %arg24_1), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_161, %add_183), kwargs = {})
#   %clone_137 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_184,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_42 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_137, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_12 = async_compile.triton('triton_red_fused_add_native_layer_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 12544
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 784
    x2 = (xindex // 1568)
    x5 = xindex
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (784*r3) + (75264*x0) + (150528*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (96*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr3 + (r3 + (96*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp9, xmask)
    tl.store(out_ptr2 + (x5), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uf/cufxsxanqq2vzwe2tmch33nfg4hfcbtkzqiuzvvdicrjvvxxq6xy.py
# Topologically Sorted Source Nodes: [x_225, x_227, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_42 => clone_137, var_mean_42
#   x_225 => add_183
#   x_227 => add_184
# Graph fragment:
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_266, %arg24_1), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_161, %add_183), kwargs = {})
#   %clone_137 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_184,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_42 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_137, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7o/c7ogadbhngjgbtyfsmkdqewfjz7t6fktro6gca47rfwkuz3xz57h.py
# Topologically Sorted Source Nodes: [x_225, x_227, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_42 => add_185, add_186, clone_137, mul_170, mul_171, rsqrt_42, sub_55, var_mean_42
#   x_225 => add_183
#   x_227 => add_184
# Graph fragment:
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_266, %arg24_1), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_161, %add_183), kwargs = {})
#   %clone_137 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_184,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_42 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_137, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_137, %getitem_189), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_188, 1e-05), kwargs = {})
#   %rsqrt_42 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_185,), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %rsqrt_42), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_170, %arg25_1), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_171, %arg26_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_14 = async_compile.triton('triton_poi_fused_add_native_layer_norm_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (150528*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (y3), ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 192.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x2 + (192*y3)), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ir/cirtvddjrdsmtjscxw5v3f4nnsxbusuj3csg6evoerlha2locy44.py
# Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_229 => add_187, erf_20, mul_172, mul_173, mul_174
# Graph fragment:
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_268, 0.5), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_268, 0.7071067811865476), kwargs = {})
#   %erf_20 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_173,), kwargs = {})
#   %add_187 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_20, 1), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %add_187), kwargs = {})
triton_poi_fused_gelu_15 = async_compile.triton('triton_poi_fused_gelu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 576
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d4/cd4yzfllmmrwgesm72tnt4xqc4uue7qtsycdwksnd5lutjit7dmz.py
# Topologically Sorted Source Nodes: [x_225, x_227, x_233, layer_norm_43], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_43 => add_189, add_190, clone_140, mul_175, mul_176, rsqrt_43, sub_56, var_mean_43
#   x_225 => add_183
#   x_227 => add_184
#   x_233 => add_188
# Graph fragment:
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_266, %arg24_1), kwargs = {})
#   %add_184 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_161, %add_183), kwargs = {})
#   %add_188 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_184, %view_270), kwargs = {})
#   %clone_140 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_188,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_140, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_140, %getitem_191), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_190, 1e-05), kwargs = {})
#   %rsqrt_43 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_189,), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_56, %rsqrt_43), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_175, %arg31_1), kwargs = {})
#   %add_190 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_176, %arg32_1), kwargs = {})
triton_red_fused_add_native_layer_norm_16 = async_compile.triton('triton_red_fused_add_native_layer_norm_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 784
    x1 = (xindex // 784)
    x3 = xindex
    tmp12_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (784*r2) + (150528*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r2 + (192*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr3 + (r2 + (192*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp9 = tmp7 + tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(rmask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(rmask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(rmask & xmask, tmp12_weight_next, tmp12_weight)
        tl.store(in_out_ptr0 + (r2 + (192*x3)), tmp10, rmask & xmask)
    tmp12_tmp, tmp13_tmp, tmp14_tmp = triton_helpers.welford(
        tmp12_mean, tmp12_m2, tmp12_weight, 1
    )
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r2 + (192*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tmp15 - tmp12
        tmp17 = 192.0
        tmp18 = tmp13 / tmp17
        tmp19 = 1e-05
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp24 = tmp22 * tmp23
        tmp26 = tmp24 + tmp25
        tl.store(out_ptr2 + (r2 + (192*x3)), tmp26, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wa/cwaxlx3w3x74ep2qfz7evdqnfwawog5ylabzjmco4wsyvct75wwi.py
# Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%permute_180, [2, 2], [2, 2], [0, 0], True), kwargs = {})
triton_poi_fused_avg_pool2d_17 = async_compile.triton('triton_poi_fused_avg_pool2d_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[524288], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192) % 14
    x2 = (xindex // 2688)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x1) + (10752*x2)), xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + (384*x1) + (10752*x2)), xmask)
    tmp3 = tl.load(in_ptr0 + (5376 + x0 + (384*x1) + (10752*x2)), xmask)
    tmp5 = tl.load(in_ptr0 + (5568 + x0 + (384*x1) + (10752*x2)), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7t/c7twqv5xyf52zxddp74e6t3q57w4tgz5yre45z6c5wtkt6btmtfv.py
# Topologically Sorted Source Nodes: [x_236, x_238, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_44 => add_197, add_198, clone_148, mul_178, mul_179, rsqrt_44, sub_58, var_mean_44
#   x_236 => add_195
#   x_238 => add_196
# Graph fragment:
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_284, %arg37_1), kwargs = {})
#   %add_196 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_188, %add_195), kwargs = {})
#   %clone_148 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_196,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_148, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_148, %getitem_193), kwargs = {})
#   %add_197 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_192, 1e-05), kwargs = {})
#   %rsqrt_44 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_197,), kwargs = {})
#   %mul_178 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %rsqrt_44), kwargs = {})
#   %mul_179 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_178, %arg38_1), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_179, %arg39_1), kwargs = {})
triton_per_fused_add_native_layer_norm_18 = async_compile.triton('triton_per_fused_add_native_layer_norm_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp31, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xa/cxazsnt752syimg2nxxrgzoohggexsuslstygotbeczs3su3nzci.py
# Topologically Sorted Source Nodes: [x_236, x_238, x_244, layer_norm_45], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_45 => add_201, add_202, clone_151, mul_183, mul_184, rsqrt_45, sub_59, var_mean_45
#   x_236 => add_195
#   x_238 => add_196
#   x_244 => add_200
# Graph fragment:
#   %add_195 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_284, %arg37_1), kwargs = {})
#   %add_196 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_188, %add_195), kwargs = {})
#   %add_200 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_196, %view_288), kwargs = {})
#   %clone_151 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_200,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_151, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_59 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_151, %getitem_195), kwargs = {})
#   %add_201 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_194, 1e-05), kwargs = {})
#   %rsqrt_45 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_201,), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_59, %rsqrt_45), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_183, %arg44_1), kwargs = {})
#   %add_202 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_184, %arg45_1), kwargs = {})
triton_per_fused_add_native_layer_norm_19 = async_compile.triton('triton_per_fused_add_native_layer_norm_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kp/ckpfwunmegcn6lw4nint52erroao3brkmoqvwghrxpn2tbdkddp2.py
# Topologically Sorted Source Nodes: [x_268], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_268 => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_218, %arg70_1, %arg71_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    tmp0 = tl.load(in_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(out_ptr0 + (y0 + (784*x2) + (150528*y1)), tmp8, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/63/c634ycojrbqxqddc2pdeoc7lfkjwg6kqiryqbs6kdokgrmudr6ne.py
# Topologically Sorted Source Nodes: [x_270, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_49 => clone_174, var_mean_49
#   x_270 => add_225
# Graph fragment:
#   %add_225 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_219, %arg72_1), kwargs = {})
#   %clone_174 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_225,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_174, [3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_native_layer_norm_21 = async_compile.triton('triton_red_fused_add_native_layer_norm_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[8192, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4704
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x1 = (xindex // 3) % 196
    x2 = (xindex // 588)
    x5 = xindex % 588
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x6 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + (196*r3) + (25088*x0) + (75264*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r3 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x6), tmp6, xmask)
    tl.store(out_ptr1 + (x6), tmp7, xmask)
    tl.store(out_ptr2 + (x6), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tl/ctlpry2qpxc6s42nhmdq7w4awtsdlcv6xbykpm7h5vglg2p4caq4.py
# Topologically Sorted Source Nodes: [x_270, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_49 => clone_174, var_mean_49
#   x_270 => add_225
# Graph fragment:
#   %add_225 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_219, %arg72_1), kwargs = {})
#   %clone_174 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_225,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_174, [3]), kwargs = {correction: 0, keepdim: True})
triton_per_fused_add_native_layer_norm_22 = async_compile.triton('triton_per_fused_add_native_layer_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (3*x0)), rmask & xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g5/cg5upgwkgh2mkjdyphl2nkjof3bcjrgjwu4hklti2wcdsxntxs4q.py
# Topologically Sorted Source Nodes: [x_270, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_49 => add_226, add_227, clone_174, mul_199, mul_200, rsqrt_49, sub_65, var_mean_49
#   x_270 => add_225
# Graph fragment:
#   %add_225 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_219, %arg72_1), kwargs = {})
#   %clone_174 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_225,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_49 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_174, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_174, %getitem_203), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_202, 1e-05), kwargs = {})
#   %rsqrt_49 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_226,), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %rsqrt_49), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %arg73_1), kwargs = {})
#   %add_227 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %arg74_1), kwargs = {})
triton_poi_fused_add_native_layer_norm_23 = async_compile.triton('triton_poi_fused_add_native_layer_norm_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + (384*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (y3), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 384.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2 + (384*y3)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ib/cibfu6ttuopd2evhzrlsrecdmsz7bw7thlcnc5mlw6nr5jpc3qxl.py
# Topologically Sorted Source Nodes: [x_270, x_276, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_50 => add_229, add_230, clone_176, mul_201, mul_202, rsqrt_50, sub_66, var_mean_50
#   x_270 => add_225
#   x_276 => add_228
# Graph fragment:
#   %add_225 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_219, %arg72_1), kwargs = {})
#   %add_228 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_225, %view_330), kwargs = {})
#   %clone_176 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_228,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_50 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_176, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_176, %getitem_212), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_211, 1e-05), kwargs = {})
#   %rsqrt_50 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_229,), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %rsqrt_50), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_201, %arg78_1), kwargs = {})
#   %add_230 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_202, %arg79_1), kwargs = {})
triton_red_fused_add_native_layer_norm_24 = async_compile.triton('triton_red_fused_add_native_layer_norm_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1568
    rnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 196
    x1 = (xindex // 196)
    x3 = xindex
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (196*r2) + (75264*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r2 + (384*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r2 + (384*x3)), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r2 + (384*x3)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp13 - tmp10
        tmp15 = 384.0
        tmp16 = tmp11 / tmp15
        tmp17 = 1e-05
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r2 + (384*x3)), tmp24, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sn/csnxatvwovm7jq5eczckmcpgs3sr5qlxisdelp3jwl6yhjp3ktvv.py
# Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_278 => add_231, erf_24, mul_203, mul_204, mul_205
# Graph fragment:
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_332, 0.5), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_332, 0.7071067811865476), kwargs = {})
#   %erf_24 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_204,), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_24, 1), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_203, %add_231), kwargs = {})
triton_poi_fused_gelu_25 = async_compile.triton('triton_poi_fused_gelu_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bf/cbfvcjrkwoy3krwzl7pnxgrgvzhzrky567gwuzv7jdobph7qgzm3.py
# Topologically Sorted Source Nodes: [x_282, layer_norm_51], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_51 => add_233, add_234, clone_179, mul_206, mul_207, rsqrt_51, sub_67, var_mean_51
#   x_282 => add_232
# Graph fragment:
#   %add_232 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_228, %view_334), kwargs = {})
#   %clone_179 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_232,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_51 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_179, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_179, %getitem_214), kwargs = {})
#   %add_233 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_213, 1e-05), kwargs = {})
#   %rsqrt_51 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_233,), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %rsqrt_51), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %arg84_1), kwargs = {})
#   %add_234 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_207, %arg85_1), kwargs = {})
triton_per_fused_add_native_layer_norm_26 = async_compile.triton('triton_per_fused_add_native_layer_norm_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qm/cqmnxjxvpzuzi4e656wrtnqvagra3m5mblvqxhtze2qrs6cl3i4n.py
# Topologically Sorted Source Nodes: [x_282, x_287, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_52 => add_236, add_237, clone_181, mul_208, mul_209, rsqrt_52, sub_68, var_mean_52
#   x_282 => add_232
#   x_287 => add_235
# Graph fragment:
#   %add_232 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_228, %view_334), kwargs = {})
#   %add_235 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_232, %view_340), kwargs = {})
#   %clone_181 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_235,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_181, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_181, %getitem_223), kwargs = {})
#   %add_236 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_222, 1e-05), kwargs = {})
#   %rsqrt_52 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_236,), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %rsqrt_52), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %arg89_1), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %arg90_1), kwargs = {})
triton_per_fused_add_native_layer_norm_27 = async_compile.triton('triton_per_fused_add_native_layer_norm_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel):
    xnumel = 1568
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3k/c3k265pgtuuov5wqd4wxzqa2ceyawrysnzpwnobwicazpmnzsojc.py
# Topologically Sorted Source Nodes: [x_427, layer_norm_77], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_77 => add_324, add_325, mul_297, mul_298, rsqrt_77, sub_93, var_mean_77
#   x_427 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%expand_25, %view_465], 1), kwargs = {})
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_3, %getitem_357), kwargs = {})
#   %add_324 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_356, 1e-05), kwargs = {})
#   %rsqrt_77 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_324,), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %rsqrt_77), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_297, %arg228_1), kwargs = {})
#   %add_325 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_298, %arg229_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_28 = async_compile.triton('triton_per_fused_cat_native_layer_norm_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp40 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp6, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp6, other=0.0)
    tmp11 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.broadcast_to(tmp17, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp22, 0))
    tmp24 = tl.full([1], 384, tl.int32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 / tmp25
    tmp27 = tmp17 - tmp26
    tmp28 = tmp27 * tmp27
    tmp29 = tl.broadcast_to(tmp28, [RBLOCK])
    tmp31 = tl.where(rmask, tmp29, 0)
    tmp32 = triton_helpers.promote_to_tensor(tl.sum(tmp31, 0))
    tmp33 = tmp16 - tmp26
    tmp34 = 384.0
    tmp35 = tmp32 / tmp34
    tmp36 = 1e-05
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr3 + (r2 + (384*x3)), tmp43, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/td/ctdyasha5nhmnccv6rbdol2orvfib5itwv6yt7hxi6vamr3khjao.py
# Topologically Sorted Source Nodes: [q_30], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   q_30 => mul_299
# Graph fragment:
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_471, 0.1767766952966369), kwargs = {})
triton_poi_fused_mul_29 = async_compile.triton('triton_poi_fused_mul_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_29(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 0.1767766952966369
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b3/cb3rzie4gfxo5gfx7gerqnxrkaomyjxmapwjbkhhjp3qggsq24ux.py
# Topologically Sorted Source Nodes: [attn_46], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn_46 => clone_244
# Graph fragment:
#   %clone_244 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_27,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_30 = async_compile.triton('triton_poi_fused_clone_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 197
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (768*x2) + (151296*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (197*y3)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2e/c2emcevgawli7xjm7kbacw6m57m5dojfickiqver3ia7o4fu2umu.py
# Topologically Sorted Source Nodes: [attn_47], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_47 => amax_10, div_10, exp_10, sub_94, sum_11
# Graph fragment:
#   %amax_10 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_474, [-1], True), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_474, %amax_10), kwargs = {})
#   %exp_10 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_94,), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_10, [-1], True), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_10, %sum_11), kwargs = {})
triton_per_fused__softmax_31 = async_compile.triton('triton_per_fused__softmax_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[128, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_31(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    rnumel = 197
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = xindex % 12
    x3 = (xindex // 12)
    tmp0 = tl.load(in_ptr0 + (r1 + (197*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(rmask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp6 / tmp10
    tl.store(out_ptr2 + (r1 + (197*x2) + (2368*x3)), tmp11, rmask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lk/clkyyhuq3nrz6fxjufhrgs4ome7rwm4khhi73lati6cct6v2fsqr.py
# Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   matmul_13 => bmm_13
# Graph fragment:
#   %bmm_13 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_475, %view_476), kwargs = {})
triton_poi_fused_bmm_32 = async_compile.triton('triton_poi_fused_bmm_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_32(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 197
    x1 = (xindex // 197)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (197*(x1 % 12)) + (2368*(x1 // 12))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2t/c2trhwh6hxpdfkwavrpovqjy5oow6p2vf4phicqwbheaebs4bc75.py
# Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_13 => clone_246
# Graph fragment:
#   %clone_246 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_29,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_33 = async_compile.triton('triton_poi_fused_clone_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 605184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 197
    x2 = (xindex // 6304) % 12
    x3 = (xindex // 75648)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (32*x2) + (768*x1) + (151296*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/34/c34oey6jzbhj3qpt3wglybsdwg3l2yhdsotodem52tlg2azz4qqr.py
# Topologically Sorted Source Nodes: [cls_embed_16, layer_norm_78], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cls_embed_16 => add_326
#   layer_norm_78 => add_327, add_328, mul_300, mul_301, rsqrt_78, sub_95, var_mean_78
# Graph fragment:
#   %add_326 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_35, %view_480), kwargs = {})
#   %var_mean_78 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_326, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_326, %getitem_361), kwargs = {})
#   %add_327 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_360, 1e-05), kwargs = {})
#   %rsqrt_78 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_327,), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %rsqrt_78), kwargs = {})
#   %mul_301 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_300, %arg234_1), kwargs = {})
#   %add_328 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_301, %arg235_1), kwargs = {})
triton_per_fused_add_native_layer_norm_34 = async_compile.triton('triton_per_fused_add_native_layer_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp16 = tl.load(in_out_ptr0 + (r1 + (384*x0)), rmask, other=0.0)
    tmp17 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp3, eviction_policy='evict_last', other=0.0)
    tmp5 = tmp0 >= tmp2
    tmp6 = tl.full([1], 197, tl.int64)
    tmp7 = tmp0 < tmp6
    tmp8 = tl.load(in_ptr1 + (r1 + (384*((-1) % 196)) + (75264*x0)), rmask & tmp5, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r1 + (384*((-1) % 196)) + (75264*x0)), rmask & tmp5, other=0.0)
    tmp10 = tl.load(in_ptr3 + (tl.broadcast_to(r1, [RBLOCK])), rmask & tmp5, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tmp15 = tl.where(tmp3, tmp4, tmp14)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 + tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tl.full([1], 384, tl.int32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 / tmp28
    tmp30 = tmp20 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tl.broadcast_to(tmp31, [RBLOCK])
    tmp34 = tl.where(rmask, tmp32, 0)
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp34, 0))
    tmp36 = tmp19 - tmp29
    tmp37 = 384.0
    tmp38 = tmp35 / tmp37
    tmp39 = 1e-05
    tmp40 = tmp38 + tmp39
    tmp41 = libdevice.rsqrt(tmp40)
    tmp42 = tmp36 * tmp41
    tmp44 = tmp42 * tmp43
    tmp46 = tmp44 + tmp45
    tl.store(in_out_ptr0 + (r1 + (384*x0)), tmp19, rmask)
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp46, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/gc/cgcuvq7kstz5bfuxxtpdwedt2hzniwmiekq7zs5lyvsxdlhklvdq.py
# Topologically Sorted Source Nodes: [x_429], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_429 => add_329, erf_38, mul_302, mul_303, mul_304
# Graph fragment:
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_482, 0.5), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_482, 0.7071067811865476), kwargs = {})
#   %erf_38 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_303,), kwargs = {})
#   %add_329 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_38, 1), kwargs = {})
#   %mul_304 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_302, %add_329), kwargs = {})
triton_poi_fused_gelu_35 = async_compile.triton('triton_poi_fused_gelu_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1152
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ov/covcyagmfaujlzxw7le3wplxta3gepztmfoyctxrc2laj4a3xwoi.py
# Topologically Sorted Source Nodes: [x_433, layer_norm_79], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_79 => add_331, add_332, mul_305, mul_306, rsqrt_79, sub_96, var_mean_79
#   x_433 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=4] = call_function[target=torch.ops.aten.cat.default](args = ([%add_330, %slice_40], 1), kwargs = {})
#   %var_mean_79 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %getitem_363), kwargs = {})
#   %add_331 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_362, 1e-05), kwargs = {})
#   %rsqrt_79 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_331,), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %rsqrt_79), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_305, %arg240_1), kwargs = {})
#   %add_332 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_306, %arg241_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_36 = async_compile.triton('triton_per_fused_cat_native_layer_norm_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 9, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp57 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tl.load(in_ptr8 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (384*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (r2 + (384*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 197, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.broadcast_to(1 + ((-1) + x0), [RBLOCK])
    tmp16 = tmp15 >= tmp1
    tmp17 = tmp15 < tmp3
    tmp18 = tmp17 & tmp12
    tmp19 = tl.load(in_ptr3 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp15 >= tmp3
    tmp21 = tmp15 < tmp13
    tmp22 = tmp20 & tmp12
    tmp23 = tl.load(in_ptr4 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp22, other=0.0)
    tmp24 = tl.load(in_ptr5 + (r2 + (384*(((-1) + x0) % 196)) + (75264*x1)), rmask & tmp22, other=0.0)
    tmp25 = tl.load(in_ptr6 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp22, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp22, tmp27, tmp28)
    tmp30 = tl.where(tmp17, tmp19, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp12, tmp30, tmp31)
    tmp33 = tl.where(tmp4, tmp11, tmp32)
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask, tmp34, 0)
    tmp37 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp39 = tl.where(rmask, tmp37, 0)
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp39, 0))
    tmp41 = tl.full([1], 384, tl.int32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 / tmp42
    tmp44 = tmp34 - tmp43
    tmp45 = tmp44 * tmp44
    tmp46 = tl.broadcast_to(tmp45, [RBLOCK])
    tmp48 = tl.where(rmask, tmp46, 0)
    tmp49 = triton_helpers.promote_to_tensor(tl.sum(tmp48, 0))
    tmp50 = tmp33 - tmp43
    tmp51 = 384.0
    tmp52 = tmp49 / tmp51
    tmp53 = 1e-05
    tmp54 = tmp52 + tmp53
    tmp55 = libdevice.rsqrt(tmp54)
    tmp56 = tmp50 * tmp55
    tmp58 = tmp56 * tmp57
    tmp60 = tmp58 + tmp59
    tl.store(out_ptr0 + (r2 + (384*x3)), tmp33, rmask)
    tl.store(out_ptr3 + (r2 + (384*x3)), tmp60, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sg/csgk3legpexxrmd3nnk47xi3nk4opotoz64q55z4ulxytkc4uege.py
# Topologically Sorted Source Nodes: [cls_embed_22, layer_norm_80], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   cls_embed_22 => add_333
#   layer_norm_80 => add_334, add_335, mul_308, mul_309, rsqrt_80, sub_98, var_mean_80
# Graph fragment:
#   %add_333 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_42, %view_499), kwargs = {})
#   %var_mean_80 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_333, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_333, %getitem_367), kwargs = {})
#   %add_334 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_366, 1e-05), kwargs = {})
#   %rsqrt_80 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_334,), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %rsqrt_80), kwargs = {})
#   %mul_309 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_308, %arg246_1), kwargs = {})
#   %add_335 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_309, %arg247_1), kwargs = {})
triton_per_fused_add_native_layer_norm_37 = async_compile.triton('triton_per_fused_add_native_layer_norm_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 8
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (75648*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (384*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp31, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w2/cw2iftebj7gddoi4rehjhivkmjy4genh3b2ppk7mxx4e6h4znj6y.py
# Topologically Sorted Source Nodes: [x_439, x_440], Original ATen: [aten.cat, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_439 => cat_5
#   x_440 => add_338, add_339, mul_313, mul_314, rsqrt_81, sub_99, var_mean_81
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_337, %slice_47], 1), kwargs = {})
#   %var_mean_81 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%cat_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %getitem_369), kwargs = {})
#   %add_338 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_368, 1e-05), kwargs = {})
#   %rsqrt_81 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_338,), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %rsqrt_81), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %arg252_1), kwargs = {})
#   %add_339 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %arg253_1), kwargs = {})
triton_per_fused_cat_native_layer_norm_38 = async_compile.triton('triton_per_fused_cat_native_layer_norm_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_native_layer_norm_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_native_layer_norm_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 1576
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    x0 = xindex % 197
    r2 = rindex
    x1 = (xindex // 197)
    x3 = xindex
    tmp44 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (75648*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (r2 + (384*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tl.load(in_ptr3 + (r2 + (384*x1)), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr4 + (tl.broadcast_to(r2, [RBLOCK])), rmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 197, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr0 + (384 + r2 + (384*((-1) + x0)) + (75648*x1)), rmask & tmp16, other=0.0)
    tmp20 = tl.where(tmp4, tmp15, tmp19)
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 384, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 384.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tl.store(out_ptr0 + (r2 + (384*x3)), tmp20, rmask)
    tl.store(out_ptr3 + (r2 + (384*x3)), tmp47, rmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qm/cqm2k6xclrgr5whs767ieskmf65akyxef7kl257y4m2xwxueaxxw.py
# Topologically Sorted Source Nodes: [aux_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   aux_1 => clone_257
# Graph fragment:
#   %clone_257 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_50,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_39 = async_compile.triton('triton_poi_fused_clone_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 75264
    x1 = (xindex // 75264)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (384 + x0 + (75648*x1)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hf/chfzdgr3am75ejhyzlkmrszkrwhom6xprjhw2bdjtgpm3hmm4hqo.py
# Topologically Sorted Source Nodes: [aux_1, max_2], Original ATen: [aten.add, aten.max]
# Source node to ATen node mapping:
#   aux_1 => add_340
#   max_2 => max_2
# Graph fragment:
#   %add_340 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_505, %arg257_1), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%add_340, 1), kwargs = {})
triton_red_fused_add_max_40 = async_compile.triton('triton_red_fused_add_max_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_max_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_max_40(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16000
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1000
    x4 = (xindex // 1000)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    x2 = (xindex // 2000)
    x5 = xindex % 2000
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1000*r3) + (98000*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = triton_helpers.maximum(_tmp4, tmp3)
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = triton_helpers.max2(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x5 + (2016*x2)), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/c6/cc6b4zhrqkgyjyz5rnotohyr37vb423qttela4pywmamfwgii6z5.py
# Topologically Sorted Source Nodes: [aux_1, max_2, mul_13, out_5], Original ATen: [aten.add, aten.max, aten.mul]
# Source node to ATen node mapping:
#   aux_1 => add_340
#   max_2 => max_2
#   mul_13 => mul_315
#   out_5 => add_341
# Graph fragment:
#   %add_340 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_505, %arg257_1), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%add_340, 1), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %arg255_1), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_370, 0.5), kwargs = {})
#   %add_341 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor, %mul_315), kwargs = {})
triton_per_fused_add_max_mul_41 = async_compile.triton('triton_per_fused_add_max_mul_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_mul_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_max_mul_41(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 1000
    x1 = (xindex // 1000)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1000*r2) + (2016*x1)), xmask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp7 = tmp5 + tmp6
    tmp8 = 0.5
    tmp9 = tmp4 * tmp8
    tmp10 = tmp7 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg12_1, (64, ), (1, ))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (192, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(arg17_1, (192, ), (1, ))
    assert_size_stride(arg18_1, (192, ), (1, ))
    assert_size_stride(arg19_1, (192, ), (1, ))
    assert_size_stride(arg20_1, (192, 192), (192, 1))
    assert_size_stride(arg21_1, (486, 192), (192, 1))
    assert_size_stride(arg22_1, (486, ), (1, ))
    assert_size_stride(arg23_1, (192, 192), (192, 1))
    assert_size_stride(arg24_1, (192, ), (1, ))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (576, 192), (192, 1))
    assert_size_stride(arg28_1, (576, ), (1, ))
    assert_size_stride(arg29_1, (192, 576), (576, 1))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, 192), (192, 1))
    assert_size_stride(arg34_1, (486, 192), (192, 1))
    assert_size_stride(arg35_1, (486, ), (1, ))
    assert_size_stride(arg36_1, (192, 192), (192, 1))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (576, 192), (192, 1))
    assert_size_stride(arg41_1, (576, ), (1, ))
    assert_size_stride(arg42_1, (192, 576), (576, 1))
    assert_size_stride(arg43_1, (192, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, 192), (192, 1))
    assert_size_stride(arg47_1, (486, 192), (192, 1))
    assert_size_stride(arg48_1, (486, ), (1, ))
    assert_size_stride(arg49_1, (192, 192), (192, 1))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (192, ), (1, ))
    assert_size_stride(arg53_1, (576, 192), (192, 1))
    assert_size_stride(arg54_1, (576, ), (1, ))
    assert_size_stride(arg55_1, (192, 576), (576, 1))
    assert_size_stride(arg56_1, (192, ), (1, ))
    assert_size_stride(arg57_1, (192, ), (1, ))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, 192), (192, 1))
    assert_size_stride(arg60_1, (486, 192), (192, 1))
    assert_size_stride(arg61_1, (486, ), (1, ))
    assert_size_stride(arg62_1, (192, 192), (192, 1))
    assert_size_stride(arg63_1, (192, ), (1, ))
    assert_size_stride(arg64_1, (192, ), (1, ))
    assert_size_stride(arg65_1, (192, ), (1, ))
    assert_size_stride(arg66_1, (576, 192), (192, 1))
    assert_size_stride(arg67_1, (576, ), (1, ))
    assert_size_stride(arg68_1, (192, 576), (576, 1))
    assert_size_stride(arg69_1, (192, ), (1, ))
    assert_size_stride(arg70_1, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (1, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (1152, 384), (384, 1))
    assert_size_stride(arg76_1, (384, 384), (384, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (1152, 384), (384, 1))
    assert_size_stride(arg81_1, (1152, ), (1, ))
    assert_size_stride(arg82_1, (384, 1152), (1152, 1))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (384, ), (1, ))
    assert_size_stride(arg86_1, (1152, 384), (384, 1))
    assert_size_stride(arg87_1, (384, 384), (384, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (1152, 384), (384, 1))
    assert_size_stride(arg92_1, (1152, ), (1, ))
    assert_size_stride(arg93_1, (384, 1152), (1152, 1))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (384, ), (1, ))
    assert_size_stride(arg97_1, (1152, 384), (384, 1))
    assert_size_stride(arg98_1, (384, 384), (384, 1))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (1152, 384), (384, 1))
    assert_size_stride(arg103_1, (1152, ), (1, ))
    assert_size_stride(arg104_1, (384, 1152), (1152, 1))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (1152, 384), (384, 1))
    assert_size_stride(arg109_1, (384, 384), (384, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (1152, 384), (384, 1))
    assert_size_stride(arg114_1, (1152, ), (1, ))
    assert_size_stride(arg115_1, (384, 1152), (1152, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (384, ), (1, ))
    assert_size_stride(arg119_1, (1152, 384), (384, 1))
    assert_size_stride(arg120_1, (384, 384), (384, 1))
    assert_size_stride(arg121_1, (384, ), (1, ))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (1152, 384), (384, 1))
    assert_size_stride(arg125_1, (1152, ), (1, ))
    assert_size_stride(arg126_1, (384, 1152), (1152, 1))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, ), (1, ))
    assert_size_stride(arg130_1, (1152, 384), (384, 1))
    assert_size_stride(arg131_1, (384, 384), (384, 1))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (384, ), (1, ))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (1152, 384), (384, 1))
    assert_size_stride(arg136_1, (1152, ), (1, ))
    assert_size_stride(arg137_1, (384, 1152), (1152, 1))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (1152, 384), (384, 1))
    assert_size_stride(arg142_1, (384, 384), (384, 1))
    assert_size_stride(arg143_1, (384, ), (1, ))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (1152, 384), (384, 1))
    assert_size_stride(arg147_1, (1152, ), (1, ))
    assert_size_stride(arg148_1, (384, 1152), (1152, 1))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (1152, 384), (384, 1))
    assert_size_stride(arg153_1, (384, 384), (384, 1))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (1152, 384), (384, 1))
    assert_size_stride(arg158_1, (1152, ), (1, ))
    assert_size_stride(arg159_1, (384, 1152), (1152, 1))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (1152, 384), (384, 1))
    assert_size_stride(arg164_1, (384, 384), (384, 1))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (1152, 384), (384, 1))
    assert_size_stride(arg169_1, (1152, ), (1, ))
    assert_size_stride(arg170_1, (384, 1152), (1152, 1))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (1152, 384), (384, 1))
    assert_size_stride(arg175_1, (384, 384), (384, 1))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (1152, 384), (384, 1))
    assert_size_stride(arg180_1, (1152, ), (1, ))
    assert_size_stride(arg181_1, (384, 1152), (1152, 1))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (384, ), (1, ))
    assert_size_stride(arg185_1, (1152, 384), (384, 1))
    assert_size_stride(arg186_1, (384, 384), (384, 1))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (1152, 384), (384, 1))
    assert_size_stride(arg191_1, (1152, ), (1, ))
    assert_size_stride(arg192_1, (384, 1152), (1152, 1))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (1152, 384), (384, 1))
    assert_size_stride(arg197_1, (384, 384), (384, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (384, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (1152, 384), (384, 1))
    assert_size_stride(arg202_1, (1152, ), (1, ))
    assert_size_stride(arg203_1, (384, 1152), (1152, 1))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (1152, 384), (384, 1))
    assert_size_stride(arg208_1, (384, 384), (384, 1))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (384, ), (1, ))
    assert_size_stride(arg212_1, (1152, 384), (384, 1))
    assert_size_stride(arg213_1, (1152, ), (1, ))
    assert_size_stride(arg214_1, (384, 1152), (1152, 1))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (384, ), (1, ))
    assert_size_stride(arg218_1, (1152, 384), (384, 1))
    assert_size_stride(arg219_1, (384, 384), (384, 1))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (1152, 384), (384, 1))
    assert_size_stride(arg224_1, (1152, ), (1, ))
    assert_size_stride(arg225_1, (384, 1152), (1152, 1))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (768, 384), (384, 1))
    assert_size_stride(arg231_1, (384, 384), (384, 1))
    assert_size_stride(arg232_1, (384, 384), (384, 1))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (384, ), (1, ))
    assert_size_stride(arg236_1, (1152, 384), (384, 1))
    assert_size_stride(arg237_1, (1152, ), (1, ))
    assert_size_stride(arg238_1, (384, 1152), (1152, 1))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, ), (1, ))
    assert_size_stride(arg242_1, (768, 384), (384, 1))
    assert_size_stride(arg243_1, (384, 384), (384, 1))
    assert_size_stride(arg244_1, (384, 384), (384, 1))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (384, ), (1, ))
    assert_size_stride(arg248_1, (1152, 384), (384, 1))
    assert_size_stride(arg249_1, (1152, ), (1, ))
    assert_size_stride(arg250_1, (384, 1152), (1152, 1))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (1000, 384), (384, 1))
    assert_size_stride(arg255_1, (1000, ), (1, ))
    assert_size_stride(arg256_1, (1000, 384), (384, 1))
    assert_size_stride(arg257_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg0_1
        del arg1_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg2_1, arg3_1, arg4_1, arg5_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf2 = extern_kernels.convolution(buf1, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg6_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf3, arg7_1, arg8_1, arg9_1, arg10_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [input_14, input_15, input_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg11_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf5, arg12_1, arg13_1, arg14_1, arg15_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [input_17, input_18, x_221], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg16_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 192, 28, 28), (150528, 784, 28, 1))
        del arg16_1
        del buf5
        buf7 = empty_strided_cuda((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), torch.float32)
        buf8 = empty_strided_cuda((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), torch.float32)
        buf9 = empty_strided_cuda((8, 28, 28, 1, 2), (1568, 28, 1, 12544, 784), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_1.run(buf6, arg17_1, buf7, buf8, buf9, 12544, 96, grid=grid(12544), stream=stream0)
        buf10 = empty_strided_cuda((8, 28, 28, 1), (784, 28, 1, 6272), torch.float32)
        buf11 = empty_strided_cuda((8, 28, 28, 1), (784, 28, 1, 6272), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_2.run(buf7, buf8, buf9, buf10, buf11, 6272, 2, grid=grid(6272), stream=stream0)
        buf13 = empty_strided_cuda((8, 28, 28, 192), (150528, 28, 1, 784), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm_41], Original ATen: [aten.native_layer_norm]
        triton_poi_fused_native_layer_norm_3.run(buf6, arg17_1, buf10, buf11, arg18_1, arg19_1, buf13, 1204224, grid=grid(1204224), stream=stream0)
        del arg18_1
        del arg19_1
        buf14 = empty_strided_cuda((8, 192, 14, 14), (37632, 1, 2688, 192), torch.float32)
        # Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_4.run(buf13, buf14, 1536, 196, grid=grid(1536, 196), stream=stream0)
        buf15 = empty_strided_cuda((1568, 486), (486, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf14, (1568, 192), (192, 1), 0), reinterpret_tensor(arg21_1, (192, 486), (1, 192), 0), out=buf15)
        del arg21_1
        buf20 = empty_strided_cuda((8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_28, attn_29], Original ATen: [aten.mul, aten._softmax]
        triton_per_fused__softmax_mul_5.run(buf15, arg22_1, buf20, 84672, 9, grid=grid(84672), stream=stream0)
        del arg22_1
        buf18 = empty_strided_cuda((6272, 192), (1, 6272), torch.float32)
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.mm]
        triton_poi_fused_mm_6.run(buf13, buf18, 1204224, grid=grid(1204224), stream=stream0)
        buf19 = reinterpret_tensor(buf13, (6272, 192), (192, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.mm]
        extern_kernels.mm(buf18, reinterpret_tensor(arg20_1, (192, 192), (1, 192), 0), out=buf19)
        del arg20_1
        buf21 = empty_strided_cuda((8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf19, buf21, 2709504, grid=grid(2709504), stream=stream0)
        buf22 = empty_strided_cuda((9408, 9, 32), (288, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf21, (9408, 9, 32), (288, 32, 1), 0), out=buf22)
        buf23 = empty_strided_cuda((8, 192, 30, 30), (172800, 900, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_8.run(buf23, 1382400, grid=grid(1382400), stream=stream0)
        buf25 = empty_strided_cuda((8, 192, 3, 14, 3, 14), (344064, 1792, 588, 14, 196, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_224, x_223], Original ATen: [aten.col2im, aten.clone]
        triton_poi_fused_clone_col2im_9.run(buf22, buf25, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        buf26 = empty_strided_cuda((3, 14), (14, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf26, 42, grid=grid(42), stream=stream0)
        buf27 = empty_strided_cuda((3, 14), (14, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_224], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf27, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf23, [None, None, reinterpret_tensor(buf26, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf27], buf25, True)
        buf29 = reinterpret_tensor(buf19, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf23, buf29, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf30 = reinterpret_tensor(buf18, (6272, 192), (192, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_225], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (6272, 192), (192, 1), 0), reinterpret_tensor(arg23_1, (192, 192), (1, 192), 0), out=buf30)
        del arg23_1
        buf31 = reinterpret_tensor(buf9, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf9  # reuse
        buf32 = reinterpret_tensor(buf8, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf8  # reuse
        buf33 = reinterpret_tensor(buf7, (8, 28, 28, 1, 2), (1568, 56, 2, 12544, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_225, x_227, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_12.run(buf6, arg17_1, buf30, arg24_1, buf31, buf32, buf33, 12544, 96, grid=grid(12544), stream=stream0)
        buf34 = buf11; del buf11  # reuse
        buf35 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_225, x_227, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_13.run(buf31, buf32, buf33, buf34, buf35, 6272, 2, grid=grid(6272), stream=stream0)
        del buf31
        del buf32
        del buf33
        buf37 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_225, x_227, layer_norm_42], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_14.run(buf6, arg17_1, buf30, arg24_1, buf34, buf35, arg25_1, arg26_1, buf37, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg25_1
        del arg26_1
        del buf34
        del buf35
        buf38 = empty_strided_cuda((6272, 576), (576, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf37, (6272, 192), (192, 1), 0), reinterpret_tensor(arg27_1, (192, 576), (1, 192), 0), out=buf38)
        del arg27_1
        buf39 = reinterpret_tensor(buf38, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf39, arg28_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg28_1
        buf40 = reinterpret_tensor(buf37, (6272, 192), (192, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf39, (6272, 576), (576, 1), 0), reinterpret_tensor(arg29_1, (576, 192), (1, 576), 0), out=buf40)
        del arg29_1
        buf41 = reinterpret_tensor(buf30, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf30  # reuse
        buf45 = empty_strided_cuda((8, 28, 28, 192), (150528, 5376, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_225, x_227, x_233, layer_norm_43], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_16.run(buf41, buf6, arg17_1, arg24_1, buf40, arg30_1, arg31_1, arg32_1, buf45, 6272, 192, grid=grid(6272), stream=stream0)
        del arg17_1
        del arg24_1
        del arg30_1
        del arg31_1
        del arg32_1
        buf46 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_17.run(buf45, buf46, 301056, grid=grid(301056), stream=stream0)
        buf47 = reinterpret_tensor(buf20, (1568, 486), (486, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf46, (1568, 192), (192, 1), 0), reinterpret_tensor(arg34_1, (192, 486), (1, 192), 0), out=buf47)
        del arg34_1
        buf51 = reinterpret_tensor(buf15, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [attn_33, attn_34], Original ATen: [aten.mul, aten._softmax]
        triton_per_fused__softmax_mul_5.run(buf47, arg35_1, buf51, 84672, 9, grid=grid(84672), stream=stream0)
        del arg35_1
        buf50 = reinterpret_tensor(buf6, (6272, 192), (192, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (6272, 192), (192, 1), 0), reinterpret_tensor(arg33_1, (192, 192), (1, 192), 0), out=buf50)
        del arg33_1
        buf52 = reinterpret_tensor(buf22, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf50, buf52, 2709504, grid=grid(2709504), stream=stream0)
        buf53 = reinterpret_tensor(buf21, (9408, 9, 32), (288, 32, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf52, (9408, 9, 32), (288, 32, 1), 0), out=buf53)
        buf54 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_8.run(buf54, 1382400, grid=grid(1382400), stream=stream0)
        buf56 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_234], Original ATen: [aten.col2im, aten.clone]
        triton_poi_fused_clone_col2im_9.run(buf53, buf56, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        buf57 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf57, 42, grid=grid(42), stream=stream0)
        buf58 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_235], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf58, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf54, [None, None, reinterpret_tensor(buf57, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf58], buf56, True)
        buf60 = reinterpret_tensor(buf50, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf54, buf60, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf61 = reinterpret_tensor(buf45, (6272, 192), (192, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (6272, 192), (192, 1), 0), reinterpret_tensor(arg36_1, (192, 192), (1, 192), 0), out=buf61)
        del arg36_1
        buf65 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_236, x_238, layer_norm_44], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf41, buf61, arg37_1, arg38_1, arg39_1, buf65, 6272, 192, grid=grid(6272), stream=stream0)
        del arg38_1
        del arg39_1
        buf66 = reinterpret_tensor(buf39, (6272, 576), (576, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf65, (6272, 192), (192, 1), 0), reinterpret_tensor(arg40_1, (192, 576), (1, 192), 0), out=buf66)
        del arg40_1
        buf67 = reinterpret_tensor(buf66, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf67, arg41_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg41_1
        buf68 = reinterpret_tensor(buf65, (6272, 192), (192, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf67, (6272, 576), (576, 1), 0), reinterpret_tensor(arg42_1, (576, 192), (1, 576), 0), out=buf68)
        del arg42_1
        buf69 = reinterpret_tensor(buf68, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf68  # reuse
        buf73 = reinterpret_tensor(buf40, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_236, x_238, x_244, layer_norm_45], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_19.run(buf69, buf41, buf61, arg37_1, arg43_1, arg44_1, arg45_1, buf73, 6272, 192, grid=grid(6272), stream=stream0)
        del arg37_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf74 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_6], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_17.run(buf73, buf74, 301056, grid=grid(301056), stream=stream0)
        buf75 = reinterpret_tensor(buf51, (1568, 486), (486, 1), 0); del buf51  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf74, (1568, 192), (192, 1), 0), reinterpret_tensor(arg47_1, (192, 486), (1, 192), 0), out=buf75)
        del arg47_1
        buf79 = reinterpret_tensor(buf47, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [attn_38, attn_39], Original ATen: [aten.mul, aten._softmax]
        triton_per_fused__softmax_mul_5.run(buf75, arg48_1, buf79, 84672, 9, grid=grid(84672), stream=stream0)
        del arg48_1
        buf78 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (6272, 192), (192, 1), 0), reinterpret_tensor(arg46_1, (192, 192), (1, 192), 0), out=buf78)
        del arg46_1
        buf80 = reinterpret_tensor(buf53, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf78, buf80, 2709504, grid=grid(2709504), stream=stream0)
        buf81 = reinterpret_tensor(buf52, (9408, 9, 32), (288, 32, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf80, (9408, 9, 32), (288, 32, 1), 0), out=buf81)
        buf82 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_8.run(buf82, 1382400, grid=grid(1382400), stream=stream0)
        buf84 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_246, x_245], Original ATen: [aten.col2im, aten.clone]
        triton_poi_fused_clone_col2im_9.run(buf81, buf84, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        buf85 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf85, 42, grid=grid(42), stream=stream0)
        buf86 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf86, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf82, [None, None, reinterpret_tensor(buf85, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf86], buf84, True)
        buf88 = reinterpret_tensor(buf78, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_247], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf82, buf88, 6272, 192, grid=grid(6272, 192), stream=stream0)
        buf89 = reinterpret_tensor(buf73, (6272, 192), (192, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_247], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (6272, 192), (192, 1), 0), reinterpret_tensor(arg49_1, (192, 192), (1, 192), 0), out=buf89)
        del arg49_1
        buf93 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_249, layer_norm_46], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf69, buf89, arg50_1, arg51_1, arg52_1, buf93, 6272, 192, grid=grid(6272), stream=stream0)
        del arg51_1
        del arg52_1
        buf94 = reinterpret_tensor(buf67, (6272, 576), (576, 1), 0); del buf67  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf93, (6272, 192), (192, 1), 0), reinterpret_tensor(arg53_1, (192, 576), (1, 192), 0), out=buf94)
        del arg53_1
        buf95 = reinterpret_tensor(buf94, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf95, arg54_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg54_1
        buf96 = reinterpret_tensor(buf93, (6272, 192), (192, 1), 0); del buf93  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf95, (6272, 576), (576, 1), 0), reinterpret_tensor(arg55_1, (576, 192), (1, 576), 0), out=buf96)
        del arg55_1
        buf97 = reinterpret_tensor(buf96, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf96  # reuse
        buf101 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_247, x_249, x_255, layer_norm_47], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_19.run(buf97, buf69, buf89, arg50_1, arg56_1, arg57_1, arg58_1, buf101, 6272, 192, grid=grid(6272), stream=stream0)
        del arg50_1
        del arg56_1
        del arg57_1
        del arg58_1
        buf102 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [avg_pool2d_7], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_17.run(buf101, buf102, 301056, grid=grid(301056), stream=stream0)
        buf103 = reinterpret_tensor(buf79, (1568, 486), (486, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf102, (1568, 192), (192, 1), 0), reinterpret_tensor(arg60_1, (192, 486), (1, 192), 0), out=buf103)
        del arg60_1
        del buf102
        buf107 = reinterpret_tensor(buf75, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [attn_43, attn_44], Original ATen: [aten.mul, aten._softmax]
        triton_per_fused__softmax_mul_5.run(buf103, arg61_1, buf107, 84672, 9, grid=grid(84672), stream=stream0)
        del arg61_1
        del buf103
        buf106 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (6272, 192), (192, 1), 0), reinterpret_tensor(arg59_1, (192, 192), (1, 192), 0), out=buf106)
        del arg59_1
        buf108 = reinterpret_tensor(buf81, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf106, buf108, 2709504, grid=grid(2709504), stream=stream0)
        buf109 = reinterpret_tensor(buf80, (9408, 9, 32), (288, 32, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf107, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf108, (9408, 9, 32), (288, 32, 1), 0), out=buf109)
        del buf107
        del buf108
        buf110 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_8.run(buf110, 1382400, grid=grid(1382400), stream=stream0)
        buf112 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_257, x_256], Original ATen: [aten.col2im, aten.clone]
        triton_poi_fused_clone_col2im_9.run(buf109, buf112, 1536, 1764, grid=grid(1536, 1764), stream=stream0)
        del buf109
        buf113 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf113, 42, grid=grid(42), stream=stream0)
        buf114 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.col2im]
        triton_poi_fused_col2im_10.run(buf114, 42, grid=grid(42), stream=stream0)
        aten.index_put_(buf110, [None, None, reinterpret_tensor(buf113, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf114], buf112, True)
        del buf112
        del buf113
        del buf114
        buf116 = reinterpret_tensor(buf106, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_258], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf110, buf116, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del buf110
        buf117 = reinterpret_tensor(buf101, (6272, 192), (192, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_258], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (6272, 192), (192, 1), 0), reinterpret_tensor(arg62_1, (192, 192), (1, 192), 0), out=buf117)
        del arg62_1
        buf121 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_258, x_260, layer_norm_48], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_18.run(buf97, buf117, arg63_1, arg64_1, arg65_1, buf121, 6272, 192, grid=grid(6272), stream=stream0)
        del arg64_1
        del arg65_1
        buf122 = reinterpret_tensor(buf95, (6272, 576), (576, 1), 0); del buf95  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf121, (6272, 192), (192, 1), 0), reinterpret_tensor(arg66_1, (192, 576), (1, 192), 0), out=buf122)
        del arg66_1
        buf123 = reinterpret_tensor(buf122, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_15.run(buf123, arg67_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg67_1
        buf124 = reinterpret_tensor(buf121, (6272, 192), (192, 1), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf123, (6272, 576), (576, 1), 0), reinterpret_tensor(arg68_1, (576, 192), (1, 576), 0), out=buf124)
        del arg68_1
        del buf123
        buf125 = reinterpret_tensor(buf69, (8, 192, 28, 28), (150528, 784, 28, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_268], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf97, buf117, arg63_1, buf124, arg69_1, buf125, 6272, 192, grid=grid(6272, 192), stream=stream0)
        del arg63_1
        del arg69_1
        del buf117
        del buf124
        del buf97
        # Topologically Sorted Source Nodes: [x_268], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, arg70_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 384, 14, 14), (75264, 196, 14, 1))
        del arg70_1
        del buf125
        buf127 = empty_strided_cuda((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), torch.float32)
        buf128 = empty_strided_cuda((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), torch.float32)
        buf129 = empty_strided_cuda((8, 14, 14, 1, 3), (588, 42, 3, 4704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_270, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_21.run(buf126, arg71_1, arg72_1, buf127, buf128, buf129, 4704, 128, grid=grid(4704), stream=stream0)
        buf130 = empty_strided_cuda((8, 14, 14, 1), (196, 14, 1, 1568), torch.float32)
        buf131 = empty_strided_cuda((8, 14, 14, 1), (196, 14, 1, 1568), torch.float32)
        # Topologically Sorted Source Nodes: [x_270, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_22.run(buf127, buf128, buf129, buf130, buf131, 1568, 3, grid=grid(1568), stream=stream0)
        del buf127
        del buf128
        del buf129
        buf133 = empty_strided_cuda((8, 14, 14, 384), (75264, 5376, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_270, layer_norm_49], Original ATen: [aten.add, aten.native_layer_norm]
        triton_poi_fused_add_native_layer_norm_23.run(buf126, arg71_1, arg72_1, buf130, buf131, arg73_1, arg74_1, buf133, 1568, 384, grid=grid(1568, 384), stream=stream0)
        del arg73_1
        del arg74_1
        del buf130
        del buf131
        buf134 = empty_strided_cuda((1568, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_108], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf133, (1568, 384), (384, 1), 0), reinterpret_tensor(arg75_1, (384, 1152), (1, 384), 0), out=buf134)
        del arg75_1
        # Topologically Sorted Source Nodes: [x_272], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf135 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf134, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf134, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf134, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf136 = buf135[0]
        del buf135
        buf140 = reinterpret_tensor(buf133, (1568, 384), (384, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf136, (1568, 384), (384, 1), 0), reinterpret_tensor(arg76_1, (384, 384), (1, 384), 0), out=buf140)
        del arg76_1
        buf141 = reinterpret_tensor(buf140, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf140  # reuse
        buf145 = reinterpret_tensor(buf136, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_270, x_276, layer_norm_50], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_24.run(buf141, buf126, arg71_1, arg72_1, arg77_1, arg78_1, arg79_1, buf145, 1568, 384, grid=grid(1568), stream=stream0)
        del arg71_1
        del arg72_1
        del arg77_1
        del arg78_1
        del arg79_1
        buf146 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf145, (1568, 384), (384, 1), 0), reinterpret_tensor(arg80_1, (384, 1152), (1, 384), 0), out=buf146)
        del arg80_1
        buf147 = reinterpret_tensor(buf146, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf147, arg81_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg81_1
        buf148 = reinterpret_tensor(buf145, (1568, 384), (384, 1), 0); del buf145  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf147, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg82_1, (1152, 384), (1, 1152), 0), out=buf148)
        del arg82_1
        buf152 = reinterpret_tensor(buf126, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_282, layer_norm_51], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf141, buf148, arg83_1, arg84_1, arg85_1, buf152, 1568, 384, grid=grid(1568), stream=stream0)
        del arg84_1
        del arg85_1
        buf153 = reinterpret_tensor(buf147, (1568, 1152), (1152, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf152, (1568, 384), (384, 1), 0), reinterpret_tensor(arg86_1, (384, 1152), (1, 384), 0), out=buf153)
        del arg86_1
        # Topologically Sorted Source Nodes: [x_283], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf154 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf153, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf153, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf153, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf155 = buf154[0]
        del buf154
        buf159 = reinterpret_tensor(buf152, (1568, 384), (384, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf155, (1568, 384), (384, 1), 0), reinterpret_tensor(arg87_1, (384, 384), (1, 384), 0), out=buf159)
        del arg87_1
        buf160 = reinterpret_tensor(buf159, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf159  # reuse
        buf164 = reinterpret_tensor(buf155, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_282, x_287, layer_norm_52], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf160, buf141, buf148, arg83_1, arg88_1, arg89_1, arg90_1, buf164, 1568, 384, grid=grid(1568), stream=stream0)
        del arg83_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf141
        buf165 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf164, (1568, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 1152), (1, 384), 0), out=buf165)
        del arg91_1
        buf166 = reinterpret_tensor(buf165, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_289], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf166, arg92_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg92_1
        buf167 = reinterpret_tensor(buf164, (1568, 384), (384, 1), 0); del buf164  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf166, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg93_1, (1152, 384), (1, 1152), 0), out=buf167)
        del arg93_1
        buf171 = reinterpret_tensor(buf148, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_293, layer_norm_53], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf160, buf167, arg94_1, arg95_1, arg96_1, buf171, 1568, 384, grid=grid(1568), stream=stream0)
        del arg95_1
        del arg96_1
        buf172 = reinterpret_tensor(buf166, (1568, 1152), (1152, 1), 0); del buf166  # reuse
        # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf171, (1568, 384), (384, 1), 0), reinterpret_tensor(arg97_1, (384, 1152), (1, 384), 0), out=buf172)
        del arg97_1
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf173 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf172, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf172, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf172, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf174 = buf173[0]
        del buf173
        buf178 = reinterpret_tensor(buf171, (1568, 384), (384, 1), 0); del buf171  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf174, (1568, 384), (384, 1), 0), reinterpret_tensor(arg98_1, (384, 384), (1, 384), 0), out=buf178)
        del arg98_1
        buf179 = reinterpret_tensor(buf178, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf178  # reuse
        buf183 = reinterpret_tensor(buf174, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_293, x_298, layer_norm_54], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf179, buf160, buf167, arg94_1, arg99_1, arg100_1, arg101_1, buf183, 1568, 384, grid=grid(1568), stream=stream0)
        del arg100_1
        del arg101_1
        del arg94_1
        del arg99_1
        del buf160
        buf184 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf183, (1568, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 1152), (1, 384), 0), out=buf184)
        del arg102_1
        buf185 = reinterpret_tensor(buf184, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_300], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf185, arg103_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg103_1
        buf186 = reinterpret_tensor(buf183, (1568, 384), (384, 1), 0); del buf183  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf185, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg104_1, (1152, 384), (1, 1152), 0), out=buf186)
        del arg104_1
        buf190 = reinterpret_tensor(buf167, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_304, layer_norm_55], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf179, buf186, arg105_1, arg106_1, arg107_1, buf190, 1568, 384, grid=grid(1568), stream=stream0)
        del arg106_1
        del arg107_1
        buf191 = reinterpret_tensor(buf185, (1568, 1152), (1152, 1), 0); del buf185  # reuse
        # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf190, (1568, 384), (384, 1), 0), reinterpret_tensor(arg108_1, (384, 1152), (1, 384), 0), out=buf191)
        del arg108_1
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf192 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf191, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf191, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf191, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf193 = buf192[0]
        del buf192
        buf197 = reinterpret_tensor(buf190, (1568, 384), (384, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf193, (1568, 384), (384, 1), 0), reinterpret_tensor(arg109_1, (384, 384), (1, 384), 0), out=buf197)
        del arg109_1
        buf198 = reinterpret_tensor(buf197, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf197  # reuse
        buf202 = reinterpret_tensor(buf193, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_304, x_309, layer_norm_56], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf198, buf179, buf186, arg105_1, arg110_1, arg111_1, arg112_1, buf202, 1568, 384, grid=grid(1568), stream=stream0)
        del arg105_1
        del arg110_1
        del arg111_1
        del arg112_1
        del buf179
        buf203 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf202, (1568, 384), (384, 1), 0), reinterpret_tensor(arg113_1, (384, 1152), (1, 384), 0), out=buf203)
        del arg113_1
        buf204 = reinterpret_tensor(buf203, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_311], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf204, arg114_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg114_1
        buf205 = reinterpret_tensor(buf202, (1568, 384), (384, 1), 0); del buf202  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf204, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg115_1, (1152, 384), (1, 1152), 0), out=buf205)
        del arg115_1
        buf209 = reinterpret_tensor(buf186, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_315, layer_norm_57], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf198, buf205, arg116_1, arg117_1, arg118_1, buf209, 1568, 384, grid=grid(1568), stream=stream0)
        del arg117_1
        del arg118_1
        buf210 = reinterpret_tensor(buf204, (1568, 1152), (1152, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf209, (1568, 384), (384, 1), 0), reinterpret_tensor(arg119_1, (384, 1152), (1, 384), 0), out=buf210)
        del arg119_1
        # Topologically Sorted Source Nodes: [x_316], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf211 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf210, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf210, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf210, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf212 = buf211[0]
        del buf211
        buf216 = reinterpret_tensor(buf209, (1568, 384), (384, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf212, (1568, 384), (384, 1), 0), reinterpret_tensor(arg120_1, (384, 384), (1, 384), 0), out=buf216)
        del arg120_1
        buf217 = reinterpret_tensor(buf216, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf216  # reuse
        buf221 = reinterpret_tensor(buf212, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf212  # reuse
        # Topologically Sorted Source Nodes: [x_315, x_320, layer_norm_58], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf217, buf198, buf205, arg116_1, arg121_1, arg122_1, arg123_1, buf221, 1568, 384, grid=grid(1568), stream=stream0)
        del arg116_1
        del arg121_1
        del arg122_1
        del arg123_1
        del buf198
        buf222 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf221, (1568, 384), (384, 1), 0), reinterpret_tensor(arg124_1, (384, 1152), (1, 384), 0), out=buf222)
        del arg124_1
        buf223 = reinterpret_tensor(buf222, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_322], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf223, arg125_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg125_1
        buf224 = reinterpret_tensor(buf221, (1568, 384), (384, 1), 0); del buf221  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf223, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg126_1, (1152, 384), (1, 1152), 0), out=buf224)
        del arg126_1
        buf228 = reinterpret_tensor(buf205, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_326, layer_norm_59], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf217, buf224, arg127_1, arg128_1, arg129_1, buf228, 1568, 384, grid=grid(1568), stream=stream0)
        del arg128_1
        del arg129_1
        buf229 = reinterpret_tensor(buf223, (1568, 1152), (1152, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (1568, 384), (384, 1), 0), reinterpret_tensor(arg130_1, (384, 1152), (1, 384), 0), out=buf229)
        del arg130_1
        # Topologically Sorted Source Nodes: [x_327], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf230 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf229, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf229, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf229, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf231 = buf230[0]
        del buf230
        buf235 = reinterpret_tensor(buf228, (1568, 384), (384, 1), 0); del buf228  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf231, (1568, 384), (384, 1), 0), reinterpret_tensor(arg131_1, (384, 384), (1, 384), 0), out=buf235)
        del arg131_1
        buf236 = reinterpret_tensor(buf235, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf235  # reuse
        buf240 = reinterpret_tensor(buf231, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_326, x_331, layer_norm_60], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf236, buf217, buf224, arg127_1, arg132_1, arg133_1, arg134_1, buf240, 1568, 384, grid=grid(1568), stream=stream0)
        del arg127_1
        del arg132_1
        del arg133_1
        del arg134_1
        del buf217
        buf241 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (1568, 384), (384, 1), 0), reinterpret_tensor(arg135_1, (384, 1152), (1, 384), 0), out=buf241)
        del arg135_1
        buf242 = reinterpret_tensor(buf241, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_333], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf242, arg136_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg136_1
        buf243 = reinterpret_tensor(buf240, (1568, 384), (384, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf242, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg137_1, (1152, 384), (1, 1152), 0), out=buf243)
        del arg137_1
        buf247 = reinterpret_tensor(buf224, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_337, layer_norm_61], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf236, buf243, arg138_1, arg139_1, arg140_1, buf247, 1568, 384, grid=grid(1568), stream=stream0)
        del arg139_1
        del arg140_1
        buf248 = reinterpret_tensor(buf242, (1568, 1152), (1152, 1), 0); del buf242  # reuse
        # Topologically Sorted Source Nodes: [linear_132], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf247, (1568, 384), (384, 1), 0), reinterpret_tensor(arg141_1, (384, 1152), (1, 384), 0), out=buf248)
        del arg141_1
        # Topologically Sorted Source Nodes: [x_338], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf249 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf248, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf248, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf248, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf250 = buf249[0]
        del buf249
        buf254 = reinterpret_tensor(buf247, (1568, 384), (384, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf250, (1568, 384), (384, 1), 0), reinterpret_tensor(arg142_1, (384, 384), (1, 384), 0), out=buf254)
        del arg142_1
        buf255 = reinterpret_tensor(buf254, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf254  # reuse
        buf259 = reinterpret_tensor(buf250, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf250  # reuse
        # Topologically Sorted Source Nodes: [x_337, x_342, layer_norm_62], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf255, buf236, buf243, arg138_1, arg143_1, arg144_1, arg145_1, buf259, 1568, 384, grid=grid(1568), stream=stream0)
        del arg138_1
        del arg143_1
        del arg144_1
        del arg145_1
        del buf236
        buf260 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf259, (1568, 384), (384, 1), 0), reinterpret_tensor(arg146_1, (384, 1152), (1, 384), 0), out=buf260)
        del arg146_1
        buf261 = reinterpret_tensor(buf260, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_344], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf261, arg147_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg147_1
        buf262 = reinterpret_tensor(buf259, (1568, 384), (384, 1), 0); del buf259  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf261, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg148_1, (1152, 384), (1, 1152), 0), out=buf262)
        del arg148_1
        buf266 = reinterpret_tensor(buf243, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_348, layer_norm_63], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf255, buf262, arg149_1, arg150_1, arg151_1, buf266, 1568, 384, grid=grid(1568), stream=stream0)
        del arg150_1
        del arg151_1
        buf267 = reinterpret_tensor(buf261, (1568, 1152), (1152, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [linear_136], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf266, (1568, 384), (384, 1), 0), reinterpret_tensor(arg152_1, (384, 1152), (1, 384), 0), out=buf267)
        del arg152_1
        # Topologically Sorted Source Nodes: [x_349], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf268 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf267, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf267, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf267, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf269 = buf268[0]
        del buf268
        buf273 = reinterpret_tensor(buf266, (1568, 384), (384, 1), 0); del buf266  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf269, (1568, 384), (384, 1), 0), reinterpret_tensor(arg153_1, (384, 384), (1, 384), 0), out=buf273)
        del arg153_1
        buf274 = reinterpret_tensor(buf273, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf273  # reuse
        buf278 = reinterpret_tensor(buf269, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_348, x_353, layer_norm_64], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf274, buf255, buf262, arg149_1, arg154_1, arg155_1, arg156_1, buf278, 1568, 384, grid=grid(1568), stream=stream0)
        del arg149_1
        del arg154_1
        del arg155_1
        del arg156_1
        del buf255
        buf279 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf278, (1568, 384), (384, 1), 0), reinterpret_tensor(arg157_1, (384, 1152), (1, 384), 0), out=buf279)
        del arg157_1
        buf280 = reinterpret_tensor(buf279, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [x_355], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf280, arg158_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg158_1
        buf281 = reinterpret_tensor(buf278, (1568, 384), (384, 1), 0); del buf278  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf280, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg159_1, (1152, 384), (1, 1152), 0), out=buf281)
        del arg159_1
        buf285 = reinterpret_tensor(buf262, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_359, layer_norm_65], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf274, buf281, arg160_1, arg161_1, arg162_1, buf285, 1568, 384, grid=grid(1568), stream=stream0)
        del arg161_1
        del arg162_1
        buf286 = reinterpret_tensor(buf280, (1568, 1152), (1152, 1), 0); del buf280  # reuse
        # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (1568, 384), (384, 1), 0), reinterpret_tensor(arg163_1, (384, 1152), (1, 384), 0), out=buf286)
        del arg163_1
        # Topologically Sorted Source Nodes: [x_360], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf287 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf286, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf286, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf286, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf288 = buf287[0]
        del buf287
        buf292 = reinterpret_tensor(buf285, (1568, 384), (384, 1), 0); del buf285  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf288, (1568, 384), (384, 1), 0), reinterpret_tensor(arg164_1, (384, 384), (1, 384), 0), out=buf292)
        del arg164_1
        buf293 = reinterpret_tensor(buf292, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf292  # reuse
        buf297 = reinterpret_tensor(buf288, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_359, x_364, layer_norm_66], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf293, buf274, buf281, arg160_1, arg165_1, arg166_1, arg167_1, buf297, 1568, 384, grid=grid(1568), stream=stream0)
        del arg160_1
        del arg165_1
        del arg166_1
        del arg167_1
        del buf274
        buf298 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf297, (1568, 384), (384, 1), 0), reinterpret_tensor(arg168_1, (384, 1152), (1, 384), 0), out=buf298)
        del arg168_1
        buf299 = reinterpret_tensor(buf298, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf298  # reuse
        # Topologically Sorted Source Nodes: [x_366], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf299, arg169_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg169_1
        buf300 = reinterpret_tensor(buf297, (1568, 384), (384, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf299, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg170_1, (1152, 384), (1, 1152), 0), out=buf300)
        del arg170_1
        buf304 = reinterpret_tensor(buf281, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_370, layer_norm_67], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf293, buf300, arg171_1, arg172_1, arg173_1, buf304, 1568, 384, grid=grid(1568), stream=stream0)
        del arg172_1
        del arg173_1
        buf305 = reinterpret_tensor(buf299, (1568, 1152), (1152, 1), 0); del buf299  # reuse
        # Topologically Sorted Source Nodes: [linear_144], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (1568, 384), (384, 1), 0), reinterpret_tensor(arg174_1, (384, 1152), (1, 384), 0), out=buf305)
        del arg174_1
        # Topologically Sorted Source Nodes: [x_371], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf306 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf305, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf305, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf305, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf307 = buf306[0]
        del buf306
        buf311 = reinterpret_tensor(buf304, (1568, 384), (384, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (1568, 384), (384, 1), 0), reinterpret_tensor(arg175_1, (384, 384), (1, 384), 0), out=buf311)
        del arg175_1
        buf312 = reinterpret_tensor(buf311, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf311  # reuse
        buf316 = reinterpret_tensor(buf307, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf307  # reuse
        # Topologically Sorted Source Nodes: [x_370, x_375, layer_norm_68], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf312, buf293, buf300, arg171_1, arg176_1, arg177_1, arg178_1, buf316, 1568, 384, grid=grid(1568), stream=stream0)
        del arg171_1
        del arg176_1
        del arg177_1
        del arg178_1
        del buf293
        buf317 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf316, (1568, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 1152), (1, 384), 0), out=buf317)
        del arg179_1
        buf318 = reinterpret_tensor(buf317, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [x_377], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf318, arg180_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg180_1
        buf319 = reinterpret_tensor(buf316, (1568, 384), (384, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf318, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg181_1, (1152, 384), (1, 1152), 0), out=buf319)
        del arg181_1
        buf323 = reinterpret_tensor(buf300, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf300  # reuse
        # Topologically Sorted Source Nodes: [x_381, layer_norm_69], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf312, buf319, arg182_1, arg183_1, arg184_1, buf323, 1568, 384, grid=grid(1568), stream=stream0)
        del arg183_1
        del arg184_1
        buf324 = reinterpret_tensor(buf318, (1568, 1152), (1152, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf323, (1568, 384), (384, 1), 0), reinterpret_tensor(arg185_1, (384, 1152), (1, 384), 0), out=buf324)
        del arg185_1
        # Topologically Sorted Source Nodes: [x_382], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf325 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf324, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf324, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf324, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf326 = buf325[0]
        del buf325
        buf330 = reinterpret_tensor(buf323, (1568, 384), (384, 1), 0); del buf323  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf326, (1568, 384), (384, 1), 0), reinterpret_tensor(arg186_1, (384, 384), (1, 384), 0), out=buf330)
        del arg186_1
        buf331 = reinterpret_tensor(buf330, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf330  # reuse
        buf335 = reinterpret_tensor(buf326, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf326  # reuse
        # Topologically Sorted Source Nodes: [x_381, x_386, layer_norm_70], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf331, buf312, buf319, arg182_1, arg187_1, arg188_1, arg189_1, buf335, 1568, 384, grid=grid(1568), stream=stream0)
        del arg182_1
        del arg187_1
        del arg188_1
        del arg189_1
        del buf312
        buf336 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf335, (1568, 384), (384, 1), 0), reinterpret_tensor(arg190_1, (384, 1152), (1, 384), 0), out=buf336)
        del arg190_1
        buf337 = reinterpret_tensor(buf336, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf336  # reuse
        # Topologically Sorted Source Nodes: [x_388], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf337, arg191_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg191_1
        buf338 = reinterpret_tensor(buf335, (1568, 384), (384, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf337, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg192_1, (1152, 384), (1, 1152), 0), out=buf338)
        del arg192_1
        buf342 = reinterpret_tensor(buf319, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [x_392, layer_norm_71], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf331, buf338, arg193_1, arg194_1, arg195_1, buf342, 1568, 384, grid=grid(1568), stream=stream0)
        del arg194_1
        del arg195_1
        buf343 = reinterpret_tensor(buf337, (1568, 1152), (1152, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (1568, 384), (384, 1), 0), reinterpret_tensor(arg196_1, (384, 1152), (1, 384), 0), out=buf343)
        del arg196_1
        # Topologically Sorted Source Nodes: [x_393], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf344 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf343, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf343, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf343, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf345 = buf344[0]
        del buf344
        buf349 = reinterpret_tensor(buf342, (1568, 384), (384, 1), 0); del buf342  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf345, (1568, 384), (384, 1), 0), reinterpret_tensor(arg197_1, (384, 384), (1, 384), 0), out=buf349)
        del arg197_1
        buf350 = reinterpret_tensor(buf349, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf349  # reuse
        buf354 = reinterpret_tensor(buf345, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [x_392, x_397, layer_norm_72], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf350, buf331, buf338, arg193_1, arg198_1, arg199_1, arg200_1, buf354, 1568, 384, grid=grid(1568), stream=stream0)
        del arg193_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf331
        buf355 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf354, (1568, 384), (384, 1), 0), reinterpret_tensor(arg201_1, (384, 1152), (1, 384), 0), out=buf355)
        del arg201_1
        buf356 = reinterpret_tensor(buf355, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf355  # reuse
        # Topologically Sorted Source Nodes: [x_399], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf356, arg202_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg202_1
        buf357 = reinterpret_tensor(buf354, (1568, 384), (384, 1), 0); del buf354  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf356, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg203_1, (1152, 384), (1, 1152), 0), out=buf357)
        del arg203_1
        buf361 = reinterpret_tensor(buf338, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [x_403, layer_norm_73], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf350, buf357, arg204_1, arg205_1, arg206_1, buf361, 1568, 384, grid=grid(1568), stream=stream0)
        del arg205_1
        del arg206_1
        buf362 = reinterpret_tensor(buf356, (1568, 1152), (1152, 1), 0); del buf356  # reuse
        # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf361, (1568, 384), (384, 1), 0), reinterpret_tensor(arg207_1, (384, 1152), (1, 384), 0), out=buf362)
        del arg207_1
        # Topologically Sorted Source Nodes: [x_404], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf363 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf362, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf362, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf362, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf364 = buf363[0]
        del buf363
        buf368 = reinterpret_tensor(buf361, (1568, 384), (384, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf364, (1568, 384), (384, 1), 0), reinterpret_tensor(arg208_1, (384, 384), (1, 384), 0), out=buf368)
        del arg208_1
        buf369 = reinterpret_tensor(buf368, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf368  # reuse
        buf373 = reinterpret_tensor(buf364, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [x_403, x_408, layer_norm_74], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf369, buf350, buf357, arg204_1, arg209_1, arg210_1, arg211_1, buf373, 1568, 384, grid=grid(1568), stream=stream0)
        del arg204_1
        del arg209_1
        del arg210_1
        del arg211_1
        del buf350
        buf374 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf373, (1568, 384), (384, 1), 0), reinterpret_tensor(arg212_1, (384, 1152), (1, 384), 0), out=buf374)
        del arg212_1
        buf375 = reinterpret_tensor(buf374, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf374  # reuse
        # Topologically Sorted Source Nodes: [x_410], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf375, arg213_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg213_1
        buf376 = reinterpret_tensor(buf373, (1568, 384), (384, 1), 0); del buf373  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf375, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg214_1, (1152, 384), (1, 1152), 0), out=buf376)
        del arg214_1
        buf380 = reinterpret_tensor(buf357, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf357  # reuse
        # Topologically Sorted Source Nodes: [x_414, layer_norm_75], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_26.run(buf369, buf376, arg215_1, arg216_1, arg217_1, buf380, 1568, 384, grid=grid(1568), stream=stream0)
        del arg216_1
        del arg217_1
        buf381 = reinterpret_tensor(buf375, (1568, 1152), (1152, 1), 0); del buf375  # reuse
        # Topologically Sorted Source Nodes: [linear_160], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (1568, 384), (384, 1), 0), reinterpret_tensor(arg218_1, (384, 1152), (1, 384), 0), out=buf381)
        del arg218_1
        # Topologically Sorted Source Nodes: [x_415], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf382 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf381, (8, 12, 196, 32), (225792, 32, 1152, 1), 0), reinterpret_tensor(buf381, (8, 12, 196, 32), (225792, 32, 1152, 1), 384), reinterpret_tensor(buf381, (8, 12, 196, 32), (225792, 32, 1152, 1), 768), None, False)
        buf383 = buf382[0]
        del buf382
        buf387 = reinterpret_tensor(buf380, (1568, 384), (384, 1), 0); del buf380  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf383, (1568, 384), (384, 1), 0), reinterpret_tensor(arg219_1, (384, 384), (1, 384), 0), out=buf387)
        del arg219_1
        buf388 = reinterpret_tensor(buf387, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf387  # reuse
        buf392 = reinterpret_tensor(buf383, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf383  # reuse
        # Topologically Sorted Source Nodes: [x_414, x_419, layer_norm_76], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_27.run(buf388, buf369, buf376, arg215_1, arg220_1, arg221_1, arg222_1, buf392, 1568, 384, grid=grid(1568), stream=stream0)
        del arg215_1
        del arg220_1
        del arg221_1
        del arg222_1
        del buf369
        del buf376
        buf393 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf392, (1568, 384), (384, 1), 0), reinterpret_tensor(arg223_1, (384, 1152), (1, 384), 0), out=buf393)
        del arg223_1
        buf394 = reinterpret_tensor(buf393, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf393  # reuse
        # Topologically Sorted Source Nodes: [x_421], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_25.run(buf394, arg224_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg224_1
        buf395 = reinterpret_tensor(buf392, (1568, 384), (384, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf394, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg225_1, (1152, 384), (1, 1152), 0), out=buf395)
        del arg225_1
        del buf394
        buf400 = empty_strided_cuda((8, 197, 384), (75648, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_427, layer_norm_77], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_28.run(arg227_1, buf388, buf395, arg226_1, arg228_1, arg229_1, buf400, 1576, 384, grid=grid(1576), stream=stream0)
        del arg228_1
        del arg229_1
        buf401 = empty_strided_cuda((1576, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_164], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (1576, 384), (384, 1), 0), reinterpret_tensor(arg230_1, (384, 768), (1, 384), 0), out=buf401)
        del arg230_1
        buf402 = empty_strided_cuda((8, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_165], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (8, 384), (75648, 1), 0), reinterpret_tensor(arg231_1, (384, 384), (1, 384), 0), out=buf402)
        del arg231_1
        buf403 = reinterpret_tensor(buf402, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf402  # reuse
        # Topologically Sorted Source Nodes: [q_30], Original ATen: [aten.mul]
        triton_poi_fused_mul_29.run(buf403, 3072, grid=grid(3072), stream=stream0)
        buf404 = reinterpret_tensor(buf400, (8, 12, 32, 197), (75648, 6304, 197, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [attn_46], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf401, buf404, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf405 = empty_strided_cuda((96, 1, 197), (197, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf403, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf404, (96, 32, 197), (6304, 197, 1), 0), out=buf405)
        buf408 = empty_strided_cuda((8, 12, 1, 197), (2368, 197, 197, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_47], Original ATen: [aten._softmax]
        triton_per_fused__softmax_31.run(buf405, buf408, 96, 197, grid=grid(96), stream=stream0)
        buf409 = reinterpret_tensor(buf405, (96, 1, 197), (197, 18912, 1), 0); del buf405  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        triton_poi_fused_bmm_32.run(buf408, buf409, 18912, grid=grid(18912), stream=stream0)
        buf410 = reinterpret_tensor(buf404, (8, 12, 197, 32), (75648, 6304, 32, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf401, buf410, 605184, grid=grid(605184), stream=stream0)
        buf411 = reinterpret_tensor(buf403, (96, 1, 32), (32, 32, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf409, reinterpret_tensor(buf410, (96, 197, 32), (6304, 32, 1), 0), out=buf411)
        buf412 = empty_strided_cuda((8, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf411, (8, 384), (384, 1), 0), reinterpret_tensor(arg232_1, (384, 384), (1, 384), 0), out=buf412)
        del arg232_1
        buf413 = reinterpret_tensor(buf412, (8, 1, 384), (384, 3072, 1), 0); del buf412  # reuse
        buf417 = reinterpret_tensor(buf411, (8, 1, 384), (384, 384, 1), 0); del buf411  # reuse
        # Topologically Sorted Source Nodes: [cls_embed_16, layer_norm_78], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_34.run(buf413, arg227_1, buf388, buf395, arg226_1, arg233_1, arg234_1, arg235_1, buf417, 8, 384, grid=grid(8), stream=stream0)
        del arg233_1
        del arg234_1
        del arg235_1
        buf418 = empty_strided_cuda((8, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf417, (8, 384), (384, 1), 0), reinterpret_tensor(arg236_1, (384, 1152), (1, 384), 0), out=buf418)
        del arg236_1
        buf419 = reinterpret_tensor(buf418, (8, 1, 1152), (1152, 1152, 1), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [x_429], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_35.run(buf419, arg237_1, 9216, grid=grid(9216), stream=stream0)
        del arg237_1
        buf420 = reinterpret_tensor(buf417, (8, 384), (384, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf419, (8, 1152), (1152, 1), 0), reinterpret_tensor(arg238_1, (1152, 384), (1, 1152), 0), out=buf420)
        del arg238_1
        buf421 = reinterpret_tensor(buf410, (8, 197, 384), (75648, 384, 1), 0); del buf410  # reuse
        buf425 = empty_strided_cuda((8, 197, 384), (75648, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_433, layer_norm_79], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_36.run(buf413, buf420, arg239_1, arg227_1, buf388, buf395, arg226_1, arg240_1, arg241_1, buf421, buf425, 1576, 384, grid=grid(1576), stream=stream0)
        del arg226_1
        del arg227_1
        del arg239_1
        del arg240_1
        del arg241_1
        del buf388
        buf426 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (1576, 384), (384, 1), 0), reinterpret_tensor(arg242_1, (384, 768), (1, 384), 0), out=buf426)
        del arg242_1
        buf427 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [linear_170], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (8, 384), (75648, 1), 0), reinterpret_tensor(arg243_1, (384, 384), (1, 384), 0), out=buf427)
        del arg243_1
        buf428 = reinterpret_tensor(buf427, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf427  # reuse
        # Topologically Sorted Source Nodes: [q_31], Original ATen: [aten.mul]
        triton_poi_fused_mul_29.run(buf428, 3072, grid=grid(3072), stream=stream0)
        buf429 = reinterpret_tensor(buf425, (8, 12, 32, 197), (75648, 6304, 197, 1), 0); del buf425  # reuse
        # Topologically Sorted Source Nodes: [attn_49], Original ATen: [aten.clone]
        triton_poi_fused_clone_30.run(buf426, buf429, 3072, 197, grid=grid(3072, 197), stream=stream0)
        buf430 = reinterpret_tensor(buf409, (96, 1, 197), (197, 197, 1), 0); del buf409  # reuse
        # Topologically Sorted Source Nodes: [attn_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf428, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf429, (96, 32, 197), (6304, 197, 1), 0), out=buf430)
        buf433 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [attn_50], Original ATen: [aten._softmax]
        triton_per_fused__softmax_31.run(buf430, buf433, 96, 197, grid=grid(96), stream=stream0)
        buf434 = reinterpret_tensor(buf430, (96, 1, 197), (197, 18912, 1), 0); del buf430  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        triton_poi_fused_bmm_32.run(buf433, buf434, 18912, grid=grid(18912), stream=stream0)
        del buf433
        buf435 = reinterpret_tensor(buf429, (8, 12, 197, 32), (75648, 6304, 32, 1), 0); del buf429  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_33.run(buf426, buf435, 605184, grid=grid(605184), stream=stream0)
        del buf426
        buf436 = reinterpret_tensor(buf428, (96, 1, 32), (32, 32, 1), 0); del buf428  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf434, reinterpret_tensor(buf435, (96, 197, 32), (6304, 32, 1), 0), out=buf436)
        del buf434
        buf437 = reinterpret_tensor(buf413, (8, 384), (384, 1), 0); del buf413  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf436, (8, 384), (384, 1), 0), reinterpret_tensor(arg244_1, (384, 384), (1, 384), 0), out=buf437)
        del arg244_1
        buf441 = reinterpret_tensor(buf436, (8, 1, 384), (384, 384, 1), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [cls_embed_22, layer_norm_80], Original ATen: [aten.add, aten.native_layer_norm]
        triton_per_fused_add_native_layer_norm_37.run(buf421, buf437, arg245_1, arg246_1, arg247_1, buf441, 8, 384, grid=grid(8), stream=stream0)
        del arg246_1
        del arg247_1
        buf442 = reinterpret_tensor(buf419, (8, 1152), (1152, 1), 0); del buf419  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf441, (8, 384), (384, 1), 0), reinterpret_tensor(arg248_1, (384, 1152), (1, 384), 0), out=buf442)
        del arg248_1
        buf443 = reinterpret_tensor(buf442, (8, 1, 1152), (1152, 1152, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [x_435], Original ATen: [aten.gelu]
        triton_poi_fused_gelu_35.run(buf443, arg249_1, 9216, grid=grid(9216), stream=stream0)
        del arg249_1
        buf444 = reinterpret_tensor(buf441, (8, 384), (384, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf443, (8, 1152), (1152, 1), 0), reinterpret_tensor(arg250_1, (1152, 384), (1, 1152), 0), out=buf444)
        del arg250_1
        del buf443
        buf445 = reinterpret_tensor(buf435, (8, 197, 384), (75648, 384, 1), 0); del buf435  # reuse
        buf449 = empty_strided_cuda((8, 197, 384), (75648, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_439, x_440], Original ATen: [aten.cat, aten.native_layer_norm]
        triton_per_fused_cat_native_layer_norm_38.run(buf421, buf437, arg245_1, buf444, arg251_1, arg252_1, arg253_1, buf445, buf449, 1576, 384, grid=grid(1576), stream=stream0)
        del arg245_1
        del arg251_1
        del arg252_1
        del arg253_1
        del buf421
        del buf437
        del buf444
        del buf445
        buf450 = reinterpret_tensor(buf395, (8, 196, 384), (75264, 384, 1), 0); del buf395  # reuse
        # Topologically Sorted Source Nodes: [aux_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_39.run(buf449, buf450, 602112, grid=grid(602112), stream=stream0)
        buf451 = empty_strided_cuda((1568, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [aux_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf450, (1568, 384), (384, 1), 0), reinterpret_tensor(arg256_1, (384, 1000), (1, 384), 0), out=buf451)
        del arg256_1
        del buf450
        buf452 = empty_strided_cuda((8, 1000, 2), (2016, 1, 1000), torch.float32)
        # Topologically Sorted Source Nodes: [aux_1, max_2], Original ATen: [aten.add, aten.max]
        triton_red_fused_add_max_40.run(buf451, arg257_1, buf452, 16000, 98, grid=grid(16000), stream=stream0)
        del arg257_1
        del buf451
        buf455 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf449, (8, 384), (75648, 1), 0), reinterpret_tensor(arg254_1, (384, 1000), (1, 384), 0), out=buf455)
        del arg254_1
        del buf449
        buf456 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [aux_1, max_2, mul_13, out_5], Original ATen: [aten.add, aten.max, aten.mul]
        triton_per_fused_add_max_mul_41.run(buf456, buf452, arg255_1, 8000, 2, grid=grid(8000), stream=stream0)
        del arg255_1
        del buf452
    return (buf456, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((192, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((486, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((486, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((192, 576), (576, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1, 14, 14, 384), (75264, 5376, 384, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('volo_d1_224', benchmark_compiled_module)
