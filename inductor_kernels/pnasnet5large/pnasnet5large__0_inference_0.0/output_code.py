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


# kernel path: /tmp/torchinductor_sahanp/6b/c6b2twl2lgy57b5xgvhfzsbdvhxhzbdf3ft3rxeobb5gzrmnxkhv.py
# Topologically Sorted Source Nodes: [x_800], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_800 => convolution_373
# Graph fragment:
#   %convolution_373 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 109561
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (109561*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (328683*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fq/cfqlonclohuaz5c7l3kf7dmcmxgm75a2lcwk5fn3gahkzowgak77.py
# Topologically Sorted Source Nodes: [x_800], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_800 => convolution_373
# Graph fragment:
#   %convolution_373 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jx/cjxh5d2axkioz2onadpookel5gnvojzkyoodr435aqt5n6epuiqi.py
# Topologically Sorted Source Nodes: [x_801, x_802], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_801 => add_473, mul_604, mul_605, sub_201
#   x_802 => relu_200
# Graph fragment:
#   %sub_201 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_373, %unsqueeze_1609), kwargs = {})
#   %mul_604 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_201, %unsqueeze_1611), kwargs = {})
#   %mul_605 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_604, %unsqueeze_1613), kwargs = {})
#   %add_473 : [num_users=5] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_605, %unsqueeze_1615), kwargs = {})
#   %relu_200 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_473,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20908800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 96
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7r/c7rw6xm4ulg42ath77iymeu5dd3wgweusaeqrq5jxhn3uabezsuv.py
# Topologically Sorted Source Nodes: [x_814, input_24, x_865, input_27, input_29, input_30], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices, aten.relu, aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_24 => _low_memory_max_pool2d_with_offsets_42
#   input_27 => avg_pool2d_8
#   input_29 => constant_pad_nd_49
#   input_30 => avg_pool2d_9
#   x_814 => constant_pad_nd_41
#   x_865 => relu_214
# Graph fragment:
#   %constant_pad_nd_41 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_473, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_42 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_41, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %relu_214 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_473,), kwargs = {})
#   %avg_pool2d_8 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_214, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
#   %constant_pad_nd_49 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_214, [-1, 1, -1, 1], 0.0), kwargs = {})
#   %avg_pool2d_9 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%constant_pad_nd_49, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5290752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 7968) % 83
    x1 = (xindex // 96) % 83
    x0 = xindex % 96
    x3 = (xindex // 661344)
    x6 = xindex
    tmp58 = tl.load(in_ptr0 + (x0 + (192*x1) + (31680*x2) + (2613600*x3)), xmask)
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-15936) + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-15840) + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-15744) + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp6
    tmp31 = tmp30 & tmp7
    tmp32 = tl.load(in_ptr0 + ((-96) + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp31 & xmask, other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp25)
    tmp34 = tmp29 & tmp13
    tmp35 = tmp34 & tmp14
    tmp36 = tl.load(in_ptr0 + (x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp35 & xmask, other=float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp29 & tmp20
    tmp39 = tmp38 & tmp21
    tmp40 = tl.load(in_ptr0 + (96 + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp39 & xmask, other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = 1 + (2*x2)
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp6
    tmp47 = tmp46 & tmp7
    tmp48 = tl.load(in_ptr0 + (15744 + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp41)
    tmp50 = tmp45 & tmp13
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (15840 + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp51 & xmask, other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp49)
    tmp54 = tmp45 & tmp20
    tmp55 = tmp54 & tmp21
    tmp56 = tl.load(in_ptr0 + (15936 + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp55 & xmask, other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tmp59 = tl.full([1], 0, tl.int32)
    tmp60 = triton_helpers.maximum(tmp59, tmp58)
    tmp61 = 1.0
    tmp62 = tmp60 * tmp61
    tmp63 = tl.load(in_ptr0 + (15936 + x0 + (192*x1) + (31680*x2) + (2613600*x3)), tmp55 & xmask, other=0.0)
    tmp64 = triton_helpers.maximum(tmp59, tmp63)
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp55, tmp64, tmp65)
    tmp67 = tmp66 * tmp61
    tl.store(out_ptr0 + (x6), tmp57, xmask)
    tl.store(out_ptr1 + (x6), tmp62, xmask)
    tl.store(out_ptr2 + (x6), tmp67, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sb/csbrxxy7t35cmyhu3pnmwirqg6cuvvy6pag6e4fuicb5zdhzmkds.py
# Topologically Sorted Source Nodes: [x_804, x_861], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_804 => add_475, mul_607, mul_608, sub_202
#   x_861 => relu_213
# Graph fragment:
#   %sub_202 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_374, %unsqueeze_1617), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_202, %unsqueeze_1619), kwargs = {})
#   %mul_608 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_607, %unsqueeze_1621), kwargs = {})
#   %add_475 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_608, %unsqueeze_1623), kwargs = {})
#   %relu_213 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_475,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11761200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 54
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jm/cjm57uhvikfr245yxl5zbmddgkca57t6orkmhkljr22zes4r5enc.py
# Topologically Sorted Source Nodes: [x_824, x_comb_iter_1_right_14, x_851, x_comb_iter_3_right_14], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_824 => constant_pad_nd_43
#   x_851 => constant_pad_nd_46
#   x_comb_iter_1_right_14 => _low_memory_max_pool2d_with_offsets_43
#   x_comb_iter_3_right_14 => _low_memory_max_pool2d_with_offsets_44
# Graph fragment:
#   %constant_pad_nd_43 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_475, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_43 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_43, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_46 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_475, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_44 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_46, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_5 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_5(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2976048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4482) % 83
    x1 = (xindex // 54) % 83
    x0 = xindex % 54
    x3 = (xindex // 372006)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-8964) + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-8910) + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-8856) + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp6
    tmp31 = tmp30 & tmp7
    tmp32 = tl.load(in_ptr0 + ((-54) + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp31 & xmask, other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp25)
    tmp34 = tmp29 & tmp13
    tmp35 = tmp34 & tmp14
    tmp36 = tl.load(in_ptr0 + (x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp35 & xmask, other=float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp29 & tmp20
    tmp39 = tmp38 & tmp21
    tmp40 = tl.load(in_ptr0 + (54 + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp39 & xmask, other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = 1 + (2*x2)
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp6
    tmp47 = tmp46 & tmp7
    tmp48 = tl.load(in_ptr0 + (8856 + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp41)
    tmp50 = tmp45 & tmp13
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (8910 + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp51 & xmask, other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp49)
    tmp54 = tmp45 & tmp20
    tmp55 = tmp54 & tmp21
    tmp56 = tl.load(in_ptr0 + (8964 + x0 + (108*x1) + (17820*x2) + (1470150*x3)), tmp55 & xmask, other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tl.store(out_ptr0 + (x6), tmp57, xmask)
    tl.store(out_ptr1 + (x6), tmp57, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wt/cwtwglkgqjarhvkoy6b3b3wh2v7c4g33xfjgpjm2chrt63o7th4x.py
# Topologically Sorted Source Nodes: [cat_19, out_4], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   cat_19 => cat_19
#   out_4 => add_510, mul_652, mul_653, sub_217
# Graph fragment:
#   %cat_19 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_401, %convolution_402], 1), kwargs = {})
#   %sub_217 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_19, %unsqueeze_1737), kwargs = {})
#   %mul_652 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_217, %unsqueeze_1739), kwargs = {})
#   %mul_653 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_652, %unsqueeze_1741), kwargs = {})
#   %add_510 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_653, %unsqueeze_1743), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5952096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 6889) % 108
    x0 = xindex % 6889
    x2 = (xindex // 744012)
    x3 = (xindex // 6889)
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 54, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((54*x0) + (372006*x2) + x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 108, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((54*x0) + (372006*x2) + ((-54) + x1)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x0 + (6912*x3)), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cd/ccdb4wqz5xgaxlk5c3fkietzdjdbvyyjlvyze7bxyz2belax7u5l.py
# Topologically Sorted Source Nodes: [x_878, x_comb_iter_0_right_13], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_878 => constant_pad_nd_51
#   x_comb_iter_0_right_13 => _low_memory_max_pool2d_with_offsets_45
# Graph fragment:
#   %constant_pad_nd_51 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_510, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_45 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_51, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_7 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 42) % 42
    x0 = xindex % 42
    x2 = (xindex // 1764)
    x4 = xindex % 1764
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x0)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-84) + (2*x0) + (166*x1) + (6912*x2)), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-83) + (2*x0) + (166*x1) + (6912*x2)), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-82) + (2*x0) + (166*x1) + (6912*x2)), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp6
    tmp31 = tmp30 & tmp7
    tmp32 = tl.load(in_ptr0 + ((-1) + (2*x0) + (166*x1) + (6912*x2)), tmp31 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp25)
    tmp34 = tmp29 & tmp13
    tmp35 = tmp34 & tmp14
    tmp36 = tl.load(in_ptr0 + ((2*x0) + (166*x1) + (6912*x2)), tmp35 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp29 & tmp20
    tmp39 = tmp38 & tmp21
    tmp40 = tl.load(in_ptr0 + (1 + (2*x0) + (166*x1) + (6912*x2)), tmp39 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = 1 + (2*x1)
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp6
    tmp47 = tmp46 & tmp7
    tmp48 = tl.load(in_ptr0 + (82 + (2*x0) + (166*x1) + (6912*x2)), tmp47 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp41)
    tmp50 = tmp45 & tmp13
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (83 + (2*x0) + (166*x1) + (6912*x2)), tmp51 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp49)
    tmp54 = tmp45 & tmp20
    tmp55 = tmp54 & tmp21
    tmp56 = tl.load(in_ptr0 + (84 + (2*x0) + (166*x1) + (6912*x2)), tmp55 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tl.store(out_ptr0 + (x4 + (1792*x2)), tmp57, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4v/c4vmzccccpa6jyizlwcncr7fh43mhshyojtqs4fugqouelemoynm.py
# Topologically Sorted Source Nodes: [x_805, x_806], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_805 => relu_201
#   x_806 => constant_pad_nd_40
# Graph fragment:
#   %relu_201 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_473,), kwargs = {})
#   %constant_pad_nd_40 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_201, [2, 2, 2, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_8 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21934848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16224) % 169
    x1 = (xindex // 96) % 169
    x3 = (xindex // 2741856)
    x4 = xindex % 16224
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-31872) + x4 + (15840*x2) + (2613600*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dp/cdpu46t5xj44n3ikogwqcelsyxslyuiurpicyis6g77h7fjglulv.py
# Topologically Sorted Source Nodes: [x_809, x_810], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_809 => add_477, mul_610, mul_611, sub_203
#   x_810 => relu_202
# Graph fragment:
#   %sub_203 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_376, %unsqueeze_1625), kwargs = {})
#   %mul_610 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_203, %unsqueeze_1627), kwargs = {})
#   %mul_611 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_610, %unsqueeze_1629), kwargs = {})
#   %add_477 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_611, %unsqueeze_1631), kwargs = {})
#   %relu_202 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_477,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2976048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 54
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5i/c5ihhlupggzlz2wey6thoinmfomp5uwrlawnge63qpcwamez5v4g.py
# Topologically Sorted Source Nodes: [x_813, input_26, x_comb_iter_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_26 => add_481, mul_616, mul_617, sub_205
#   x_813 => add_479, mul_613, mul_614, sub_204
#   x_comb_iter_70 => add_482
# Graph fragment:
#   %sub_204 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_378, %unsqueeze_1633), kwargs = {})
#   %mul_613 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_204, %unsqueeze_1635), kwargs = {})
#   %mul_614 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_613, %unsqueeze_1637), kwargs = {})
#   %add_479 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_614, %unsqueeze_1639), kwargs = {})
#   %sub_205 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_379, %unsqueeze_1641), kwargs = {})
#   %mul_616 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_205, %unsqueeze_1643), kwargs = {})
#   %mul_617 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_616, %unsqueeze_1645), kwargs = {})
#   %add_481 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_617, %unsqueeze_1647), kwargs = {})
#   %add_482 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_479, %add_481), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2976048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 54
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sy/csy5mmpqbb5zywb7ena77ljxe7zacmnptveos5rzroahrn6ur2ps.py
# Topologically Sorted Source Nodes: [x_815, x_816], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_815 => relu_203
#   x_816 => constant_pad_nd_42
# Graph fragment:
#   %relu_203 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_475,), kwargs = {})
#   %constant_pad_nd_42 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_203, [3, 3, 3, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_11 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12632112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9234) % 171
    x1 = (xindex // 54) % 171
    x3 = (xindex // 1579014)
    x4 = xindex % 9234
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-26892) + x4 + (8910*x2) + (1470150*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2h/c2hhl3zcvsrrxqteazkkocvma6vdaezwzxbj2pccmamrzibbaaem.py
# Topologically Sorted Source Nodes: [x_825, x_826], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_825 => relu_205
#   x_826 => constant_pad_nd_44
# Graph fragment:
#   %relu_205 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_475,), kwargs = {})
#   %constant_pad_nd_44 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_205, [2, 2, 2, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_12 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12338352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9126) % 169
    x1 = (xindex // 54) % 169
    x3 = (xindex // 1542294)
    x4 = xindex % 9126
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-17928) + x4 + (8910*x2) + (1470150*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yi/cyixp6k4jm2q6gpyqeqddqnbysgdvitx7kl5jejxj6q4fv73o32h.py
# Topologically Sorted Source Nodes: [x_834, x_835], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_834 => relu_207
#   x_835 => constant_pad_nd_45
# Graph fragment:
#   %relu_207 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_475,), kwargs = {})
#   %constant_pad_nd_45 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_207, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_13 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12048048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9018) % 167
    x1 = (xindex // 54) % 167
    x3 = (xindex // 1506006)
    x4 = xindex % 9018
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-8964) + x4 + (8910*x2) + (1470150*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qw/cqwic6ejdycmovparwvwmm4zet7yggvktuozycpvf65gm2d44jx2.py
# Topologically Sorted Source Nodes: [x_833, x_842, x_comb_iter_72, x_843], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_833 => add_491, mul_628, mul_629, sub_209
#   x_842 => add_495, mul_634, mul_635, sub_211
#   x_843 => relu_209
#   x_comb_iter_72 => add_496
# Graph fragment:
#   %sub_209 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_387, %unsqueeze_1673), kwargs = {})
#   %mul_628 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_209, %unsqueeze_1675), kwargs = {})
#   %mul_629 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_628, %unsqueeze_1677), kwargs = {})
#   %add_491 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_629, %unsqueeze_1679), kwargs = {})
#   %sub_211 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_391, %unsqueeze_1689), kwargs = {})
#   %mul_634 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_211, %unsqueeze_1691), kwargs = {})
#   %mul_635 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_634, %unsqueeze_1693), kwargs = {})
#   %add_495 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_635, %unsqueeze_1695), kwargs = {})
#   %add_496 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_491, %add_495), kwargs = {})
#   %relu_209 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_496,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2976048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 54
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3w/c3wtnzdah2wupj3h4vtlzkadsjf5kotvq6xu7klkda2oknpxrlf3.py
# Topologically Sorted Source Nodes: [x_852, x_853], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_852 => relu_211
#   x_853 => constant_pad_nd_47
# Graph fragment:
#   %relu_211 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_473,), kwargs = {})
#   %constant_pad_nd_47 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_211, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_15 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[33554432], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21418752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 16032) % 167
    x1 = (xindex // 96) % 167
    x3 = (xindex // 2677344)
    x4 = xindex % 16032
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 165, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-15936) + x4 + (15840*x2) + (2613600*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b7/cb7pszt63oh52kxcukdut74azifboevot7qt5d73niraymhdfxpg.py
# Topologically Sorted Source Nodes: [x_out_14, x_866], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_866 => relu_215
#   x_out_14 => cat_18
# Graph fragment:
#   %cat_18 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_482, %add_487, %add_496, %add_501, %add_508], 1), kwargs = {})
#   %relu_215 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_18,), kwargs = {})
triton_poi_fused_cat_relu_16 = async_compile.triton('triton_poi_fused_cat_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2160
    xnumel = 6889
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 270
    x2 = xindex
    y1 = (yindex // 270)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 54, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((54*x2) + (372006*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 108, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((54*x2) + (372006*y1) + ((-54) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 - tmp11
    tmp13 = tl.load(in_ptr3 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1, 1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp22 = tl.load(in_ptr4 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + (tl.broadcast_to((-54) + y0, [XBLOCK, YBLOCK])), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr6 + ((54*x2) + (372006*y1) + ((-54) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp9, tmp27, tmp28)
    tmp30 = tmp0 >= tmp7
    tmp31 = tl.full([1, 1], 162, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr7 + ((54*x2) + (372006*y1) + ((-108) + y0)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp0 >= tmp31
    tmp36 = tl.full([1, 1], 216, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tmp35 & tmp37
    tmp39 = tl.load(in_ptr8 + ((54*x2) + (372006*y1) + ((-162) + y0)), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.load(in_ptr9 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 - tmp40
    tmp42 = tl.load(in_ptr10 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp42 + tmp14
    tmp44 = libdevice.sqrt(tmp43)
    tmp45 = tmp17 / tmp44
    tmp46 = tmp45 * tmp19
    tmp47 = tmp41 * tmp46
    tmp48 = tl.load(in_ptr11 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 * tmp48
    tmp50 = tl.load(in_ptr12 + (tl.broadcast_to((-162) + y0, [XBLOCK, YBLOCK])), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp49 + tmp50
    tmp52 = tl.load(in_ptr13 + ((54*x2) + (372006*y1) + ((-162) + y0)), tmp38 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp38, tmp53, tmp54)
    tmp56 = tmp0 >= tmp36
    tmp57 = tl.full([1, 1], 270, tl.int64)
    tmp58 = tmp0 < tmp57
    tmp59 = tl.load(in_ptr14 + ((54*x2) + (372006*y1) + ((-216) + y0)), tmp56 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp60 = tl.where(tmp38, tmp55, tmp59)
    tmp61 = tl.where(tmp33, tmp34, tmp60)
    tmp62 = tl.where(tmp9, tmp29, tmp61)
    tmp63 = tl.where(tmp4, tmp5, tmp62)
    tmp64 = tl.full([1, 1], 0, tl.int32)
    tmp65 = triton_helpers.maximum(tmp64, tmp63)
    tl.store(out_ptr0 + (x2 + (6912*y3)), tmp63, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (270*x2) + (1860030*y1)), tmp65, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7j/c7jcqycpegmn3b6smqiedbdl7htmesw5oenm2yuygmfxgyjipktw.py
# Topologically Sorted Source Nodes: [x_868, x_925], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_868 => add_512, mul_655, mul_656, sub_218
#   x_925 => relu_228
# Graph fragment:
#   %sub_218 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_403, %unsqueeze_1745), kwargs = {})
#   %mul_655 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_218, %unsqueeze_1747), kwargs = {})
#   %mul_656 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_655, %unsqueeze_1749), kwargs = {})
#   %add_512 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_656, %unsqueeze_1751), kwargs = {})
#   %relu_228 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_512,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5952096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 108
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/66/c66vheh2tst5foyzypuxvij4mu4r2rqz7cibpyatlv75i7rlfowy.py
# Topologically Sorted Source Nodes: [x_888, x_comb_iter_1_right_15, x_915, x_comb_iter_3_right_15], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_888 => constant_pad_nd_53
#   x_915 => constant_pad_nd_56
#   x_comb_iter_1_right_15 => _low_memory_max_pool2d_with_offsets_46
#   x_comb_iter_3_right_15 => _low_memory_max_pool2d_with_offsets_47
# Graph fragment:
#   %constant_pad_nd_53 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_512, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_46 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_53, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_56 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_512, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_47 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_56, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_18 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_18(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4536) % 42
    x1 = (xindex // 108) % 42
    x0 = xindex % 108
    x3 = (xindex // 190512)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-9072) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-8964) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-8856) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp6
    tmp31 = tmp30 & tmp7
    tmp32 = tl.load(in_ptr0 + ((-108) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp31 & xmask, other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp25)
    tmp34 = tmp29 & tmp13
    tmp35 = tmp34 & tmp14
    tmp36 = tl.load(in_ptr0 + (x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp35 & xmask, other=float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp29 & tmp20
    tmp39 = tmp38 & tmp21
    tmp40 = tl.load(in_ptr0 + (108 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp39 & xmask, other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = 1 + (2*x2)
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp6
    tmp47 = tmp46 & tmp7
    tmp48 = tl.load(in_ptr0 + (8856 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp41)
    tmp50 = tmp45 & tmp13
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (8964 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp51 & xmask, other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp49)
    tmp54 = tmp45 & tmp20
    tmp55 = tmp54 & tmp21
    tmp56 = tl.load(in_ptr0 + (9072 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp55 & xmask, other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tl.store(out_ptr0 + (x6), tmp57, xmask)
    tl.store(out_ptr1 + (x6), tmp57, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fk/cfkqvntynnqsc4i22pi5xdkwckg25aatw2cmywndzxvml342bo3s.py
# Topologically Sorted Source Nodes: [x_929, input_32], Original ATen: [aten.relu, aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_32 => avg_pool2d_10
#   x_929 => relu_229
# Graph fragment:
#   %relu_229 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat_18,), kwargs = {})
#   %avg_pool2d_10 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_229, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_relu_19 = async_compile.triton('triton_poi_fused_avg_pool2d_relu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_relu_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2160
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 42
    x3 = (xindex // 42)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 270
    y1 = (yindex // 270)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (166*x3) + (6912*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (y0 + (270*x5) + (476280*y1)), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dq/cdqvhsaq5q3lgfdrbgwrpppmzzrt57aiufwanopnqurcz5bnnxau.py
# Topologically Sorted Source Nodes: [x_929, input_34, input_35], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_34 => constant_pad_nd_59
#   input_35 => avg_pool2d_11
#   x_929 => relu_229
# Graph fragment:
#   %relu_229 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat_18,), kwargs = {})
#   %constant_pad_nd_59 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_229, [-1, 1, -1, 1], 0.0), kwargs = {})
#   %avg_pool2d_11 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%constant_pad_nd_59, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_constant_pad_nd_relu_20 = async_compile.triton('triton_poi_fused_avg_pool2d_constant_pad_nd_relu_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_relu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_constant_pad_nd_relu_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2160
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 42)
    x2 = xindex % 42
    y4 = yindex
    x5 = xindex
    y0 = yindex % 270
    y1 = (yindex // 270)
    tmp0 = 1 + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (84 + (2*x2) + (166*x3) + (6912*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (y0 + (270*x5) + (476280*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ss/csst43vf3jq3tduricrbpznzot6qw5jb33oveynjmpgzcmg4ckp4.py
# Topologically Sorted Source Nodes: [cat_21, out_5, x_933, x_973], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   cat_21 => cat_21
#   out_5 => add_545, mul_697, mul_698, sub_232
#   x_933 => relu_231
#   x_973 => relu_241
# Graph fragment:
#   %cat_21 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_429, %convolution_430], 1), kwargs = {})
#   %sub_232 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_21, %unsqueeze_1857), kwargs = {})
#   %mul_697 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_232, %unsqueeze_1859), kwargs = {})
#   %mul_698 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_697, %unsqueeze_1861), kwargs = {})
#   %add_545 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_698, %unsqueeze_1863), kwargs = {})
#   %relu_231 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_545,), kwargs = {})
#   %relu_241 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_545,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1728
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 216
    x2 = xindex
    y1 = (yindex // 216)
    y3 = yindex
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((108*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 216, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((108*x2) + (190512*y1) + ((-108) + y0)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1, 1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x2 + (1792*y3)), tmp25, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (216*x2) + (381024*y1)), tmp27, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (216*x2) + (381024*y1)), tmp27, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6z/c6zg6ohfue3nzbnlwmnnbypwtng6hbc2nlqlbqhpuzmawym2kh3u.py
# Topologically Sorted Source Nodes: [x_comb_iter_0_right_14], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_comb_iter_0_right_14 => _low_memory_max_pool2d_with_offsets_48
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_48 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_545, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_22 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 42) % 42
    x0 = xindex % 42
    x2 = (xindex // 1764)
    x4 = xindex % 1764
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-43) + x4 + (1792*x2)), tmp10 & xmask, other=float("-inf"))
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-42) + x4 + (1792*x2)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-41) + x4 + (1792*x2)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4 + (1792*x2)), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4 + (1792*x2)), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4 + (1792*x2)), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (41 + x4 + (1792*x2)), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (42 + x4 + (1792*x2)), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (43 + x4 + (1792*x2)), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x4 + (1792*x2)), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yl/cylohzw34pr2hwqxfqsh6wwlvqsujfndowt2dhyuml55cnu27h42.py
# Topologically Sorted Source Nodes: [x_869, x_870], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_869 => relu_216
#   x_870 => constant_pad_nd_50
# Graph fragment:
#   %relu_216 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_510,), kwargs = {})
#   %constant_pad_nd_50 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_216, [2, 2, 2, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_23 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 7569
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 87)
    x2 = xindex % 87
    y4 = yindex
    x5 = xindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-168) + x2 + (83*x3) + (6912*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (108*x5) + (817452*y1)), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uw/cuwxa5ucu7s5b2t3625uw5t6v2n2rk6nxn6u3mtoeeipdxz2ucwr.py
# Topologically Sorted Source Nodes: [x_873, x_874], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_873 => add_514, mul_658, mul_659, sub_219
#   x_874 => relu_217
# Graph fragment:
#   %sub_219 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_405, %unsqueeze_1753), kwargs = {})
#   %mul_658 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_219, %unsqueeze_1755), kwargs = {})
#   %mul_659 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_658, %unsqueeze_1757), kwargs = {})
#   %add_514 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_659, %unsqueeze_1759), kwargs = {})
#   %relu_217 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_514,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 108
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/os/cosr7tvcus7m3oqligl5a4yenjzkez3wi7ax22qb5augfczawyvr.py
# Topologically Sorted Source Nodes: [x_879, x_880], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_879 => relu_218
#   x_880 => constant_pad_nd_52
# Graph fragment:
#   %relu_218 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_512,), kwargs = {})
#   %constant_pad_nd_52 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_218, [3, 3, 3, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_25 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6843744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9612) % 89
    x1 = (xindex // 108) % 89
    x3 = (xindex // 855468)
    x4 = xindex % 9612
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-27216) + x4 + (8964*x2) + (744012*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yx/cyxulfjg7spmwpdv6ljcf3ih6c36rmy7ez63hrspbecna3lc3uqh.py
# Topologically Sorted Source Nodes: [x_889, x_890], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_889 => relu_220
#   x_890 => constant_pad_nd_54
# Graph fragment:
#   %relu_220 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_512,), kwargs = {})
#   %constant_pad_nd_54 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_220, [2, 2, 2, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_26 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6539616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9396) % 87
    x1 = (xindex // 108) % 87
    x3 = (xindex // 817452)
    x4 = xindex % 9396
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-18144) + x4 + (8964*x2) + (744012*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n3/cn3mcp7bwxjooajkxpyzojahlaabctiin6qbpws3yikgbmc7pgqa.py
# Topologically Sorted Source Nodes: [x_898, x_899], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_898 => relu_222
#   x_899 => constant_pad_nd_55
# Graph fragment:
#   %relu_222 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_512,), kwargs = {})
#   %constant_pad_nd_55 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_222, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_27 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6242400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9180) % 85
    x1 = (xindex // 108) % 85
    x3 = (xindex // 780300)
    x4 = xindex % 9180
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-9072) + x4 + (8964*x2) + (744012*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m3/cm3pjayufdmhca7yglwrgvaupp4vms3qj66rwsiqv53flvrqwn6a.py
# Topologically Sorted Source Nodes: [x_897, x_906, x_comb_iter_77, x_907], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_897 => add_526, mul_673, mul_674, sub_224
#   x_906 => add_530, mul_679, mul_680, sub_226
#   x_907 => relu_224
#   x_comb_iter_77 => add_531
# Graph fragment:
#   %sub_224 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_415, %unsqueeze_1793), kwargs = {})
#   %mul_673 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_224, %unsqueeze_1795), kwargs = {})
#   %mul_674 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_673, %unsqueeze_1797), kwargs = {})
#   %add_526 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_674, %unsqueeze_1799), kwargs = {})
#   %sub_226 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_419, %unsqueeze_1809), kwargs = {})
#   %mul_679 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_226, %unsqueeze_1811), kwargs = {})
#   %mul_680 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_679, %unsqueeze_1813), kwargs = {})
#   %add_530 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_680, %unsqueeze_1815), kwargs = {})
#   %add_531 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_526, %add_530), kwargs = {})
#   %relu_224 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_531,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 108
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5f/c5fyjev7vk5pp7ellnjve2hmgohb23n26hpqqr76xjh3zwlxgvin.py
# Topologically Sorted Source Nodes: [x_916, x_917], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_916 => relu_226
#   x_917 => constant_pad_nd_57
# Graph fragment:
#   %relu_226 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_510,), kwargs = {})
#   %constant_pad_nd_57 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_226, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_29 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_29(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 864
    xnumel = 7225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 85)
    x2 = xindex % 85
    y4 = yindex
    x5 = xindex
    y0 = yindex % 108
    y1 = (yindex // 108)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x2
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-84) + x2 + (83*x3) + (6912*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (y0 + (108*x5) + (780300*y1)), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ev/cev7z3yokvndgkdjrm73fguswlhidn5mlekaxdjsyufrm7ul5jee.py
# Topologically Sorted Source Nodes: [x_924, x_928, x_comb_iter_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_924 => add_540, mul_691, mul_692, sub_230
#   x_928 => add_542, mul_694, mul_695, sub_231
#   x_comb_iter_79 => add_543
# Graph fragment:
#   %sub_230 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_427, %unsqueeze_1841), kwargs = {})
#   %mul_691 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_230, %unsqueeze_1843), kwargs = {})
#   %mul_692 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_691, %unsqueeze_1845), kwargs = {})
#   %add_540 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_692, %unsqueeze_1847), kwargs = {})
#   %sub_231 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_428, %unsqueeze_1849), kwargs = {})
#   %mul_694 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_231, %unsqueeze_1851), kwargs = {})
#   %mul_695 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_694, %unsqueeze_1853), kwargs = {})
#   %add_542 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_695, %unsqueeze_1855), kwargs = {})
#   %add_543 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_540, %add_542), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 108
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w7/cw7tg3d4l7yt3ieawtbg6ttxsh3vavg5lt3qadnyyrhqgxze54pa.py
# Topologically Sorted Source Nodes: [x_out_15, x_930, x_981], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_930 => relu_230
#   x_981 => relu_243
#   x_out_15 => cat_20
# Graph fragment:
#   %cat_20 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_517, %add_522, %add_531, %add_536, %add_543], 1), kwargs = {})
#   %relu_230 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_20,), kwargs = {})
#   %relu_243 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_20,), kwargs = {})
triton_poi_fused_cat_relu_31 = async_compile.triton('triton_poi_fused_cat_relu_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4320
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 540
    x2 = xindex
    y1 = (yindex // 540)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 108, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((108*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + (x2 + (1792*y0) + (193536*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 216, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((108*x2) + (190512*y1) + ((-108) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-108) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-108) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-108) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-108) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((108*x2) + (190512*y1) + ((-108) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 324, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((108*x2) + (190512*y1) + ((-216) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 432, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((108*x2) + (190512*y1) + ((-324) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-324) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-324) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-324) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-324) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((108*x2) + (190512*y1) + ((-324) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 540, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((108*x2) + (190512*y1) + ((-432) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.where(tmp54, tmp71, tmp75)
    tmp77 = tl.where(tmp49, tmp50, tmp76)
    tmp78 = tl.where(tmp28, tmp45, tmp77)
    tmp79 = tl.where(tmp4, tmp24, tmp78)
    tmp80 = tl.full([1, 1], 0, tl.int32)
    tmp81 = triton_helpers.maximum(tmp80, tmp79)
    tl.store(out_ptr1 + (y0 + (540*x2) + (952560*y1)), tmp81, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (540*x2) + (952560*y1)), tmp81, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/r5/cr5ratjsfznywp7k6pu7y645vwprjt7kcr3pddrmkpkaxz7w7lwr.py
# Topologically Sorted Source Nodes: [x_932], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_932 => add_547, mul_700, mul_701, sub_233
# Graph fragment:
#   %sub_233 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_431, %unsqueeze_1865), kwargs = {})
#   %mul_700 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_233, %unsqueeze_1867), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_700, %unsqueeze_1869), kwargs = {})
#   %add_547 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_701, %unsqueeze_1871), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 216
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5m/c5mmnzkmzr534dtt6yavorpiooxgw2nwbcn7utzuqpttgvhyl2rr.py
# Topologically Sorted Source Nodes: [x_comb_iter_1_right_16, x_comb_iter_3_right_16, x_941, x_949, x_957], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
# Source node to ATen node mapping:
#   x_941 => relu_233
#   x_949 => relu_235
#   x_957 => relu_237
#   x_comb_iter_1_right_16 => _low_memory_max_pool2d_with_offsets_49
#   x_comb_iter_3_right_16 => _low_memory_max_pool2d_with_offsets_50
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_49 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_547, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_50 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_547, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
#   %relu_233 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_547,), kwargs = {})
#   %relu_235 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_547,), kwargs = {})
#   %relu_237 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_547,), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_relu_33 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_relu_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_relu_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_relu_33(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 42
    x1 = (xindex // 216) % 42
    x4 = xindex
    tmp52 = tl.load(in_ptr0 + (x4), xmask)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9288) + x4), tmp10 & xmask, other=float("-inf"))
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9072) + x4), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8856) + x4), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-216) + x4), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (216 + x4), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (8856 + x4), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (9072 + x4), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9288 + x4), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp53 = tl.full([1], 0, tl.int32)
    tmp54 = triton_helpers.maximum(tmp53, tmp52)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
    tl.store(out_ptr1 + (x4), tmp51, xmask)
    tl.store(out_ptr2 + (x4), tmp54, xmask)
    tl.store(out_ptr3 + (x4), tmp54, xmask)
    tl.store(out_ptr4 + (x4), tmp54, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bc/cbcepitutlw7sppzog57c4oqhdqo3csc7v7lfosl37vpvhmtbhyd.py
# Topologically Sorted Source Nodes: [x_983, x_987, x_1027], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1027 => relu_255
#   x_983 => add_578, mul_739, mul_740, sub_246
#   x_987 => relu_245
# Graph fragment:
#   %sub_246 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_456, %unsqueeze_1969), kwargs = {})
#   %mul_739 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_246, %unsqueeze_1971), kwargs = {})
#   %mul_740 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_739, %unsqueeze_1973), kwargs = {})
#   %add_578 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_740, %unsqueeze_1975), kwargs = {})
#   %relu_245 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_578,), kwargs = {})
#   %relu_255 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_578,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 216
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i2/ci27pkgqlv7f4lpa2cmnxw4ozlqunmmxsgnlavzl4mntxefexzml.py
# Topologically Sorted Source Nodes: [x_comb_iter_0_right_15], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_comb_iter_0_right_15 => _low_memory_max_pool2d_with_offsets_51
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_51 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_578, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_35 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 42
    x1 = (xindex // 216) % 42
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9288) + x6), tmp10 & xmask, other=float("-inf"))
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9072) + x6), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8856) + x6), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-216) + x6), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (216 + x6), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (8856 + x6), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (9072 + x6), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9288 + x6), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3o/c3opu23mq4qdyccj7ltxzmd62epsjcn5eiapehlxztapu62l2iay.py
# Topologically Sorted Source Nodes: [x_936, x_937], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_936 => add_549, mul_703, mul_704, sub_234
#   x_937 => relu_232
# Graph fragment:
#   %sub_234 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_433, %unsqueeze_1873), kwargs = {})
#   %mul_703 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_234, %unsqueeze_1875), kwargs = {})
#   %mul_704 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_703, %unsqueeze_1877), kwargs = {})
#   %add_549 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_704, %unsqueeze_1879), kwargs = {})
#   %relu_232 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_549,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 216
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g7/cg7727oielyb3d3arqiuj7cdtloipxyx2ymgjn32i4ny4nl3ms6d.py
# Topologically Sorted Source Nodes: [x_956, x_964, x_comb_iter_82, x_965], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_956 => add_561, mul_718, mul_719, sub_239
#   x_964 => add_565, mul_724, mul_725, sub_241
#   x_965 => relu_239
#   x_comb_iter_82 => add_566
# Graph fragment:
#   %sub_239 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_443, %unsqueeze_1913), kwargs = {})
#   %mul_718 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_239, %unsqueeze_1915), kwargs = {})
#   %mul_719 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_718, %unsqueeze_1917), kwargs = {})
#   %add_561 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_719, %unsqueeze_1919), kwargs = {})
#   %sub_241 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_447, %unsqueeze_1929), kwargs = {})
#   %mul_724 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_241, %unsqueeze_1931), kwargs = {})
#   %mul_725 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_724, %unsqueeze_1933), kwargs = {})
#   %add_565 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_725, %unsqueeze_1935), kwargs = {})
#   %add_566 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_561, %add_565), kwargs = {})
#   %relu_239 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_566,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 216
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lk/clkhz6g4tqikbk2xonplcfrulxp2jr6fredrlfxvcyynuq3l6rbf.py
# Topologically Sorted Source Nodes: [x_out_16, x_984, x_1035], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1035 => relu_257
#   x_984 => relu_244
#   x_out_16 => cat_22
# Graph fragment:
#   %cat_22 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_552, %add_557, %add_566, %add_571, %add_576], 1), kwargs = {})
#   %relu_244 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_22,), kwargs = {})
#   %relu_257 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_22,), kwargs = {})
triton_poi_fused_cat_relu_38 = async_compile.triton('triton_poi_fused_cat_relu_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1080
    x2 = xindex
    y1 = (yindex // 1080)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 216, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((216*x2) + (381024*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + (x2 + (1792*y0) + (387072*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 432, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 648, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((216*x2) + (381024*y1) + ((-432) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 864, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 1080, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr1 + (y0 + (1080*x2) + (1905120*y1)), tmp97, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (1080*x2) + (1905120*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2a/c2aj244zvlomya6cgthghspete6jqirjqyyaamwrm33a745zobop.py
# Topologically Sorted Source Nodes: [x_out_17, x_1038, x_1089], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1038 => relu_258
#   x_1089 => relu_271
#   x_out_17 => cat_23
# Graph fragment:
#   %cat_23 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_585, %add_590, %add_599, %add_604, %add_609], 1), kwargs = {})
#   %relu_258 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_23,), kwargs = {})
#   %relu_271 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_23,), kwargs = {})
triton_poi_fused_cat_relu_39 = async_compile.triton('triton_poi_fused_cat_relu_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1080
    x2 = xindex
    y1 = (yindex // 1080)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 216, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((216*x2) + (381024*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((216*x2) + (381024*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 432, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 648, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((216*x2) + (381024*y1) + ((-432) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 864, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 1080, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr1 + (y0 + (1080*x2) + (1905120*y1)), tmp97, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (1080*x2) + (1905120*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qd/cqdlowhwiiyxabwjcmlhobqmm6unf6vxc5rrhjloi72672k466z7.py
# Topologically Sorted Source Nodes: [x_1145], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_1145 => add_677, mul_865, mul_866, sub_288
# Graph fragment:
#   %sub_288 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_534, %unsqueeze_2305), kwargs = {})
#   %mul_865 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_288, %unsqueeze_2307), kwargs = {})
#   %mul_866 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_865, %unsqueeze_2309), kwargs = {})
#   %add_677 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_866, %unsqueeze_2311), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6096384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pr/cprdrecf4sqyj47q5xf2aqerdahvkzdys6smk3tb357entld4m43.py
# Topologically Sorted Source Nodes: [x_1158, x_comb_iter_0_right_18], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1158 => constant_pad_nd_61
#   x_comb_iter_0_right_18 => _low_memory_max_pool2d_with_offsets_60
# Graph fragment:
#   %constant_pad_nd_61 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_677, [0, 1, 0, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_60 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_61, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_41 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_41(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 21
    x1 = (xindex // 432) % 21
    x0 = xindex % 432
    x5 = (xindex // 9072)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2*x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (864*x1) + (36288*x5)), tmp5 & xmask, other=float("-inf"))
    tmp7 = 1 + (2*x1)
    tmp8 = tmp7 < tmp1
    tmp9 = tmp2 & tmp8
    tmp10 = tl.load(in_ptr0 + (432 + x0 + (864*x1) + (36288*x5)), tmp9 & xmask, other=float("-inf"))
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 2 + (2*x1)
    tmp13 = tmp12 < tmp1
    tmp14 = tmp2 & tmp13
    tmp15 = tl.load(in_ptr0 + (864 + x0 + (864*x1) + (36288*x5)), tmp14 & xmask, other=float("-inf"))
    tmp16 = triton_helpers.maximum(tmp15, tmp11)
    tmp17 = 1 + (2*x2)
    tmp18 = tmp17 < tmp1
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr0 + (18144 + x0 + (864*x1) + (36288*x5)), tmp19 & xmask, other=float("-inf"))
    tmp21 = triton_helpers.maximum(tmp20, tmp16)
    tmp22 = tmp18 & tmp8
    tmp23 = tl.load(in_ptr0 + (18576 + x0 + (864*x1) + (36288*x5)), tmp22 & xmask, other=float("-inf"))
    tmp24 = triton_helpers.maximum(tmp23, tmp21)
    tmp25 = tmp18 & tmp13
    tmp26 = tl.load(in_ptr0 + (19008 + x0 + (864*x1) + (36288*x5)), tmp25 & xmask, other=float("-inf"))
    tmp27 = triton_helpers.maximum(tmp26, tmp24)
    tmp28 = 2 + (2*x2)
    tmp29 = tmp28 < tmp1
    tmp30 = tmp29 & tmp4
    tmp31 = tl.load(in_ptr0 + (36288 + x0 + (864*x1) + (36288*x5)), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp27)
    tmp33 = tmp29 & tmp8
    tmp34 = tl.load(in_ptr0 + (36720 + x0 + (864*x1) + (36288*x5)), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp13
    tmp37 = tl.load(in_ptr0 + (37152 + x0 + (864*x1) + (36288*x5)), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tl.store(out_ptr0 + (x6), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xs/cxsiih6z7y6fqfmtainecet4okl3xk4tb4n3m3kes6fyebn4enje.py
# Topologically Sorted Source Nodes: [x_out_19, x_1146], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1146 => relu_286
#   x_out_19 => cat_25
# Graph fragment:
#   %cat_25 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_651, %add_656, %add_665, %add_670, %add_675], 1), kwargs = {})
#   %relu_286 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_25,), kwargs = {})
triton_poi_fused_cat_relu_42 = async_compile.triton('triton_poi_fused_cat_relu_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 1764
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 1080
    x2 = xindex
    y1 = (yindex // 1080)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 216, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((216*x2) + (381024*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((216*x2) + (381024*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 432, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-216) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((216*x2) + (381024*y1) + ((-216) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 648, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((216*x2) + (381024*y1) + ((-432) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 864, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-648) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((216*x2) + (381024*y1) + ((-648) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 1080, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((216*x2) + (381024*y1) + ((-864) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr0 + (x2 + (1792*y3)), tmp95, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (1080*x2) + (1905120*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2x/c2xzkr2a4dvapg6cspksjidkrwy6rxezenr5lxbpxfw5fior4mvg.py
# Topologically Sorted Source Nodes: [x_1148, x_1205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1148 => add_679, mul_868, mul_869, sub_289
#   x_1205 => relu_299
# Graph fragment:
#   %sub_289 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_535, %unsqueeze_2313), kwargs = {})
#   %mul_868 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_289, %unsqueeze_2315), kwargs = {})
#   %mul_869 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_868, %unsqueeze_2317), kwargs = {})
#   %add_679 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_869, %unsqueeze_2319), kwargs = {})
#   %relu_299 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_679,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6096384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5x/c5xp5grvp7fj4gql5owxqwd5okigmcv3uzemb4rh6pig6hndhir2.py
# Topologically Sorted Source Nodes: [x_1168, x_comb_iter_1_right_20, x_1195, x_comb_iter_3_right_20], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1168 => constant_pad_nd_63
#   x_1195 => constant_pad_nd_66
#   x_comb_iter_1_right_20 => _low_memory_max_pool2d_with_offsets_61
#   x_comb_iter_3_right_20 => _low_memory_max_pool2d_with_offsets_62
# Graph fragment:
#   %constant_pad_nd_63 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_679, [0, 1, 0, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_61 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_63, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_66 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_679, [0, 1, 0, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_62 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_66, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_44 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_44(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 21
    x1 = (xindex // 432) % 21
    x0 = xindex % 432
    x5 = (xindex // 9072)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = 2*x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (864*x1) + (36288*x5)), tmp5 & xmask, other=float("-inf"))
    tmp7 = 1 + (2*x1)
    tmp8 = tmp7 < tmp1
    tmp9 = tmp2 & tmp8
    tmp10 = tl.load(in_ptr0 + (432 + x0 + (864*x1) + (36288*x5)), tmp9 & xmask, other=float("-inf"))
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 2 + (2*x1)
    tmp13 = tmp12 < tmp1
    tmp14 = tmp2 & tmp13
    tmp15 = tl.load(in_ptr0 + (864 + x0 + (864*x1) + (36288*x5)), tmp14 & xmask, other=float("-inf"))
    tmp16 = triton_helpers.maximum(tmp15, tmp11)
    tmp17 = 1 + (2*x2)
    tmp18 = tmp17 < tmp1
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr0 + (18144 + x0 + (864*x1) + (36288*x5)), tmp19 & xmask, other=float("-inf"))
    tmp21 = triton_helpers.maximum(tmp20, tmp16)
    tmp22 = tmp18 & tmp8
    tmp23 = tl.load(in_ptr0 + (18576 + x0 + (864*x1) + (36288*x5)), tmp22 & xmask, other=float("-inf"))
    tmp24 = triton_helpers.maximum(tmp23, tmp21)
    tmp25 = tmp18 & tmp13
    tmp26 = tl.load(in_ptr0 + (19008 + x0 + (864*x1) + (36288*x5)), tmp25 & xmask, other=float("-inf"))
    tmp27 = triton_helpers.maximum(tmp26, tmp24)
    tmp28 = 2 + (2*x2)
    tmp29 = tmp28 < tmp1
    tmp30 = tmp29 & tmp4
    tmp31 = tl.load(in_ptr0 + (36288 + x0 + (864*x1) + (36288*x5)), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp27)
    tmp33 = tmp29 & tmp8
    tmp34 = tl.load(in_ptr0 + (36720 + x0 + (864*x1) + (36288*x5)), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp13
    tmp37 = tl.load(in_ptr0 + (37152 + x0 + (864*x1) + (36288*x5)), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tl.store(out_ptr0 + (x6), tmp38, xmask)
    tl.store(out_ptr1 + (x6), tmp38, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dj/cdj6fe4t4h77pzht565pnsk4vmvwiry3oewlif2zezsbq2o76252.py
# Topologically Sorted Source Nodes: [x_1209, input_37], Original ATen: [aten.relu, aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_37 => avg_pool2d_12
#   x_1209 => relu_300
# Graph fragment:
#   %relu_300 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat_25,), kwargs = {})
#   %avg_pool2d_12 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_300, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_relu_45 = async_compile.triton('triton_poi_fused_avg_pool2d_relu_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_relu_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_relu_45(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 21
    x3 = (xindex // 21)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 1080
    y1 = (yindex // 1080)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (84*x3) + (1792*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (y0 + (1080*x5) + (476280*y1)), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7x/c7xiaxf3sqcom3eerefa3f43uuifsh3uz3e3fjl7zowefnctgoqk.py
# Topologically Sorted Source Nodes: [x_1209, input_39, input_40], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_39 => constant_pad_nd_69
#   input_40 => avg_pool2d_13
#   x_1209 => relu_300
# Graph fragment:
#   %relu_300 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat_25,), kwargs = {})
#   %constant_pad_nd_69 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_300, [-1, 1, -1, 1], 0.0), kwargs = {})
#   %avg_pool2d_13 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%constant_pad_nd_69, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_constant_pad_nd_relu_46 = async_compile.triton('triton_poi_fused_avg_pool2d_constant_pad_nd_relu_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_relu_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_constant_pad_nd_relu_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8640
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 21)
    x2 = xindex % 21
    y4 = yindex
    x5 = xindex
    y0 = yindex % 1080
    y1 = (yindex // 1080)
    tmp0 = 1 + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (43 + (2*x2) + (84*x3) + (1792*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (y0 + (1080*x5) + (476280*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ah/cahjboggvcxihjohugebnt73chq7fpnk7ubrrnf6ezyzroix7ozj.py
# Topologically Sorted Source Nodes: [cat_27, out_6, x_1213, x_1253], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   cat_27 => cat_27
#   out_6 => add_712, mul_910, mul_911, sub_303
#   x_1213 => relu_302
#   x_1253 => relu_312
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_561, %convolution_562], 1), kwargs = {})
#   %sub_303 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_27, %unsqueeze_2425), kwargs = {})
#   %mul_910 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_303, %unsqueeze_2427), kwargs = {})
#   %mul_911 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_910, %unsqueeze_2429), kwargs = {})
#   %add_712 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_911, %unsqueeze_2431), kwargs = {})
#   %relu_302 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_712,), kwargs = {})
#   %relu_312 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_712,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3456
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 432
    x2 = xindex
    y1 = (yindex // 432)
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 216, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((216*x2) + (95256*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 432, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((216*x2) + (95256*y1) + ((-216) + y0)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1, 1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x2 + (441*y0) + (190528*y1)), tmp25, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (432*x2) + (190512*y1)), tmp27, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (432*x2) + (190512*y1)), tmp27, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b4/cb4eljxx4hcyqpyngse4xrlgxrt6g3pkc6lsdfvsyw35eyuc7hrb.py
# Topologically Sorted Source Nodes: [x_comb_iter_0_right_19], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_comb_iter_0_right_19 => _low_memory_max_pool2d_with_offsets_63
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_63 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_712, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_48 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 21) % 21
    x0 = xindex % 21
    x3 = (xindex // 190512)
    x6 = xindex % 190512
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-22) + x6 + (190528*x3)), tmp10 & xmask, other=float("-inf"))
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-21) + x6 + (190528*x3)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-20) + x6 + (190528*x3)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x6 + (190528*x3)), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6 + (190528*x3)), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x6 + (190528*x3)), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (20 + x6 + (190528*x3)), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (21 + x6 + (190528*x3)), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (22 + x6 + (190528*x3)), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6 + (190528*x3)), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/p7/cp7mq72wbdr3s4utbwql4bzgw4z2glsuqt3l42dog6qvr7nz6tol.py
# Topologically Sorted Source Nodes: [x_1149, x_1150], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1149 => relu_287
#   x_1150 => constant_pad_nd_60
# Graph fragment:
#   %relu_287 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_677,), kwargs = {})
#   %constant_pad_nd_60 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_287, [1, 2, 1, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_49 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6998400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 19440) % 45
    x1 = (xindex // 432) % 45
    x3 = (xindex // 874800)
    x4 = xindex % 19440
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-18576) + x4 + (18144*x2) + (762048*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zc/czcrkgqiqujhkcj22uzlmglaokq4naqwot2x5cs5jvga5obkbk4v.py
# Topologically Sorted Source Nodes: [x_1153, x_1154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1153 => add_681, mul_871, mul_872, sub_290
#   x_1154 => relu_288
# Graph fragment:
#   %sub_290 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_537, %unsqueeze_2321), kwargs = {})
#   %mul_871 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_290, %unsqueeze_2323), kwargs = {})
#   %mul_872 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_871, %unsqueeze_2325), kwargs = {})
#   %add_681 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_872, %unsqueeze_2327), kwargs = {})
#   %relu_288 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_681,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ip/cipolfgrakw2q2rqvifiecwdj2dtt4txmulkkawrrj3pjh5jkxa5.py
# Topologically Sorted Source Nodes: [x_1159, x_1160], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1159 => relu_289
#   x_1160 => constant_pad_nd_62
# Graph fragment:
#   %relu_289 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_679,), kwargs = {})
#   %constant_pad_nd_62 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_289, [2, 3, 2, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_51 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_51', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7634304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 20304) % 47
    x1 = (xindex // 432) % 47
    x3 = (xindex // 954288)
    x4 = xindex % 20304
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 42, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-37152) + x4 + (18144*x2) + (762048*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6j/c6jxocwb7dhcexkff7bgrkuzuryhj7wkcjhvqmx4shuv7n5lzrmf.py
# Topologically Sorted Source Nodes: [x_1178, x_1179], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1178 => relu_293
#   x_1179 => constant_pad_nd_65
# Graph fragment:
#   %relu_293 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_679,), kwargs = {})
#   %constant_pad_nd_65 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_293, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_52 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6390144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 18576) % 43
    x1 = (xindex // 432) % 43
    x3 = (xindex // 798768)
    x4 = xindex % 18576
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 42, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (18144*x2) + (762048*x3)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp5, tmp8, tmp9)
    tl.store(out_ptr0 + (x5), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mf/cmfeulduv2qsuor4cscc7qfhov6boslu233h7zuwrbd7v3pteqja.py
# Topologically Sorted Source Nodes: [x_1177, x_1186, x_comb_iter_102, x_1187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_1177 => add_693, mul_886, mul_887, sub_295
#   x_1186 => add_697, mul_892, mul_893, sub_297
#   x_1187 => relu_295
#   x_comb_iter_102 => add_698
# Graph fragment:
#   %sub_295 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_547, %unsqueeze_2361), kwargs = {})
#   %mul_886 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_295, %unsqueeze_2363), kwargs = {})
#   %mul_887 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_886, %unsqueeze_2365), kwargs = {})
#   %add_693 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_887, %unsqueeze_2367), kwargs = {})
#   %sub_297 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_551, %unsqueeze_2377), kwargs = {})
#   %mul_892 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_297, %unsqueeze_2379), kwargs = {})
#   %mul_893 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_892, %unsqueeze_2381), kwargs = {})
#   %add_697 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_893, %unsqueeze_2383), kwargs = {})
#   %add_698 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_693, %add_697), kwargs = {})
#   %relu_295 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_698,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jz/cjzdbx3r4sgden2woudwwbqtml246risexw2sntb3mik35sebpn4.py
# Topologically Sorted Source Nodes: [x_1204, x_1208, x_comb_iter_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_1204 => add_707, mul_904, mul_905, sub_301
#   x_1208 => add_709, mul_907, mul_908, sub_302
#   x_comb_iter_104 => add_710
# Graph fragment:
#   %sub_301 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_559, %unsqueeze_2409), kwargs = {})
#   %mul_904 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_301, %unsqueeze_2411), kwargs = {})
#   %mul_905 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_904, %unsqueeze_2413), kwargs = {})
#   %add_707 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_905, %unsqueeze_2415), kwargs = {})
#   %sub_302 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_560, %unsqueeze_2417), kwargs = {})
#   %mul_907 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_302, %unsqueeze_2419), kwargs = {})
#   %mul_908 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_907, %unsqueeze_2421), kwargs = {})
#   %add_709 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_908, %unsqueeze_2423), kwargs = {})
#   %add_710 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_707, %add_709), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tm/ctm42aa4ofhkvdtb7iswxvjryj57cwvjebixv5af66byyixqdr43.py
# Topologically Sorted Source Nodes: [x_out_20, x_1210, x_1261], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1210 => relu_301
#   x_1261 => relu_314
#   x_out_20 => cat_26
# Graph fragment:
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_684, %add_689, %add_698, %add_703, %add_710], 1), kwargs = {})
#   %relu_301 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_26,), kwargs = {})
#   %relu_314 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_26,), kwargs = {})
triton_poi_fused_cat_relu_55 = async_compile.triton('triton_poi_fused_cat_relu_55', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2160
    x2 = xindex
    y1 = (yindex // 2160)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 432, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((432*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((432*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 864, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 1296, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((432*x2) + (190512*y1) + ((-864) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 1728, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 2160, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((432*x2) + (190512*y1) + ((-1728) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.where(tmp54, tmp71, tmp75)
    tmp77 = tl.where(tmp49, tmp50, tmp76)
    tmp78 = tl.where(tmp28, tmp45, tmp77)
    tmp79 = tl.where(tmp4, tmp24, tmp78)
    tmp80 = tl.full([1, 1], 0, tl.int32)
    tmp81 = triton_helpers.maximum(tmp80, tmp79)
    tl.store(out_ptr1 + (y0 + (2160*x2) + (952560*y1)), tmp81, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (2160*x2) + (952560*y1)), tmp81, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6u/c6udw4235gcpdgdztw2uhjkzziy6kvqzxnkp7eomsdc4e6kw2low.py
# Topologically Sorted Source Nodes: [x_1212], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_1212 => add_714, mul_913, mul_914, sub_304
# Graph fragment:
#   %sub_304 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_563, %unsqueeze_2433), kwargs = {})
#   %mul_913 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_304, %unsqueeze_2435), kwargs = {})
#   %mul_914 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_913, %unsqueeze_2437), kwargs = {})
#   %add_714 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_914, %unsqueeze_2439), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_56', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7l/c7ldni5wmnhcbiv4yyvgklkilge3j6ds4ylf7ghw5e6jhjkcvhny.py
# Topologically Sorted Source Nodes: [x_comb_iter_1_right_21, x_comb_iter_3_right_21, x_1221, x_1229, x_1237], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
# Source node to ATen node mapping:
#   x_1221 => relu_304
#   x_1229 => relu_306
#   x_1237 => relu_308
#   x_comb_iter_1_right_21 => _low_memory_max_pool2d_with_offsets_64
#   x_comb_iter_3_right_21 => _low_memory_max_pool2d_with_offsets_65
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_64 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_714, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_65 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_714, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
#   %relu_304 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_714,), kwargs = {})
#   %relu_306 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_714,), kwargs = {})
#   %relu_308 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_714,), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_relu_57 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_relu_57', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_relu_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_relu_57(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 21
    x1 = (xindex // 432) % 21
    x4 = xindex
    tmp52 = tl.load(in_ptr0 + (x4), xmask)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9504) + x4), tmp10 & xmask, other=float("-inf"))
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9072) + x4), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8640) + x4), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-432) + x4), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (432 + x4), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (8640 + x4), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (9072 + x4), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9504 + x4), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp53 = tl.full([1], 0, tl.int32)
    tmp54 = triton_helpers.maximum(tmp53, tmp52)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
    tl.store(out_ptr1 + (x4), tmp51, xmask)
    tl.store(out_ptr2 + (x4), tmp54, xmask)
    tl.store(out_ptr3 + (x4), tmp54, xmask)
    tl.store(out_ptr4 + (x4), tmp54, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mo/cmoaoxrncy4uuanx3ezazca6f3iricz77p7obbhuecmllgwdgtkl.py
# Topologically Sorted Source Nodes: [x_1263, x_1267, x_1307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1263 => add_745, mul_952, mul_953, sub_317
#   x_1267 => relu_316
#   x_1307 => relu_326
# Graph fragment:
#   %sub_317 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_588, %unsqueeze_2537), kwargs = {})
#   %mul_952 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_317, %unsqueeze_2539), kwargs = {})
#   %mul_953 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_952, %unsqueeze_2541), kwargs = {})
#   %add_745 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_953, %unsqueeze_2543), kwargs = {})
#   %relu_316 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_745,), kwargs = {})
#   %relu_326 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_745,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_58', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 432
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/x3/cx3xxzaxzuwgsu4xfqjyeqmbk2dgpblh5757yuson2wy2rxcdtwf.py
# Topologically Sorted Source Nodes: [x_comb_iter_0_right_20], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_comb_iter_0_right_20 => _low_memory_max_pool2d_with_offsets_66
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_66 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_745, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_59 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1524096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9072) % 21
    x1 = (xindex // 432) % 21
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9504) + x6), tmp10 & xmask, other=float("-inf"))
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9072) + x6), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8640) + x6), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-432) + x6), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (432 + x6), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (8640 + x6), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (9072 + x6), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (9504 + x6), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ss/cssdhuoau4peej73bt4pmmli4ljuw7iahd3zjffp4xbnogt4qlfa.py
# Topologically Sorted Source Nodes: [x_out_21, x_1264, x_1315], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1264 => relu_315
#   x_1315 => relu_328
#   x_out_21 => cat_28
# Graph fragment:
#   %cat_28 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_719, %add_724, %add_733, %add_738, %add_743], 1), kwargs = {})
#   %relu_315 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_28,), kwargs = {})
#   %relu_328 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_28,), kwargs = {})
triton_poi_fused_cat_relu_60 = async_compile.triton('triton_poi_fused_cat_relu_60', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_60(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2160
    x2 = xindex
    y1 = (yindex // 2160)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 432, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((432*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + (x2 + (441*y0) + (190528*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 864, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 1296, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((432*x2) + (190512*y1) + ((-864) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 1728, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 2160, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((432*x2) + (190512*y1) + ((-1728) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((432*x2) + (190512*y1) + ((-1728) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr1 + (y0 + (2160*x2) + (952560*y1)), tmp97, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (2160*x2) + (952560*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/p3/cp3fiy6udsso2gvy5wnz5sr6dsb4yce7h7zhdhko2tusfbkkgdnz.py
# Topologically Sorted Source Nodes: [x_out_22, x_1318, x_1369], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1318 => relu_329
#   x_1369 => relu_342
#   x_out_22 => cat_29
# Graph fragment:
#   %cat_29 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_752, %add_757, %add_766, %add_771, %add_776], 1), kwargs = {})
#   %relu_329 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_29,), kwargs = {})
#   %relu_342 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_29,), kwargs = {})
triton_poi_fused_cat_relu_61 = async_compile.triton('triton_poi_fused_cat_relu_61', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2160
    x2 = xindex
    y1 = (yindex // 2160)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 432, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((432*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((432*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 864, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 1296, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((432*x2) + (190512*y1) + ((-864) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 1728, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 2160, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((432*x2) + (190512*y1) + ((-1728) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((432*x2) + (190512*y1) + ((-1728) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr1 + (y0 + (2160*x2) + (952560*y1)), tmp97, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (2160*x2) + (952560*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jp/cjp53r4efci6at5ntngbnkzsici65cyjm2tll6fdymic2a7yzxay.py
# Topologically Sorted Source Nodes: [x_1371], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_1371 => add_811, mul_1036, mul_1037, sub_345
# Graph fragment:
#   %sub_345 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_640, %unsqueeze_2761), kwargs = {})
#   %mul_1036 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_345, %unsqueeze_2763), kwargs = {})
#   %mul_1037 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1036, %unsqueeze_2765), kwargs = {})
#   %add_811 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1037, %unsqueeze_2767), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_62', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 864
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qb/cqbilg7i2jchtbryma43zqudfkrnbbfbaojbr6hjqfapczyi65uy.py
# Topologically Sorted Source Nodes: [x_1384, x_comb_iter_0_right_22], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1384 => constant_pad_nd_71
#   x_comb_iter_0_right_22 => _low_memory_max_pool2d_with_offsets_72
# Graph fragment:
#   %constant_pad_nd_71 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_811, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_72 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_71, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_63 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_63(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9504) % 11
    x1 = (xindex // 864) % 11
    x0 = xindex % 864
    x3 = (xindex // 104544)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-19008) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-18144) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-17280) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp6
    tmp31 = tmp30 & tmp7
    tmp32 = tl.load(in_ptr0 + ((-864) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp31 & xmask, other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp25)
    tmp34 = tmp29 & tmp13
    tmp35 = tmp34 & tmp14
    tmp36 = tl.load(in_ptr0 + (x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp35 & xmask, other=float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp29 & tmp20
    tmp39 = tmp38 & tmp21
    tmp40 = tl.load(in_ptr0 + (864 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp39 & xmask, other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = 1 + (2*x2)
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp6
    tmp47 = tmp46 & tmp7
    tmp48 = tl.load(in_ptr0 + (17280 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp41)
    tmp50 = tmp45 & tmp13
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (18144 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp51 & xmask, other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp49)
    tmp54 = tmp45 & tmp20
    tmp55 = tmp54 & tmp21
    tmp56 = tl.load(in_ptr0 + (19008 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp55 & xmask, other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tl.store(out_ptr0 + (x6), tmp57, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sd/csdw7iuewbspkfu3nj3ngupt43v46xygpfvn4qfs4byya6muoykg.py
# Topologically Sorted Source Nodes: [x_out_23, x_1372], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1372 => relu_343
#   x_out_23 => cat_30
# Graph fragment:
#   %cat_30 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_785, %add_790, %add_799, %add_804, %add_809], 1), kwargs = {})
#   %relu_343 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_30,), kwargs = {})
triton_poi_fused_cat_relu_64 = async_compile.triton('triton_poi_fused_cat_relu_64', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 441
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 2160
    x2 = xindex
    y1 = (yindex // 2160)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 432, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((432*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((432*x2) + (190512*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 864, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-432) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((432*x2) + (190512*y1) + ((-432) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 1296, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((432*x2) + (190512*y1) + ((-864) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 1728, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-1296) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((432*x2) + (190512*y1) + ((-1296) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 2160, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((432*x2) + (190512*y1) + ((-1728) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-1728) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((432*x2) + (190512*y1) + ((-1728) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr0 + (x2 + (441*y0) + (952576*y1)), tmp95, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (2160*x2) + (952560*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ww/cwwmmc7xcuoudjjfvgjcybc64dxhhaflpbdvghivilk3gzgeed5z.py
# Topologically Sorted Source Nodes: [x_1374, x_1431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1374 => add_813, mul_1039, mul_1040, sub_346
#   x_1431 => relu_356
# Graph fragment:
#   %sub_346 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_641, %unsqueeze_2769), kwargs = {})
#   %mul_1039 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_346, %unsqueeze_2771), kwargs = {})
#   %mul_1040 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1039, %unsqueeze_2773), kwargs = {})
#   %add_813 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1040, %unsqueeze_2775), kwargs = {})
#   %relu_356 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_813,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_65', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 864
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/26/c26432ypjvtj2o3dasz4g6anvtiatssofjortqj4zuir4inbueyf.py
# Topologically Sorted Source Nodes: [x_1394, x_comb_iter_1_right_24, x_1421, x_comb_iter_3_right_24], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1394 => constant_pad_nd_73
#   x_1421 => constant_pad_nd_76
#   x_comb_iter_1_right_24 => _low_memory_max_pool2d_with_offsets_73
#   x_comb_iter_3_right_24 => _low_memory_max_pool2d_with_offsets_74
# Graph fragment:
#   %constant_pad_nd_73 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_813, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_73 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_73, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %constant_pad_nd_76 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_813, [1, 1, 1, 1], -inf), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_74 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%constant_pad_nd_76, [3, 3], [2, 2], [0, 0], [1, 1], False), kwargs = {})
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_66 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_66', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_66(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9504) % 11
    x1 = (xindex // 864) % 11
    x0 = xindex % 864
    x3 = (xindex // 104544)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-19008) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp8 & tmp13
    tmp16 = tmp15 & tmp14
    tmp17 = tl.load(in_ptr0 + ((-18144) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp8 & tmp20
    tmp23 = tmp22 & tmp21
    tmp24 = tl.load(in_ptr0 + ((-17280) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp6
    tmp31 = tmp30 & tmp7
    tmp32 = tl.load(in_ptr0 + ((-864) + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp31 & xmask, other=float("-inf"))
    tmp33 = triton_helpers.maximum(tmp32, tmp25)
    tmp34 = tmp29 & tmp13
    tmp35 = tmp34 & tmp14
    tmp36 = tl.load(in_ptr0 + (x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp35 & xmask, other=float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp33)
    tmp38 = tmp29 & tmp20
    tmp39 = tmp38 & tmp21
    tmp40 = tl.load(in_ptr0 + (864 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp39 & xmask, other=float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = 1 + (2*x2)
    tmp43 = tmp42 >= tmp1
    tmp44 = tmp42 < tmp3
    tmp45 = tmp43 & tmp44
    tmp46 = tmp45 & tmp6
    tmp47 = tmp46 & tmp7
    tmp48 = tl.load(in_ptr0 + (17280 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp47 & xmask, other=float("-inf"))
    tmp49 = triton_helpers.maximum(tmp48, tmp41)
    tmp50 = tmp45 & tmp13
    tmp51 = tmp50 & tmp14
    tmp52 = tl.load(in_ptr0 + (18144 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp51 & xmask, other=float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp49)
    tmp54 = tmp45 & tmp20
    tmp55 = tmp54 & tmp21
    tmp56 = tl.load(in_ptr0 + (19008 + x0 + (1728*x1) + (36288*x2) + (381024*x3)), tmp55 & xmask, other=float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tl.store(out_ptr0 + (x6), tmp57, xmask)
    tl.store(out_ptr1 + (x6), tmp57, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3k/c3k3l4w7fgwtz6azmrl2socldsi66xind7wyfqcim676ruetgxgf.py
# Topologically Sorted Source Nodes: [x_1435, input_42], Original ATen: [aten.relu, aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_42 => avg_pool2d_14
#   x_1435 => relu_357
# Graph fragment:
#   %relu_357 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat_30,), kwargs = {})
#   %avg_pool2d_14 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_357, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_relu_67 = async_compile.triton('triton_poi_fused_avg_pool2d_relu_67', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_relu_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_relu_67(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 11
    x3 = (xindex // 11)
    y0 = yindex % 2160
    y1 = (yindex // 2160)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (42*x3) + (441*y0) + (952576*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (y0 + (2160*x4) + (261360*y1)), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cb/ccb33cvtxeqjujldgu42f7y6wewc3txscqn5yh2a2sluiwoek3ut.py
# Topologically Sorted Source Nodes: [x_1435, input_44, input_45], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_44 => constant_pad_nd_79
#   input_45 => avg_pool2d_15
#   x_1435 => relu_357
# Graph fragment:
#   %relu_357 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat_30,), kwargs = {})
#   %constant_pad_nd_79 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_357, [-1, 1, -1, 1], 0.0), kwargs = {})
#   %avg_pool2d_15 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%constant_pad_nd_79, [1, 1], [2, 2], [0, 0], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_constant_pad_nd_relu_68 = async_compile.triton('triton_poi_fused_avg_pool2d_constant_pad_nd_relu_68', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_constant_pad_nd_relu_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_constant_pad_nd_relu_68(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 17280
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 11)
    x2 = xindex % 11
    y0 = yindex % 2160
    y1 = (yindex // 2160)
    x5 = xindex
    tmp0 = 1 + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 1 + (2*x2)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + (22 + (2*x2) + (42*x3) + (441*y0) + (952576*y1)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (y0 + (2160*x5) + (261360*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s7/cs7ghn5vwsb6revyfr5dnz3spjg6tf5htsmkcmbesqsfb72rttyx.py
# Topologically Sorted Source Nodes: [cat_32, out_7, x_1439, x_1479], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   cat_32 => cat_32
#   out_7 => add_846, mul_1081, mul_1082, sub_360
#   x_1439 => relu_359
#   x_1479 => relu_369
# Graph fragment:
#   %cat_32 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_667, %convolution_668], 1), kwargs = {})
#   %sub_360 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_32, %unsqueeze_2881), kwargs = {})
#   %mul_1081 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_360, %unsqueeze_2883), kwargs = {})
#   %mul_1082 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1081, %unsqueeze_2885), kwargs = {})
#   %add_846 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1082, %unsqueeze_2887), kwargs = {})
#   %relu_359 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_846,), kwargs = {})
#   %relu_369 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_846,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_69 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_69', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6912
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 864
    x2 = xindex
    y1 = (yindex // 864)
    y3 = yindex
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 432, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((432*x2) + (52272*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 864, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((432*x2) + (52272*y1) + ((-432) + y0)), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1, 1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tl.full([1, 1], 0, tl.int32)
    tmp27 = triton_helpers.maximum(tmp26, tmp25)
    tl.store(out_ptr0 + (x2 + (121*y3)), tmp25, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (864*x2) + (104544*y1)), tmp27, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (864*x2) + (104544*y1)), tmp27, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/os/coslfmrxpdg4uqwrcnmdzn3jdc5t4askbabd33zfuwlf4ki3dhlh.py
# Topologically Sorted Source Nodes: [x_comb_iter_0_right_23], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_comb_iter_0_right_23 => _low_memory_max_pool2d_with_offsets_75
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_75 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_846, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_70 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_70', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_70(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 11) % 11
    x0 = xindex % 11
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 11, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-12) + x4), tmp10 & xmask, other=float("-inf"))
    tmp12 = x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-11) + x4), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-10) + x4), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (10 + x4), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (11 + x4), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (12 + x4), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ma/cmascrdj3whzautvsyzi7iche6xwe62cawnj3bbvj4dws7bhucgz.py
# Topologically Sorted Source Nodes: [x_1375, x_1376], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1375 => relu_344
#   x_1376 => constant_pad_nd_70
# Graph fragment:
#   %relu_344 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_811,), kwargs = {})
#   %constant_pad_nd_70 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_344, [2, 2, 2, 2], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_71 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_71', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_71(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4320000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 21600) % 25
    x1 = (xindex // 864) % 25
    x3 = (xindex // 540000)
    x4 = xindex % 21600
    x6 = xindex
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-2) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-38016) + x4 + (18144*x2) + (381024*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jj/cjjxtmepo4brkgaxqbzzpoen7xfqjcsecte3rjbv4zushmsaovah.py
# Topologically Sorted Source Nodes: [x_1379, x_1380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1379 => add_815, mul_1042, mul_1043, sub_347
#   x_1380 => relu_345
# Graph fragment:
#   %sub_347 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_643, %unsqueeze_2777), kwargs = {})
#   %mul_1042 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_347, %unsqueeze_2779), kwargs = {})
#   %mul_1043 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1042, %unsqueeze_2781), kwargs = {})
#   %add_815 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1043, %unsqueeze_2783), kwargs = {})
#   %relu_345 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_815,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_72 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_72', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_72', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_72(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 864
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/42/c42zupkerrapk5szg43xxkguc4x3nv3lhv2xznoxvrvezpdnilv7.py
# Topologically Sorted Source Nodes: [x_1385, x_1386], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1385 => relu_346
#   x_1386 => constant_pad_nd_72
# Graph fragment:
#   %relu_346 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_813,), kwargs = {})
#   %constant_pad_nd_72 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_346, [3, 3, 3, 3], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_73 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_73', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_73(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5038848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 23328) % 27
    x1 = (xindex // 864) % 27
    x3 = (xindex // 629856)
    x4 = xindex % 23328
    x6 = xindex
    tmp0 = (-3) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-3) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-57024) + x4 + (18144*x2) + (381024*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bu/cbu37bk3uhclgm5yr6ie3eqjpghh63phhf54sek2nhvecttigje6.py
# Topologically Sorted Source Nodes: [x_1404, x_1405], Original ATen: [aten.relu, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_1404 => relu_350
#   x_1405 => constant_pad_nd_75
# Graph fragment:
#   %relu_350 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_813,), kwargs = {})
#   %constant_pad_nd_75 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%relu_350, [1, 1, 1, 1], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_relu_74 = async_compile.triton('triton_poi_fused_constant_pad_nd_relu_74', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_relu_74', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_relu_74(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3656448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 19872) % 23
    x1 = (xindex // 864) % 23
    x3 = (xindex // 457056)
    x4 = xindex % 19872
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 21, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + x1
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-19008) + x4 + (18144*x2) + (381024*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = triton_helpers.maximum(tmp12, tmp11)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tl.store(out_ptr0 + (x6), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b3/cb36r5gbsryuxajbkslqfbu3qbdwxqkbmstgs3ukznugysu6wdmq.py
# Topologically Sorted Source Nodes: [x_1403, x_1412, x_comb_iter_122, x_1413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_1403 => add_827, mul_1057, mul_1058, sub_352
#   x_1412 => add_831, mul_1063, mul_1064, sub_354
#   x_1413 => relu_352
#   x_comb_iter_122 => add_832
# Graph fragment:
#   %sub_352 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_653, %unsqueeze_2817), kwargs = {})
#   %mul_1057 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_352, %unsqueeze_2819), kwargs = {})
#   %mul_1058 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1057, %unsqueeze_2821), kwargs = {})
#   %add_827 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1058, %unsqueeze_2823), kwargs = {})
#   %sub_354 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_657, %unsqueeze_2833), kwargs = {})
#   %mul_1063 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_354, %unsqueeze_2835), kwargs = {})
#   %mul_1064 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1063, %unsqueeze_2837), kwargs = {})
#   %add_831 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1064, %unsqueeze_2839), kwargs = {})
#   %add_832 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_827, %add_831), kwargs = {})
#   %relu_352 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_832,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 864
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3n/c3ngmeon3hgharlweazrrzpr4yrkdeuyl4q7h2twv4xqu7ganmsy.py
# Topologically Sorted Source Nodes: [x_1430, x_1434, x_comb_iter_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_1430 => add_841, mul_1075, mul_1076, sub_358
#   x_1434 => add_843, mul_1078, mul_1079, sub_359
#   x_comb_iter_124 => add_844
# Graph fragment:
#   %sub_358 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_665, %unsqueeze_2865), kwargs = {})
#   %mul_1075 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_358, %unsqueeze_2867), kwargs = {})
#   %mul_1076 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1075, %unsqueeze_2869), kwargs = {})
#   %add_841 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1076, %unsqueeze_2871), kwargs = {})
#   %sub_359 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_666, %unsqueeze_2873), kwargs = {})
#   %mul_1078 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_359, %unsqueeze_2875), kwargs = {})
#   %mul_1079 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1078, %unsqueeze_2877), kwargs = {})
#   %add_843 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1079, %unsqueeze_2879), kwargs = {})
#   %add_844 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_841, %add_843), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_76 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_76', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_76', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_76(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 864
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7w/c7wkst5slts3ddqvp2zroa6sm6lghpolfufuirvi4hatr5ynhjbs.py
# Topologically Sorted Source Nodes: [x_out_24, x_1436, x_1487], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1436 => relu_358
#   x_1487 => relu_371
#   x_out_24 => cat_31
# Graph fragment:
#   %cat_31 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_818, %add_823, %add_832, %add_837, %add_844], 1), kwargs = {})
#   %relu_358 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_31,), kwargs = {})
#   %relu_371 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_31,), kwargs = {})
triton_poi_fused_cat_relu_77 = async_compile.triton('triton_poi_fused_cat_relu_77', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_77', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_77(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 34560
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 4320
    x2 = xindex
    y1 = (yindex // 4320)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 864, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((864*x2) + (104544*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((864*x2) + (104544*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 1728, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((864*x2) + (104544*y1) + ((-864) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((864*x2) + (104544*y1) + ((-864) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 2592, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((864*x2) + (104544*y1) + ((-1728) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 3456, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((864*x2) + (104544*y1) + ((-2592) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((864*x2) + (104544*y1) + ((-2592) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 4320, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((864*x2) + (104544*y1) + ((-3456) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.where(tmp54, tmp71, tmp75)
    tmp77 = tl.where(tmp49, tmp50, tmp76)
    tmp78 = tl.where(tmp28, tmp45, tmp77)
    tmp79 = tl.where(tmp4, tmp24, tmp78)
    tmp80 = tl.full([1, 1], 0, tl.int32)
    tmp81 = triton_helpers.maximum(tmp80, tmp79)
    tl.store(out_ptr1 + (y0 + (4320*x2) + (522720*y1)), tmp81, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (4320*x2) + (522720*y1)), tmp81, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3i/c3iktzmgafjkvtqjvqazy6j6e7ohdvuuu6wo7z7ni7itxhkfknvr.py
# Topologically Sorted Source Nodes: [x_1438], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_1438 => add_848, mul_1084, mul_1085, sub_361
# Graph fragment:
#   %sub_361 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_669, %unsqueeze_2889), kwargs = {})
#   %mul_1084 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_361, %unsqueeze_2891), kwargs = {})
#   %mul_1085 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1084, %unsqueeze_2893), kwargs = {})
#   %add_848 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1085, %unsqueeze_2895), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_78 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_78', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_78', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_78(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 864
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mp/cmpzhwfqvbfglxeyvu5emq4xopmioh5s74ypjf2ko6jqra3zhlay.py
# Topologically Sorted Source Nodes: [x_comb_iter_1_right_25, x_comb_iter_3_right_25, x_1447, x_1455, x_1463], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
# Source node to ATen node mapping:
#   x_1447 => relu_361
#   x_1455 => relu_363
#   x_1463 => relu_365
#   x_comb_iter_1_right_25 => _low_memory_max_pool2d_with_offsets_76
#   x_comb_iter_3_right_25 => _low_memory_max_pool2d_with_offsets_77
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_76 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_848, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_77 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_848, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
#   %relu_361 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_848,), kwargs = {})
#   %relu_363 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_848,), kwargs = {})
#   %relu_365 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_848,), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_relu_79 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_relu_79', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_relu_79', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_relu_79(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9504) % 11
    x1 = (xindex // 864) % 11
    x4 = xindex
    tmp52 = tl.load(in_ptr0 + (x4), xmask)
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 11, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10368) + x4), tmp10 & xmask, other=float("-inf"))
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9504) + x4), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8640) + x4), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-864) + x4), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (864 + x4), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (8640 + x4), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (9504 + x4), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (10368 + x4), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp53 = tl.full([1], 0, tl.int32)
    tmp54 = triton_helpers.maximum(tmp53, tmp52)
    tl.store(out_ptr0 + (x4), tmp51, xmask)
    tl.store(out_ptr1 + (x4), tmp51, xmask)
    tl.store(out_ptr2 + (x4), tmp54, xmask)
    tl.store(out_ptr3 + (x4), tmp54, xmask)
    tl.store(out_ptr4 + (x4), tmp54, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5h/c5h7x27vhu7jjknunry25nv2fd5qmf3n5to5wphoiiaecm3mrcti.py
# Topologically Sorted Source Nodes: [x_1489, x_1493, x_1533], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1489 => add_879, mul_1123, mul_1124, sub_374
#   x_1493 => relu_373
#   x_1533 => relu_383
# Graph fragment:
#   %sub_374 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_694, %unsqueeze_2993), kwargs = {})
#   %mul_1123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_374, %unsqueeze_2995), kwargs = {})
#   %mul_1124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1123, %unsqueeze_2997), kwargs = {})
#   %add_879 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1124, %unsqueeze_2999), kwargs = {})
#   %relu_373 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_879,), kwargs = {})
#   %relu_383 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_879,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_80 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_80', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_80', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_80(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 864
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tl.store(in_out_ptr0 + (x2), tmp15, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2j/c2jxoofq5xxrn5g4ko5ax2r3f72m6cu3nhyfk64njbr7vl7ccypv.py
# Topologically Sorted Source Nodes: [x_comb_iter_0_right_24], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_comb_iter_0_right_24 => _low_memory_max_pool2d_with_offsets_78
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_78 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_879, [3, 3], [1, 1], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_81 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_81', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_81', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_81(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 836352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 9504) % 11
    x1 = (xindex // 864) % 11
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 11, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10368) + x6), tmp10 & xmask, other=float("-inf"))
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-9504) + x6), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-8640) + x6), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-864) + x6), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (864 + x6), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (8640 + x6), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (9504 + x6), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (10368 + x6), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7r/c7rqveaaureuoawzbvu7jkgxtng5agwgyghr7zc2ak6gn4n4wvrp.py
# Topologically Sorted Source Nodes: [x_out_25, x_1490, x_1541], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1490 => relu_372
#   x_1541 => relu_385
#   x_out_25 => cat_33
# Graph fragment:
#   %cat_33 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_853, %add_858, %add_867, %add_872, %add_877], 1), kwargs = {})
#   %relu_372 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_33,), kwargs = {})
#   %relu_385 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_33,), kwargs = {})
triton_poi_fused_cat_relu_82 = async_compile.triton('triton_poi_fused_cat_relu_82', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_82', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_82(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 34560
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 4320
    x2 = xindex
    y1 = (yindex // 4320)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 864, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((864*x2) + (104544*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + (x2 + (121*y0) + (104544*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 1728, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((864*x2) + (104544*y1) + ((-864) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((864*x2) + (104544*y1) + ((-864) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 2592, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((864*x2) + (104544*y1) + ((-1728) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 3456, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((864*x2) + (104544*y1) + ((-2592) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((864*x2) + (104544*y1) + ((-2592) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 4320, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((864*x2) + (104544*y1) + ((-3456) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((864*x2) + (104544*y1) + ((-3456) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr1 + (y0 + (4320*x2) + (522720*y1)), tmp97, xmask & ymask)
    tl.store(out_ptr2 + (y0 + (4320*x2) + (522720*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/34/c34itirfisy6wgdqftvz5zlwq46a42flzeh6kktfpfayjdmdro2o.py
# Topologically Sorted Source Nodes: [x_out_26, x_1544], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   x_1544 => relu_386
#   x_out_26 => cat_34
# Graph fragment:
#   %cat_34 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_886, %add_891, %add_900, %add_905, %add_910], 1), kwargs = {})
#   %relu_386 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_34,), kwargs = {})
triton_poi_fused_cat_relu_83 = async_compile.triton('triton_poi_fused_cat_relu_83', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_83', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_83(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 34560
    xnumel = 121
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 4320
    x2 = xindex
    y1 = (yindex // 4320)
    y3 = yindex
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 864, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((864*x2) + (104544*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1, 1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr5 + ((864*x2) + (104544*y1) + y0), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 + tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1, 1], 1728, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr6 + ((864*x2) + (104544*y1) + ((-864) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp9
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tmp12 / tmp34
    tmp36 = tmp35 * tmp14
    tmp37 = tmp31 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-864) + y0, [XBLOCK, YBLOCK])), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.load(in_ptr11 + ((864*x2) + (104544*y1) + ((-864) + y0)), tmp28 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp28, tmp43, tmp44)
    tmp46 = tmp0 >= tmp26
    tmp47 = tl.full([1, 1], 2592, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tl.load(in_ptr12 + ((864*x2) + (104544*y1) + ((-1728) + y0)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp0 >= tmp47
    tmp52 = tl.full([1, 1], 3456, tl.int64)
    tmp53 = tmp0 < tmp52
    tmp54 = tmp51 & tmp53
    tmp55 = tl.load(in_ptr13 + ((864*x2) + (104544*y1) + ((-2592) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 - tmp56
    tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58 + tmp9
    tmp60 = libdevice.sqrt(tmp59)
    tmp61 = tmp12 / tmp60
    tmp62 = tmp61 * tmp14
    tmp63 = tmp57 * tmp62
    tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp63 * tmp64
    tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-2592) + y0, [XBLOCK, YBLOCK])), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 + tmp66
    tmp68 = tl.load(in_ptr18 + ((864*x2) + (104544*y1) + ((-2592) + y0)), tmp54 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp67 + tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp54, tmp69, tmp70)
    tmp72 = tmp0 >= tmp52
    tmp73 = tl.full([1, 1], 4320, tl.int64)
    tmp74 = tmp0 < tmp73
    tmp75 = tl.load(in_ptr19 + ((864*x2) + (104544*y1) + ((-3456) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp75 - tmp76
    tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp9
    tmp80 = libdevice.sqrt(tmp79)
    tmp81 = tmp12 / tmp80
    tmp82 = tmp81 * tmp14
    tmp83 = tmp77 * tmp82
    tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-3456) + y0, [XBLOCK, YBLOCK])), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 + tmp86
    tmp88 = tl.load(in_ptr24 + ((864*x2) + (104544*y1) + ((-3456) + y0)), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 + tmp88
    tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
    tmp91 = tl.where(tmp72, tmp89, tmp90)
    tmp92 = tl.where(tmp54, tmp71, tmp91)
    tmp93 = tl.where(tmp49, tmp50, tmp92)
    tmp94 = tl.where(tmp28, tmp45, tmp93)
    tmp95 = tl.where(tmp4, tmp24, tmp94)
    tmp96 = tl.full([1, 1], 0, tl.int32)
    tmp97 = triton_helpers.maximum(tmp96, tmp95)
    tl.store(out_ptr1 + (y0 + (4320*x2) + (522720*y1)), tmp97, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mt/cmtlwmck4y3bn4ru7eiv24wkcbc6mnw5zqew5npah3abvszafw2c.py
# Topologically Sorted Source Nodes: [x_out_27, x_1595, x_1596], Original ATen: [aten.cat, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_1595 => relu_399
#   x_1596 => mean_1
#   x_out_27 => cat_35
# Graph fragment:
#   %cat_35 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_919, %add_924, %add_933, %add_938, %add_943], 1), kwargs = {})
#   %relu_399 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%cat_35,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_399, [-1, -2], True), kwargs = {})
triton_red_fused_cat_mean_relu_84 = async_compile.triton('triton_red_fused_cat_mean_relu_84', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_mean_relu_84', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_mean_relu_84(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 34560
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4320
    x1 = (xindex // 4320)
    x3 = xindex
    _tmp99 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 864, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + ((864*r2) + (104544*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 - tmp6
        tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = 0.001
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.sqrt(tmp10)
        tmp12 = tl.full([1, 1], 1, tl.int32)
        tmp13 = tmp12 / tmp11
        tmp14 = 1.0
        tmp15 = tmp13 * tmp14
        tmp16 = tmp7 * tmp15
        tmp17 = tl.load(in_ptr3 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tmp16 * tmp17
        tmp19 = tl.load(in_ptr4 + (tl.broadcast_to(x0, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tmp18 + tmp19
        tmp21 = tl.load(in_ptr5 + ((864*r2) + (104544*x1) + x0), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 + tmp21
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp4, tmp22, tmp23)
        tmp25 = tmp0 >= tmp3
        tmp26 = tl.full([1, 1], 1728, tl.int64)
        tmp27 = tmp0 < tmp26
        tmp28 = tmp25 & tmp27
        tmp29 = tl.load(in_ptr6 + ((864*r2) + (104544*x1) + ((-864) + x0)), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-864) + x0, [XBLOCK, RBLOCK])), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tmp29 - tmp30
        tmp32 = tl.load(in_ptr8 + (tl.broadcast_to((-864) + x0, [XBLOCK, RBLOCK])), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tmp32 + tmp9
        tmp34 = libdevice.sqrt(tmp33)
        tmp35 = tmp12 / tmp34
        tmp36 = tmp35 * tmp14
        tmp37 = tmp31 * tmp36
        tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-864) + x0, [XBLOCK, RBLOCK])), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
        tmp39 = tmp37 * tmp38
        tmp40 = tl.load(in_ptr10 + (tl.broadcast_to((-864) + x0, [XBLOCK, RBLOCK])), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tmp39 + tmp40
        tmp42 = tl.load(in_ptr11 + ((864*r2) + (104544*x1) + ((-864) + x0)), rmask & tmp28 & xmask, eviction_policy='evict_last', other=0.0)
        tmp43 = tmp41 + tmp42
        tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
        tmp45 = tl.where(tmp28, tmp43, tmp44)
        tmp46 = tmp0 >= tmp26
        tmp47 = tl.full([1, 1], 2592, tl.int64)
        tmp48 = tmp0 < tmp47
        tmp49 = tmp46 & tmp48
        tmp50 = tl.load(in_ptr12 + ((864*r2) + (104544*x1) + ((-1728) + x0)), rmask & tmp49 & xmask, eviction_policy='evict_last', other=0.0)
        tmp51 = tmp0 >= tmp47
        tmp52 = tl.full([1, 1], 3456, tl.int64)
        tmp53 = tmp0 < tmp52
        tmp54 = tmp51 & tmp53
        tmp55 = tl.load(in_ptr13 + ((864*r2) + (104544*x1) + ((-2592) + x0)), rmask & tmp54 & xmask, eviction_policy='evict_last', other=0.0)
        tmp56 = tl.load(in_ptr14 + (tl.broadcast_to((-2592) + x0, [XBLOCK, RBLOCK])), rmask & tmp54 & xmask, eviction_policy='evict_last', other=0.0)
        tmp57 = tmp55 - tmp56
        tmp58 = tl.load(in_ptr15 + (tl.broadcast_to((-2592) + x0, [XBLOCK, RBLOCK])), rmask & tmp54 & xmask, eviction_policy='evict_last', other=0.0)
        tmp59 = tmp58 + tmp9
        tmp60 = libdevice.sqrt(tmp59)
        tmp61 = tmp12 / tmp60
        tmp62 = tmp61 * tmp14
        tmp63 = tmp57 * tmp62
        tmp64 = tl.load(in_ptr16 + (tl.broadcast_to((-2592) + x0, [XBLOCK, RBLOCK])), rmask & tmp54 & xmask, eviction_policy='evict_last', other=0.0)
        tmp65 = tmp63 * tmp64
        tmp66 = tl.load(in_ptr17 + (tl.broadcast_to((-2592) + x0, [XBLOCK, RBLOCK])), rmask & tmp54 & xmask, eviction_policy='evict_last', other=0.0)
        tmp67 = tmp65 + tmp66
        tmp68 = tl.load(in_ptr18 + ((864*r2) + (104544*x1) + ((-2592) + x0)), rmask & tmp54 & xmask, eviction_policy='evict_last', other=0.0)
        tmp69 = tmp67 + tmp68
        tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
        tmp71 = tl.where(tmp54, tmp69, tmp70)
        tmp72 = tmp0 >= tmp52
        tmp73 = tl.full([1, 1], 4320, tl.int64)
        tmp74 = tmp0 < tmp73
        tmp75 = tl.load(in_ptr19 + ((864*r2) + (104544*x1) + ((-3456) + x0)), rmask & tmp72 & xmask, eviction_policy='evict_last', other=0.0)
        tmp76 = tl.load(in_ptr20 + (tl.broadcast_to((-3456) + x0, [XBLOCK, RBLOCK])), rmask & tmp72 & xmask, eviction_policy='evict_last', other=0.0)
        tmp77 = tmp75 - tmp76
        tmp78 = tl.load(in_ptr21 + (tl.broadcast_to((-3456) + x0, [XBLOCK, RBLOCK])), rmask & tmp72 & xmask, eviction_policy='evict_last', other=0.0)
        tmp79 = tmp78 + tmp9
        tmp80 = libdevice.sqrt(tmp79)
        tmp81 = tmp12 / tmp80
        tmp82 = tmp81 * tmp14
        tmp83 = tmp77 * tmp82
        tmp84 = tl.load(in_ptr22 + (tl.broadcast_to((-3456) + x0, [XBLOCK, RBLOCK])), rmask & tmp72 & xmask, eviction_policy='evict_last', other=0.0)
        tmp85 = tmp83 * tmp84
        tmp86 = tl.load(in_ptr23 + (tl.broadcast_to((-3456) + x0, [XBLOCK, RBLOCK])), rmask & tmp72 & xmask, eviction_policy='evict_last', other=0.0)
        tmp87 = tmp85 + tmp86
        tmp88 = tl.load(in_ptr24 + ((864*r2) + (104544*x1) + ((-3456) + x0)), rmask & tmp72 & xmask, eviction_policy='evict_last', other=0.0)
        tmp89 = tmp87 + tmp88
        tmp90 = tl.full(tmp89.shape, 0.0, tmp89.dtype)
        tmp91 = tl.where(tmp72, tmp89, tmp90)
        tmp92 = tl.where(tmp54, tmp71, tmp91)
        tmp93 = tl.where(tmp49, tmp50, tmp92)
        tmp94 = tl.where(tmp28, tmp45, tmp93)
        tmp95 = tl.where(tmp4, tmp24, tmp94)
        tmp96 = tl.full([1, 1], 0, tl.int32)
        tmp97 = triton_helpers.maximum(tmp96, tmp95)
        tmp98 = tl.broadcast_to(tmp97, [XBLOCK, RBLOCK])
        tmp100 = _tmp99 + tmp98
        _tmp99 = tl.where(rmask & xmask, tmp100, _tmp99)
    tmp99 = tl.sum(_tmp99, 1)[:, None]
    tmp101 = 121.0
    tmp102 = tmp99 / tmp101
    tl.store(out_ptr2 + (x3), tmp102, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1 = args
    args.clear()
    assert_size_stride(arg0_1, (96, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 331, 331), (328683, 109561, 331, 1))
    assert_size_stride(arg2_1, (96, ), (1, ))
    assert_size_stride(arg3_1, (96, ), (1, ))
    assert_size_stride(arg4_1, (96, ), (1, ))
    assert_size_stride(arg5_1, (96, ), (1, ))
    assert_size_stride(arg6_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg7_1, (54, ), (1, ))
    assert_size_stride(arg8_1, (54, ), (1, ))
    assert_size_stride(arg9_1, (54, ), (1, ))
    assert_size_stride(arg10_1, (54, ), (1, ))
    assert_size_stride(arg11_1, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg12_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg13_1, (54, ), (1, ))
    assert_size_stride(arg14_1, (54, ), (1, ))
    assert_size_stride(arg15_1, (54, ), (1, ))
    assert_size_stride(arg16_1, (54, ), (1, ))
    assert_size_stride(arg17_1, (54, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg18_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg19_1, (54, ), (1, ))
    assert_size_stride(arg20_1, (54, ), (1, ))
    assert_size_stride(arg21_1, (54, ), (1, ))
    assert_size_stride(arg22_1, (54, ), (1, ))
    assert_size_stride(arg23_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg24_1, (54, ), (1, ))
    assert_size_stride(arg25_1, (54, ), (1, ))
    assert_size_stride(arg26_1, (54, ), (1, ))
    assert_size_stride(arg27_1, (54, ), (1, ))
    assert_size_stride(arg28_1, (54, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg29_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg30_1, (54, ), (1, ))
    assert_size_stride(arg31_1, (54, ), (1, ))
    assert_size_stride(arg32_1, (54, ), (1, ))
    assert_size_stride(arg33_1, (54, ), (1, ))
    assert_size_stride(arg34_1, (54, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg35_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg36_1, (54, ), (1, ))
    assert_size_stride(arg37_1, (54, ), (1, ))
    assert_size_stride(arg38_1, (54, ), (1, ))
    assert_size_stride(arg39_1, (54, ), (1, ))
    assert_size_stride(arg40_1, (54, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg41_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg42_1, (54, ), (1, ))
    assert_size_stride(arg43_1, (54, ), (1, ))
    assert_size_stride(arg44_1, (54, ), (1, ))
    assert_size_stride(arg45_1, (54, ), (1, ))
    assert_size_stride(arg46_1, (54, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg47_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg48_1, (54, ), (1, ))
    assert_size_stride(arg49_1, (54, ), (1, ))
    assert_size_stride(arg50_1, (54, ), (1, ))
    assert_size_stride(arg51_1, (54, ), (1, ))
    assert_size_stride(arg52_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg53_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg54_1, (54, ), (1, ))
    assert_size_stride(arg55_1, (54, ), (1, ))
    assert_size_stride(arg56_1, (54, ), (1, ))
    assert_size_stride(arg57_1, (54, ), (1, ))
    assert_size_stride(arg58_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg59_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg60_1, (54, ), (1, ))
    assert_size_stride(arg61_1, (54, ), (1, ))
    assert_size_stride(arg62_1, (54, ), (1, ))
    assert_size_stride(arg63_1, (54, ), (1, ))
    assert_size_stride(arg64_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg65_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg66_1, (54, ), (1, ))
    assert_size_stride(arg67_1, (54, ), (1, ))
    assert_size_stride(arg68_1, (54, ), (1, ))
    assert_size_stride(arg69_1, (54, ), (1, ))
    assert_size_stride(arg70_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg71_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg72_1, (54, ), (1, ))
    assert_size_stride(arg73_1, (54, ), (1, ))
    assert_size_stride(arg74_1, (54, ), (1, ))
    assert_size_stride(arg75_1, (54, ), (1, ))
    assert_size_stride(arg76_1, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg77_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg78_1, (54, ), (1, ))
    assert_size_stride(arg79_1, (54, ), (1, ))
    assert_size_stride(arg80_1, (54, ), (1, ))
    assert_size_stride(arg81_1, (54, ), (1, ))
    assert_size_stride(arg82_1, (54, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg83_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg84_1, (54, ), (1, ))
    assert_size_stride(arg85_1, (54, ), (1, ))
    assert_size_stride(arg86_1, (54, ), (1, ))
    assert_size_stride(arg87_1, (54, ), (1, ))
    assert_size_stride(arg88_1, (54, 54, 1, 1), (54, 1, 1, 1))
    assert_size_stride(arg89_1, (54, ), (1, ))
    assert_size_stride(arg90_1, (54, ), (1, ))
    assert_size_stride(arg91_1, (54, ), (1, ))
    assert_size_stride(arg92_1, (54, ), (1, ))
    assert_size_stride(arg93_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg94_1, (54, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg95_1, (108, ), (1, ))
    assert_size_stride(arg96_1, (108, ), (1, ))
    assert_size_stride(arg97_1, (108, ), (1, ))
    assert_size_stride(arg98_1, (108, ), (1, ))
    assert_size_stride(arg99_1, (108, 270, 1, 1), (270, 1, 1, 1))
    assert_size_stride(arg100_1, (108, ), (1, ))
    assert_size_stride(arg101_1, (108, ), (1, ))
    assert_size_stride(arg102_1, (108, ), (1, ))
    assert_size_stride(arg103_1, (108, ), (1, ))
    assert_size_stride(arg104_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg105_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg106_1, (108, ), (1, ))
    assert_size_stride(arg107_1, (108, ), (1, ))
    assert_size_stride(arg108_1, (108, ), (1, ))
    assert_size_stride(arg109_1, (108, ), (1, ))
    assert_size_stride(arg110_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg111_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg112_1, (108, ), (1, ))
    assert_size_stride(arg113_1, (108, ), (1, ))
    assert_size_stride(arg114_1, (108, ), (1, ))
    assert_size_stride(arg115_1, (108, ), (1, ))
    assert_size_stride(arg116_1, (108, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg117_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg118_1, (108, ), (1, ))
    assert_size_stride(arg119_1, (108, ), (1, ))
    assert_size_stride(arg120_1, (108, ), (1, ))
    assert_size_stride(arg121_1, (108, ), (1, ))
    assert_size_stride(arg122_1, (108, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg123_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg124_1, (108, ), (1, ))
    assert_size_stride(arg125_1, (108, ), (1, ))
    assert_size_stride(arg126_1, (108, ), (1, ))
    assert_size_stride(arg127_1, (108, ), (1, ))
    assert_size_stride(arg128_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg129_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg130_1, (108, ), (1, ))
    assert_size_stride(arg131_1, (108, ), (1, ))
    assert_size_stride(arg132_1, (108, ), (1, ))
    assert_size_stride(arg133_1, (108, ), (1, ))
    assert_size_stride(arg134_1, (108, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg135_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg136_1, (108, ), (1, ))
    assert_size_stride(arg137_1, (108, ), (1, ))
    assert_size_stride(arg138_1, (108, ), (1, ))
    assert_size_stride(arg139_1, (108, ), (1, ))
    assert_size_stride(arg140_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg141_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg142_1, (108, ), (1, ))
    assert_size_stride(arg143_1, (108, ), (1, ))
    assert_size_stride(arg144_1, (108, ), (1, ))
    assert_size_stride(arg145_1, (108, ), (1, ))
    assert_size_stride(arg146_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg147_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg148_1, (108, ), (1, ))
    assert_size_stride(arg149_1, (108, ), (1, ))
    assert_size_stride(arg150_1, (108, ), (1, ))
    assert_size_stride(arg151_1, (108, ), (1, ))
    assert_size_stride(arg152_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg153_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg154_1, (108, ), (1, ))
    assert_size_stride(arg155_1, (108, ), (1, ))
    assert_size_stride(arg156_1, (108, ), (1, ))
    assert_size_stride(arg157_1, (108, ), (1, ))
    assert_size_stride(arg158_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg159_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg160_1, (108, ), (1, ))
    assert_size_stride(arg161_1, (108, ), (1, ))
    assert_size_stride(arg162_1, (108, ), (1, ))
    assert_size_stride(arg163_1, (108, ), (1, ))
    assert_size_stride(arg164_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg165_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg166_1, (108, ), (1, ))
    assert_size_stride(arg167_1, (108, ), (1, ))
    assert_size_stride(arg168_1, (108, ), (1, ))
    assert_size_stride(arg169_1, (108, ), (1, ))
    assert_size_stride(arg170_1, (108, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg171_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg172_1, (108, ), (1, ))
    assert_size_stride(arg173_1, (108, ), (1, ))
    assert_size_stride(arg174_1, (108, ), (1, ))
    assert_size_stride(arg175_1, (108, ), (1, ))
    assert_size_stride(arg176_1, (108, 108, 1, 1), (108, 1, 1, 1))
    assert_size_stride(arg177_1, (108, ), (1, ))
    assert_size_stride(arg178_1, (108, ), (1, ))
    assert_size_stride(arg179_1, (108, ), (1, ))
    assert_size_stride(arg180_1, (108, ), (1, ))
    assert_size_stride(arg181_1, (108, 270, 1, 1), (270, 1, 1, 1))
    assert_size_stride(arg182_1, (108, 270, 1, 1), (270, 1, 1, 1))
    assert_size_stride(arg183_1, (216, ), (1, ))
    assert_size_stride(arg184_1, (216, ), (1, ))
    assert_size_stride(arg185_1, (216, ), (1, ))
    assert_size_stride(arg186_1, (216, ), (1, ))
    assert_size_stride(arg187_1, (216, 540, 1, 1), (540, 1, 1, 1))
    assert_size_stride(arg188_1, (216, ), (1, ))
    assert_size_stride(arg189_1, (216, ), (1, ))
    assert_size_stride(arg190_1, (216, ), (1, ))
    assert_size_stride(arg191_1, (216, ), (1, ))
    assert_size_stride(arg192_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg193_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg194_1, (216, ), (1, ))
    assert_size_stride(arg195_1, (216, ), (1, ))
    assert_size_stride(arg196_1, (216, ), (1, ))
    assert_size_stride(arg197_1, (216, ), (1, ))
    assert_size_stride(arg198_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg199_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg200_1, (216, ), (1, ))
    assert_size_stride(arg201_1, (216, ), (1, ))
    assert_size_stride(arg202_1, (216, ), (1, ))
    assert_size_stride(arg203_1, (216, ), (1, ))
    assert_size_stride(arg204_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg205_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg206_1, (216, ), (1, ))
    assert_size_stride(arg207_1, (216, ), (1, ))
    assert_size_stride(arg208_1, (216, ), (1, ))
    assert_size_stride(arg209_1, (216, ), (1, ))
    assert_size_stride(arg210_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg211_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg212_1, (216, ), (1, ))
    assert_size_stride(arg213_1, (216, ), (1, ))
    assert_size_stride(arg214_1, (216, ), (1, ))
    assert_size_stride(arg215_1, (216, ), (1, ))
    assert_size_stride(arg216_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg217_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg218_1, (216, ), (1, ))
    assert_size_stride(arg219_1, (216, ), (1, ))
    assert_size_stride(arg220_1, (216, ), (1, ))
    assert_size_stride(arg221_1, (216, ), (1, ))
    assert_size_stride(arg222_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg223_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg224_1, (216, ), (1, ))
    assert_size_stride(arg225_1, (216, ), (1, ))
    assert_size_stride(arg226_1, (216, ), (1, ))
    assert_size_stride(arg227_1, (216, ), (1, ))
    assert_size_stride(arg228_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg229_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg230_1, (216, ), (1, ))
    assert_size_stride(arg231_1, (216, ), (1, ))
    assert_size_stride(arg232_1, (216, ), (1, ))
    assert_size_stride(arg233_1, (216, ), (1, ))
    assert_size_stride(arg234_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg235_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg236_1, (216, ), (1, ))
    assert_size_stride(arg237_1, (216, ), (1, ))
    assert_size_stride(arg238_1, (216, ), (1, ))
    assert_size_stride(arg239_1, (216, ), (1, ))
    assert_size_stride(arg240_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg241_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg242_1, (216, ), (1, ))
    assert_size_stride(arg243_1, (216, ), (1, ))
    assert_size_stride(arg244_1, (216, ), (1, ))
    assert_size_stride(arg245_1, (216, ), (1, ))
    assert_size_stride(arg246_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg247_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg248_1, (216, ), (1, ))
    assert_size_stride(arg249_1, (216, ), (1, ))
    assert_size_stride(arg250_1, (216, ), (1, ))
    assert_size_stride(arg251_1, (216, ), (1, ))
    assert_size_stride(arg252_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg253_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg254_1, (216, ), (1, ))
    assert_size_stride(arg255_1, (216, ), (1, ))
    assert_size_stride(arg256_1, (216, ), (1, ))
    assert_size_stride(arg257_1, (216, ), (1, ))
    assert_size_stride(arg258_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg259_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg260_1, (216, ), (1, ))
    assert_size_stride(arg261_1, (216, ), (1, ))
    assert_size_stride(arg262_1, (216, ), (1, ))
    assert_size_stride(arg263_1, (216, ), (1, ))
    assert_size_stride(arg264_1, (216, 540, 1, 1), (540, 1, 1, 1))
    assert_size_stride(arg265_1, (216, ), (1, ))
    assert_size_stride(arg266_1, (216, ), (1, ))
    assert_size_stride(arg267_1, (216, ), (1, ))
    assert_size_stride(arg268_1, (216, ), (1, ))
    assert_size_stride(arg269_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg270_1, (216, ), (1, ))
    assert_size_stride(arg271_1, (216, ), (1, ))
    assert_size_stride(arg272_1, (216, ), (1, ))
    assert_size_stride(arg273_1, (216, ), (1, ))
    assert_size_stride(arg274_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg275_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg276_1, (216, ), (1, ))
    assert_size_stride(arg277_1, (216, ), (1, ))
    assert_size_stride(arg278_1, (216, ), (1, ))
    assert_size_stride(arg279_1, (216, ), (1, ))
    assert_size_stride(arg280_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg281_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg282_1, (216, ), (1, ))
    assert_size_stride(arg283_1, (216, ), (1, ))
    assert_size_stride(arg284_1, (216, ), (1, ))
    assert_size_stride(arg285_1, (216, ), (1, ))
    assert_size_stride(arg286_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg287_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg288_1, (216, ), (1, ))
    assert_size_stride(arg289_1, (216, ), (1, ))
    assert_size_stride(arg290_1, (216, ), (1, ))
    assert_size_stride(arg291_1, (216, ), (1, ))
    assert_size_stride(arg292_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg293_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg294_1, (216, ), (1, ))
    assert_size_stride(arg295_1, (216, ), (1, ))
    assert_size_stride(arg296_1, (216, ), (1, ))
    assert_size_stride(arg297_1, (216, ), (1, ))
    assert_size_stride(arg298_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg299_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg300_1, (216, ), (1, ))
    assert_size_stride(arg301_1, (216, ), (1, ))
    assert_size_stride(arg302_1, (216, ), (1, ))
    assert_size_stride(arg303_1, (216, ), (1, ))
    assert_size_stride(arg304_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg305_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg306_1, (216, ), (1, ))
    assert_size_stride(arg307_1, (216, ), (1, ))
    assert_size_stride(arg308_1, (216, ), (1, ))
    assert_size_stride(arg309_1, (216, ), (1, ))
    assert_size_stride(arg310_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg311_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg312_1, (216, ), (1, ))
    assert_size_stride(arg313_1, (216, ), (1, ))
    assert_size_stride(arg314_1, (216, ), (1, ))
    assert_size_stride(arg315_1, (216, ), (1, ))
    assert_size_stride(arg316_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg317_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg318_1, (216, ), (1, ))
    assert_size_stride(arg319_1, (216, ), (1, ))
    assert_size_stride(arg320_1, (216, ), (1, ))
    assert_size_stride(arg321_1, (216, ), (1, ))
    assert_size_stride(arg322_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg323_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg324_1, (216, ), (1, ))
    assert_size_stride(arg325_1, (216, ), (1, ))
    assert_size_stride(arg326_1, (216, ), (1, ))
    assert_size_stride(arg327_1, (216, ), (1, ))
    assert_size_stride(arg328_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg329_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg330_1, (216, ), (1, ))
    assert_size_stride(arg331_1, (216, ), (1, ))
    assert_size_stride(arg332_1, (216, ), (1, ))
    assert_size_stride(arg333_1, (216, ), (1, ))
    assert_size_stride(arg334_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg335_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg336_1, (216, ), (1, ))
    assert_size_stride(arg337_1, (216, ), (1, ))
    assert_size_stride(arg338_1, (216, ), (1, ))
    assert_size_stride(arg339_1, (216, ), (1, ))
    assert_size_stride(arg340_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg341_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg342_1, (216, ), (1, ))
    assert_size_stride(arg343_1, (216, ), (1, ))
    assert_size_stride(arg344_1, (216, ), (1, ))
    assert_size_stride(arg345_1, (216, ), (1, ))
    assert_size_stride(arg346_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg347_1, (216, ), (1, ))
    assert_size_stride(arg348_1, (216, ), (1, ))
    assert_size_stride(arg349_1, (216, ), (1, ))
    assert_size_stride(arg350_1, (216, ), (1, ))
    assert_size_stride(arg351_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg352_1, (216, ), (1, ))
    assert_size_stride(arg353_1, (216, ), (1, ))
    assert_size_stride(arg354_1, (216, ), (1, ))
    assert_size_stride(arg355_1, (216, ), (1, ))
    assert_size_stride(arg356_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg357_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg358_1, (216, ), (1, ))
    assert_size_stride(arg359_1, (216, ), (1, ))
    assert_size_stride(arg360_1, (216, ), (1, ))
    assert_size_stride(arg361_1, (216, ), (1, ))
    assert_size_stride(arg362_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg363_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg364_1, (216, ), (1, ))
    assert_size_stride(arg365_1, (216, ), (1, ))
    assert_size_stride(arg366_1, (216, ), (1, ))
    assert_size_stride(arg367_1, (216, ), (1, ))
    assert_size_stride(arg368_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg369_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg370_1, (216, ), (1, ))
    assert_size_stride(arg371_1, (216, ), (1, ))
    assert_size_stride(arg372_1, (216, ), (1, ))
    assert_size_stride(arg373_1, (216, ), (1, ))
    assert_size_stride(arg374_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg375_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg376_1, (216, ), (1, ))
    assert_size_stride(arg377_1, (216, ), (1, ))
    assert_size_stride(arg378_1, (216, ), (1, ))
    assert_size_stride(arg379_1, (216, ), (1, ))
    assert_size_stride(arg380_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg381_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg382_1, (216, ), (1, ))
    assert_size_stride(arg383_1, (216, ), (1, ))
    assert_size_stride(arg384_1, (216, ), (1, ))
    assert_size_stride(arg385_1, (216, ), (1, ))
    assert_size_stride(arg386_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg387_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg388_1, (216, ), (1, ))
    assert_size_stride(arg389_1, (216, ), (1, ))
    assert_size_stride(arg390_1, (216, ), (1, ))
    assert_size_stride(arg391_1, (216, ), (1, ))
    assert_size_stride(arg392_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg393_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg394_1, (216, ), (1, ))
    assert_size_stride(arg395_1, (216, ), (1, ))
    assert_size_stride(arg396_1, (216, ), (1, ))
    assert_size_stride(arg397_1, (216, ), (1, ))
    assert_size_stride(arg398_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg399_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg400_1, (216, ), (1, ))
    assert_size_stride(arg401_1, (216, ), (1, ))
    assert_size_stride(arg402_1, (216, ), (1, ))
    assert_size_stride(arg403_1, (216, ), (1, ))
    assert_size_stride(arg404_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg405_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg406_1, (216, ), (1, ))
    assert_size_stride(arg407_1, (216, ), (1, ))
    assert_size_stride(arg408_1, (216, ), (1, ))
    assert_size_stride(arg409_1, (216, ), (1, ))
    assert_size_stride(arg410_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg411_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg412_1, (216, ), (1, ))
    assert_size_stride(arg413_1, (216, ), (1, ))
    assert_size_stride(arg414_1, (216, ), (1, ))
    assert_size_stride(arg415_1, (216, ), (1, ))
    assert_size_stride(arg416_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg417_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg418_1, (216, ), (1, ))
    assert_size_stride(arg419_1, (216, ), (1, ))
    assert_size_stride(arg420_1, (216, ), (1, ))
    assert_size_stride(arg421_1, (216, ), (1, ))
    assert_size_stride(arg422_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg423_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg424_1, (216, ), (1, ))
    assert_size_stride(arg425_1, (216, ), (1, ))
    assert_size_stride(arg426_1, (216, ), (1, ))
    assert_size_stride(arg427_1, (216, ), (1, ))
    assert_size_stride(arg428_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg429_1, (216, ), (1, ))
    assert_size_stride(arg430_1, (216, ), (1, ))
    assert_size_stride(arg431_1, (216, ), (1, ))
    assert_size_stride(arg432_1, (216, ), (1, ))
    assert_size_stride(arg433_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg434_1, (216, ), (1, ))
    assert_size_stride(arg435_1, (216, ), (1, ))
    assert_size_stride(arg436_1, (216, ), (1, ))
    assert_size_stride(arg437_1, (216, ), (1, ))
    assert_size_stride(arg438_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg439_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg440_1, (216, ), (1, ))
    assert_size_stride(arg441_1, (216, ), (1, ))
    assert_size_stride(arg442_1, (216, ), (1, ))
    assert_size_stride(arg443_1, (216, ), (1, ))
    assert_size_stride(arg444_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg445_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg446_1, (216, ), (1, ))
    assert_size_stride(arg447_1, (216, ), (1, ))
    assert_size_stride(arg448_1, (216, ), (1, ))
    assert_size_stride(arg449_1, (216, ), (1, ))
    assert_size_stride(arg450_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg451_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg452_1, (216, ), (1, ))
    assert_size_stride(arg453_1, (216, ), (1, ))
    assert_size_stride(arg454_1, (216, ), (1, ))
    assert_size_stride(arg455_1, (216, ), (1, ))
    assert_size_stride(arg456_1, (216, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg457_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg458_1, (216, ), (1, ))
    assert_size_stride(arg459_1, (216, ), (1, ))
    assert_size_stride(arg460_1, (216, ), (1, ))
    assert_size_stride(arg461_1, (216, ), (1, ))
    assert_size_stride(arg462_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg463_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg464_1, (216, ), (1, ))
    assert_size_stride(arg465_1, (216, ), (1, ))
    assert_size_stride(arg466_1, (216, ), (1, ))
    assert_size_stride(arg467_1, (216, ), (1, ))
    assert_size_stride(arg468_1, (216, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg469_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg470_1, (216, ), (1, ))
    assert_size_stride(arg471_1, (216, ), (1, ))
    assert_size_stride(arg472_1, (216, ), (1, ))
    assert_size_stride(arg473_1, (216, ), (1, ))
    assert_size_stride(arg474_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg475_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg476_1, (216, ), (1, ))
    assert_size_stride(arg477_1, (216, ), (1, ))
    assert_size_stride(arg478_1, (216, ), (1, ))
    assert_size_stride(arg479_1, (216, ), (1, ))
    assert_size_stride(arg480_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg481_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg482_1, (216, ), (1, ))
    assert_size_stride(arg483_1, (216, ), (1, ))
    assert_size_stride(arg484_1, (216, ), (1, ))
    assert_size_stride(arg485_1, (216, ), (1, ))
    assert_size_stride(arg486_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg487_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg488_1, (216, ), (1, ))
    assert_size_stride(arg489_1, (216, ), (1, ))
    assert_size_stride(arg490_1, (216, ), (1, ))
    assert_size_stride(arg491_1, (216, ), (1, ))
    assert_size_stride(arg492_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg493_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg494_1, (216, ), (1, ))
    assert_size_stride(arg495_1, (216, ), (1, ))
    assert_size_stride(arg496_1, (216, ), (1, ))
    assert_size_stride(arg497_1, (216, ), (1, ))
    assert_size_stride(arg498_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg499_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg500_1, (216, ), (1, ))
    assert_size_stride(arg501_1, (216, ), (1, ))
    assert_size_stride(arg502_1, (216, ), (1, ))
    assert_size_stride(arg503_1, (216, ), (1, ))
    assert_size_stride(arg504_1, (216, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg505_1, (216, 216, 1, 1), (216, 1, 1, 1))
    assert_size_stride(arg506_1, (216, ), (1, ))
    assert_size_stride(arg507_1, (216, ), (1, ))
    assert_size_stride(arg508_1, (216, ), (1, ))
    assert_size_stride(arg509_1, (216, ), (1, ))
    assert_size_stride(arg510_1, (432, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg511_1, (432, ), (1, ))
    assert_size_stride(arg512_1, (432, ), (1, ))
    assert_size_stride(arg513_1, (432, ), (1, ))
    assert_size_stride(arg514_1, (432, ), (1, ))
    assert_size_stride(arg515_1, (432, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg516_1, (432, ), (1, ))
    assert_size_stride(arg517_1, (432, ), (1, ))
    assert_size_stride(arg518_1, (432, ), (1, ))
    assert_size_stride(arg519_1, (432, ), (1, ))
    assert_size_stride(arg520_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg521_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg522_1, (432, ), (1, ))
    assert_size_stride(arg523_1, (432, ), (1, ))
    assert_size_stride(arg524_1, (432, ), (1, ))
    assert_size_stride(arg525_1, (432, ), (1, ))
    assert_size_stride(arg526_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg527_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg528_1, (432, ), (1, ))
    assert_size_stride(arg529_1, (432, ), (1, ))
    assert_size_stride(arg530_1, (432, ), (1, ))
    assert_size_stride(arg531_1, (432, ), (1, ))
    assert_size_stride(arg532_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg533_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg534_1, (432, ), (1, ))
    assert_size_stride(arg535_1, (432, ), (1, ))
    assert_size_stride(arg536_1, (432, ), (1, ))
    assert_size_stride(arg537_1, (432, ), (1, ))
    assert_size_stride(arg538_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg539_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg540_1, (432, ), (1, ))
    assert_size_stride(arg541_1, (432, ), (1, ))
    assert_size_stride(arg542_1, (432, ), (1, ))
    assert_size_stride(arg543_1, (432, ), (1, ))
    assert_size_stride(arg544_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg545_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg546_1, (432, ), (1, ))
    assert_size_stride(arg547_1, (432, ), (1, ))
    assert_size_stride(arg548_1, (432, ), (1, ))
    assert_size_stride(arg549_1, (432, ), (1, ))
    assert_size_stride(arg550_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg551_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg552_1, (432, ), (1, ))
    assert_size_stride(arg553_1, (432, ), (1, ))
    assert_size_stride(arg554_1, (432, ), (1, ))
    assert_size_stride(arg555_1, (432, ), (1, ))
    assert_size_stride(arg556_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg557_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg558_1, (432, ), (1, ))
    assert_size_stride(arg559_1, (432, ), (1, ))
    assert_size_stride(arg560_1, (432, ), (1, ))
    assert_size_stride(arg561_1, (432, ), (1, ))
    assert_size_stride(arg562_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg563_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg564_1, (432, ), (1, ))
    assert_size_stride(arg565_1, (432, ), (1, ))
    assert_size_stride(arg566_1, (432, ), (1, ))
    assert_size_stride(arg567_1, (432, ), (1, ))
    assert_size_stride(arg568_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg569_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg570_1, (432, ), (1, ))
    assert_size_stride(arg571_1, (432, ), (1, ))
    assert_size_stride(arg572_1, (432, ), (1, ))
    assert_size_stride(arg573_1, (432, ), (1, ))
    assert_size_stride(arg574_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg575_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg576_1, (432, ), (1, ))
    assert_size_stride(arg577_1, (432, ), (1, ))
    assert_size_stride(arg578_1, (432, ), (1, ))
    assert_size_stride(arg579_1, (432, ), (1, ))
    assert_size_stride(arg580_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg581_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg582_1, (432, ), (1, ))
    assert_size_stride(arg583_1, (432, ), (1, ))
    assert_size_stride(arg584_1, (432, ), (1, ))
    assert_size_stride(arg585_1, (432, ), (1, ))
    assert_size_stride(arg586_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg587_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg588_1, (432, ), (1, ))
    assert_size_stride(arg589_1, (432, ), (1, ))
    assert_size_stride(arg590_1, (432, ), (1, ))
    assert_size_stride(arg591_1, (432, ), (1, ))
    assert_size_stride(arg592_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg593_1, (432, ), (1, ))
    assert_size_stride(arg594_1, (432, ), (1, ))
    assert_size_stride(arg595_1, (432, ), (1, ))
    assert_size_stride(arg596_1, (432, ), (1, ))
    assert_size_stride(arg597_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg598_1, (216, 1080, 1, 1), (1080, 1, 1, 1))
    assert_size_stride(arg599_1, (432, ), (1, ))
    assert_size_stride(arg600_1, (432, ), (1, ))
    assert_size_stride(arg601_1, (432, ), (1, ))
    assert_size_stride(arg602_1, (432, ), (1, ))
    assert_size_stride(arg603_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg604_1, (432, ), (1, ))
    assert_size_stride(arg605_1, (432, ), (1, ))
    assert_size_stride(arg606_1, (432, ), (1, ))
    assert_size_stride(arg607_1, (432, ), (1, ))
    assert_size_stride(arg608_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg609_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg610_1, (432, ), (1, ))
    assert_size_stride(arg611_1, (432, ), (1, ))
    assert_size_stride(arg612_1, (432, ), (1, ))
    assert_size_stride(arg613_1, (432, ), (1, ))
    assert_size_stride(arg614_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg615_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg616_1, (432, ), (1, ))
    assert_size_stride(arg617_1, (432, ), (1, ))
    assert_size_stride(arg618_1, (432, ), (1, ))
    assert_size_stride(arg619_1, (432, ), (1, ))
    assert_size_stride(arg620_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg621_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg622_1, (432, ), (1, ))
    assert_size_stride(arg623_1, (432, ), (1, ))
    assert_size_stride(arg624_1, (432, ), (1, ))
    assert_size_stride(arg625_1, (432, ), (1, ))
    assert_size_stride(arg626_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg627_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg628_1, (432, ), (1, ))
    assert_size_stride(arg629_1, (432, ), (1, ))
    assert_size_stride(arg630_1, (432, ), (1, ))
    assert_size_stride(arg631_1, (432, ), (1, ))
    assert_size_stride(arg632_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg633_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg634_1, (432, ), (1, ))
    assert_size_stride(arg635_1, (432, ), (1, ))
    assert_size_stride(arg636_1, (432, ), (1, ))
    assert_size_stride(arg637_1, (432, ), (1, ))
    assert_size_stride(arg638_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg639_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg640_1, (432, ), (1, ))
    assert_size_stride(arg641_1, (432, ), (1, ))
    assert_size_stride(arg642_1, (432, ), (1, ))
    assert_size_stride(arg643_1, (432, ), (1, ))
    assert_size_stride(arg644_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg645_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg646_1, (432, ), (1, ))
    assert_size_stride(arg647_1, (432, ), (1, ))
    assert_size_stride(arg648_1, (432, ), (1, ))
    assert_size_stride(arg649_1, (432, ), (1, ))
    assert_size_stride(arg650_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg651_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg652_1, (432, ), (1, ))
    assert_size_stride(arg653_1, (432, ), (1, ))
    assert_size_stride(arg654_1, (432, ), (1, ))
    assert_size_stride(arg655_1, (432, ), (1, ))
    assert_size_stride(arg656_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg657_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg658_1, (432, ), (1, ))
    assert_size_stride(arg659_1, (432, ), (1, ))
    assert_size_stride(arg660_1, (432, ), (1, ))
    assert_size_stride(arg661_1, (432, ), (1, ))
    assert_size_stride(arg662_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg663_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg664_1, (432, ), (1, ))
    assert_size_stride(arg665_1, (432, ), (1, ))
    assert_size_stride(arg666_1, (432, ), (1, ))
    assert_size_stride(arg667_1, (432, ), (1, ))
    assert_size_stride(arg668_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg669_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg670_1, (432, ), (1, ))
    assert_size_stride(arg671_1, (432, ), (1, ))
    assert_size_stride(arg672_1, (432, ), (1, ))
    assert_size_stride(arg673_1, (432, ), (1, ))
    assert_size_stride(arg674_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg675_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg676_1, (432, ), (1, ))
    assert_size_stride(arg677_1, (432, ), (1, ))
    assert_size_stride(arg678_1, (432, ), (1, ))
    assert_size_stride(arg679_1, (432, ), (1, ))
    assert_size_stride(arg680_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg681_1, (432, ), (1, ))
    assert_size_stride(arg682_1, (432, ), (1, ))
    assert_size_stride(arg683_1, (432, ), (1, ))
    assert_size_stride(arg684_1, (432, ), (1, ))
    assert_size_stride(arg685_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg686_1, (432, ), (1, ))
    assert_size_stride(arg687_1, (432, ), (1, ))
    assert_size_stride(arg688_1, (432, ), (1, ))
    assert_size_stride(arg689_1, (432, ), (1, ))
    assert_size_stride(arg690_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg691_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg692_1, (432, ), (1, ))
    assert_size_stride(arg693_1, (432, ), (1, ))
    assert_size_stride(arg694_1, (432, ), (1, ))
    assert_size_stride(arg695_1, (432, ), (1, ))
    assert_size_stride(arg696_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg697_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg698_1, (432, ), (1, ))
    assert_size_stride(arg699_1, (432, ), (1, ))
    assert_size_stride(arg700_1, (432, ), (1, ))
    assert_size_stride(arg701_1, (432, ), (1, ))
    assert_size_stride(arg702_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg703_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg704_1, (432, ), (1, ))
    assert_size_stride(arg705_1, (432, ), (1, ))
    assert_size_stride(arg706_1, (432, ), (1, ))
    assert_size_stride(arg707_1, (432, ), (1, ))
    assert_size_stride(arg708_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg709_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg710_1, (432, ), (1, ))
    assert_size_stride(arg711_1, (432, ), (1, ))
    assert_size_stride(arg712_1, (432, ), (1, ))
    assert_size_stride(arg713_1, (432, ), (1, ))
    assert_size_stride(arg714_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg715_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg716_1, (432, ), (1, ))
    assert_size_stride(arg717_1, (432, ), (1, ))
    assert_size_stride(arg718_1, (432, ), (1, ))
    assert_size_stride(arg719_1, (432, ), (1, ))
    assert_size_stride(arg720_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg721_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg722_1, (432, ), (1, ))
    assert_size_stride(arg723_1, (432, ), (1, ))
    assert_size_stride(arg724_1, (432, ), (1, ))
    assert_size_stride(arg725_1, (432, ), (1, ))
    assert_size_stride(arg726_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg727_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg728_1, (432, ), (1, ))
    assert_size_stride(arg729_1, (432, ), (1, ))
    assert_size_stride(arg730_1, (432, ), (1, ))
    assert_size_stride(arg731_1, (432, ), (1, ))
    assert_size_stride(arg732_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg733_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg734_1, (432, ), (1, ))
    assert_size_stride(arg735_1, (432, ), (1, ))
    assert_size_stride(arg736_1, (432, ), (1, ))
    assert_size_stride(arg737_1, (432, ), (1, ))
    assert_size_stride(arg738_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg739_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg740_1, (432, ), (1, ))
    assert_size_stride(arg741_1, (432, ), (1, ))
    assert_size_stride(arg742_1, (432, ), (1, ))
    assert_size_stride(arg743_1, (432, ), (1, ))
    assert_size_stride(arg744_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg745_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg746_1, (432, ), (1, ))
    assert_size_stride(arg747_1, (432, ), (1, ))
    assert_size_stride(arg748_1, (432, ), (1, ))
    assert_size_stride(arg749_1, (432, ), (1, ))
    assert_size_stride(arg750_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg751_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg752_1, (432, ), (1, ))
    assert_size_stride(arg753_1, (432, ), (1, ))
    assert_size_stride(arg754_1, (432, ), (1, ))
    assert_size_stride(arg755_1, (432, ), (1, ))
    assert_size_stride(arg756_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg757_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg758_1, (432, ), (1, ))
    assert_size_stride(arg759_1, (432, ), (1, ))
    assert_size_stride(arg760_1, (432, ), (1, ))
    assert_size_stride(arg761_1, (432, ), (1, ))
    assert_size_stride(arg762_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg763_1, (432, ), (1, ))
    assert_size_stride(arg764_1, (432, ), (1, ))
    assert_size_stride(arg765_1, (432, ), (1, ))
    assert_size_stride(arg766_1, (432, ), (1, ))
    assert_size_stride(arg767_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg768_1, (432, ), (1, ))
    assert_size_stride(arg769_1, (432, ), (1, ))
    assert_size_stride(arg770_1, (432, ), (1, ))
    assert_size_stride(arg771_1, (432, ), (1, ))
    assert_size_stride(arg772_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg773_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg774_1, (432, ), (1, ))
    assert_size_stride(arg775_1, (432, ), (1, ))
    assert_size_stride(arg776_1, (432, ), (1, ))
    assert_size_stride(arg777_1, (432, ), (1, ))
    assert_size_stride(arg778_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg779_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg780_1, (432, ), (1, ))
    assert_size_stride(arg781_1, (432, ), (1, ))
    assert_size_stride(arg782_1, (432, ), (1, ))
    assert_size_stride(arg783_1, (432, ), (1, ))
    assert_size_stride(arg784_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg785_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg786_1, (432, ), (1, ))
    assert_size_stride(arg787_1, (432, ), (1, ))
    assert_size_stride(arg788_1, (432, ), (1, ))
    assert_size_stride(arg789_1, (432, ), (1, ))
    assert_size_stride(arg790_1, (432, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg791_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg792_1, (432, ), (1, ))
    assert_size_stride(arg793_1, (432, ), (1, ))
    assert_size_stride(arg794_1, (432, ), (1, ))
    assert_size_stride(arg795_1, (432, ), (1, ))
    assert_size_stride(arg796_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg797_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg798_1, (432, ), (1, ))
    assert_size_stride(arg799_1, (432, ), (1, ))
    assert_size_stride(arg800_1, (432, ), (1, ))
    assert_size_stride(arg801_1, (432, ), (1, ))
    assert_size_stride(arg802_1, (432, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg803_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg804_1, (432, ), (1, ))
    assert_size_stride(arg805_1, (432, ), (1, ))
    assert_size_stride(arg806_1, (432, ), (1, ))
    assert_size_stride(arg807_1, (432, ), (1, ))
    assert_size_stride(arg808_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg809_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg810_1, (432, ), (1, ))
    assert_size_stride(arg811_1, (432, ), (1, ))
    assert_size_stride(arg812_1, (432, ), (1, ))
    assert_size_stride(arg813_1, (432, ), (1, ))
    assert_size_stride(arg814_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg815_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg816_1, (432, ), (1, ))
    assert_size_stride(arg817_1, (432, ), (1, ))
    assert_size_stride(arg818_1, (432, ), (1, ))
    assert_size_stride(arg819_1, (432, ), (1, ))
    assert_size_stride(arg820_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg821_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg822_1, (432, ), (1, ))
    assert_size_stride(arg823_1, (432, ), (1, ))
    assert_size_stride(arg824_1, (432, ), (1, ))
    assert_size_stride(arg825_1, (432, ), (1, ))
    assert_size_stride(arg826_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg827_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg828_1, (432, ), (1, ))
    assert_size_stride(arg829_1, (432, ), (1, ))
    assert_size_stride(arg830_1, (432, ), (1, ))
    assert_size_stride(arg831_1, (432, ), (1, ))
    assert_size_stride(arg832_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg833_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg834_1, (432, ), (1, ))
    assert_size_stride(arg835_1, (432, ), (1, ))
    assert_size_stride(arg836_1, (432, ), (1, ))
    assert_size_stride(arg837_1, (432, ), (1, ))
    assert_size_stride(arg838_1, (432, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg839_1, (432, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg840_1, (432, ), (1, ))
    assert_size_stride(arg841_1, (432, ), (1, ))
    assert_size_stride(arg842_1, (432, ), (1, ))
    assert_size_stride(arg843_1, (432, ), (1, ))
    assert_size_stride(arg844_1, (864, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg845_1, (864, ), (1, ))
    assert_size_stride(arg846_1, (864, ), (1, ))
    assert_size_stride(arg847_1, (864, ), (1, ))
    assert_size_stride(arg848_1, (864, ), (1, ))
    assert_size_stride(arg849_1, (864, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg850_1, (864, ), (1, ))
    assert_size_stride(arg851_1, (864, ), (1, ))
    assert_size_stride(arg852_1, (864, ), (1, ))
    assert_size_stride(arg853_1, (864, ), (1, ))
    assert_size_stride(arg854_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg855_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg856_1, (864, ), (1, ))
    assert_size_stride(arg857_1, (864, ), (1, ))
    assert_size_stride(arg858_1, (864, ), (1, ))
    assert_size_stride(arg859_1, (864, ), (1, ))
    assert_size_stride(arg860_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg861_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg862_1, (864, ), (1, ))
    assert_size_stride(arg863_1, (864, ), (1, ))
    assert_size_stride(arg864_1, (864, ), (1, ))
    assert_size_stride(arg865_1, (864, ), (1, ))
    assert_size_stride(arg866_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg867_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg868_1, (864, ), (1, ))
    assert_size_stride(arg869_1, (864, ), (1, ))
    assert_size_stride(arg870_1, (864, ), (1, ))
    assert_size_stride(arg871_1, (864, ), (1, ))
    assert_size_stride(arg872_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg873_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg874_1, (864, ), (1, ))
    assert_size_stride(arg875_1, (864, ), (1, ))
    assert_size_stride(arg876_1, (864, ), (1, ))
    assert_size_stride(arg877_1, (864, ), (1, ))
    assert_size_stride(arg878_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg879_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg880_1, (864, ), (1, ))
    assert_size_stride(arg881_1, (864, ), (1, ))
    assert_size_stride(arg882_1, (864, ), (1, ))
    assert_size_stride(arg883_1, (864, ), (1, ))
    assert_size_stride(arg884_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg885_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg886_1, (864, ), (1, ))
    assert_size_stride(arg887_1, (864, ), (1, ))
    assert_size_stride(arg888_1, (864, ), (1, ))
    assert_size_stride(arg889_1, (864, ), (1, ))
    assert_size_stride(arg890_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg891_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg892_1, (864, ), (1, ))
    assert_size_stride(arg893_1, (864, ), (1, ))
    assert_size_stride(arg894_1, (864, ), (1, ))
    assert_size_stride(arg895_1, (864, ), (1, ))
    assert_size_stride(arg896_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg897_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg898_1, (864, ), (1, ))
    assert_size_stride(arg899_1, (864, ), (1, ))
    assert_size_stride(arg900_1, (864, ), (1, ))
    assert_size_stride(arg901_1, (864, ), (1, ))
    assert_size_stride(arg902_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg903_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg904_1, (864, ), (1, ))
    assert_size_stride(arg905_1, (864, ), (1, ))
    assert_size_stride(arg906_1, (864, ), (1, ))
    assert_size_stride(arg907_1, (864, ), (1, ))
    assert_size_stride(arg908_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg909_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg910_1, (864, ), (1, ))
    assert_size_stride(arg911_1, (864, ), (1, ))
    assert_size_stride(arg912_1, (864, ), (1, ))
    assert_size_stride(arg913_1, (864, ), (1, ))
    assert_size_stride(arg914_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg915_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg916_1, (864, ), (1, ))
    assert_size_stride(arg917_1, (864, ), (1, ))
    assert_size_stride(arg918_1, (864, ), (1, ))
    assert_size_stride(arg919_1, (864, ), (1, ))
    assert_size_stride(arg920_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg921_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg922_1, (864, ), (1, ))
    assert_size_stride(arg923_1, (864, ), (1, ))
    assert_size_stride(arg924_1, (864, ), (1, ))
    assert_size_stride(arg925_1, (864, ), (1, ))
    assert_size_stride(arg926_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg927_1, (864, ), (1, ))
    assert_size_stride(arg928_1, (864, ), (1, ))
    assert_size_stride(arg929_1, (864, ), (1, ))
    assert_size_stride(arg930_1, (864, ), (1, ))
    assert_size_stride(arg931_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg932_1, (432, 2160, 1, 1), (2160, 1, 1, 1))
    assert_size_stride(arg933_1, (864, ), (1, ))
    assert_size_stride(arg934_1, (864, ), (1, ))
    assert_size_stride(arg935_1, (864, ), (1, ))
    assert_size_stride(arg936_1, (864, ), (1, ))
    assert_size_stride(arg937_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg938_1, (864, ), (1, ))
    assert_size_stride(arg939_1, (864, ), (1, ))
    assert_size_stride(arg940_1, (864, ), (1, ))
    assert_size_stride(arg941_1, (864, ), (1, ))
    assert_size_stride(arg942_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg943_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg944_1, (864, ), (1, ))
    assert_size_stride(arg945_1, (864, ), (1, ))
    assert_size_stride(arg946_1, (864, ), (1, ))
    assert_size_stride(arg947_1, (864, ), (1, ))
    assert_size_stride(arg948_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg949_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg950_1, (864, ), (1, ))
    assert_size_stride(arg951_1, (864, ), (1, ))
    assert_size_stride(arg952_1, (864, ), (1, ))
    assert_size_stride(arg953_1, (864, ), (1, ))
    assert_size_stride(arg954_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg955_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg956_1, (864, ), (1, ))
    assert_size_stride(arg957_1, (864, ), (1, ))
    assert_size_stride(arg958_1, (864, ), (1, ))
    assert_size_stride(arg959_1, (864, ), (1, ))
    assert_size_stride(arg960_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg961_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg962_1, (864, ), (1, ))
    assert_size_stride(arg963_1, (864, ), (1, ))
    assert_size_stride(arg964_1, (864, ), (1, ))
    assert_size_stride(arg965_1, (864, ), (1, ))
    assert_size_stride(arg966_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg967_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg968_1, (864, ), (1, ))
    assert_size_stride(arg969_1, (864, ), (1, ))
    assert_size_stride(arg970_1, (864, ), (1, ))
    assert_size_stride(arg971_1, (864, ), (1, ))
    assert_size_stride(arg972_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg973_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg974_1, (864, ), (1, ))
    assert_size_stride(arg975_1, (864, ), (1, ))
    assert_size_stride(arg976_1, (864, ), (1, ))
    assert_size_stride(arg977_1, (864, ), (1, ))
    assert_size_stride(arg978_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg979_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg980_1, (864, ), (1, ))
    assert_size_stride(arg981_1, (864, ), (1, ))
    assert_size_stride(arg982_1, (864, ), (1, ))
    assert_size_stride(arg983_1, (864, ), (1, ))
    assert_size_stride(arg984_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg985_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg986_1, (864, ), (1, ))
    assert_size_stride(arg987_1, (864, ), (1, ))
    assert_size_stride(arg988_1, (864, ), (1, ))
    assert_size_stride(arg989_1, (864, ), (1, ))
    assert_size_stride(arg990_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg991_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg992_1, (864, ), (1, ))
    assert_size_stride(arg993_1, (864, ), (1, ))
    assert_size_stride(arg994_1, (864, ), (1, ))
    assert_size_stride(arg995_1, (864, ), (1, ))
    assert_size_stride(arg996_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg997_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg998_1, (864, ), (1, ))
    assert_size_stride(arg999_1, (864, ), (1, ))
    assert_size_stride(arg1000_1, (864, ), (1, ))
    assert_size_stride(arg1001_1, (864, ), (1, ))
    assert_size_stride(arg1002_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1003_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1004_1, (864, ), (1, ))
    assert_size_stride(arg1005_1, (864, ), (1, ))
    assert_size_stride(arg1006_1, (864, ), (1, ))
    assert_size_stride(arg1007_1, (864, ), (1, ))
    assert_size_stride(arg1008_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1009_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1010_1, (864, ), (1, ))
    assert_size_stride(arg1011_1, (864, ), (1, ))
    assert_size_stride(arg1012_1, (864, ), (1, ))
    assert_size_stride(arg1013_1, (864, ), (1, ))
    assert_size_stride(arg1014_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg1015_1, (864, ), (1, ))
    assert_size_stride(arg1016_1, (864, ), (1, ))
    assert_size_stride(arg1017_1, (864, ), (1, ))
    assert_size_stride(arg1018_1, (864, ), (1, ))
    assert_size_stride(arg1019_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg1020_1, (864, ), (1, ))
    assert_size_stride(arg1021_1, (864, ), (1, ))
    assert_size_stride(arg1022_1, (864, ), (1, ))
    assert_size_stride(arg1023_1, (864, ), (1, ))
    assert_size_stride(arg1024_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1025_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1026_1, (864, ), (1, ))
    assert_size_stride(arg1027_1, (864, ), (1, ))
    assert_size_stride(arg1028_1, (864, ), (1, ))
    assert_size_stride(arg1029_1, (864, ), (1, ))
    assert_size_stride(arg1030_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1031_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1032_1, (864, ), (1, ))
    assert_size_stride(arg1033_1, (864, ), (1, ))
    assert_size_stride(arg1034_1, (864, ), (1, ))
    assert_size_stride(arg1035_1, (864, ), (1, ))
    assert_size_stride(arg1036_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg1037_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1038_1, (864, ), (1, ))
    assert_size_stride(arg1039_1, (864, ), (1, ))
    assert_size_stride(arg1040_1, (864, ), (1, ))
    assert_size_stride(arg1041_1, (864, ), (1, ))
    assert_size_stride(arg1042_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg1043_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1044_1, (864, ), (1, ))
    assert_size_stride(arg1045_1, (864, ), (1, ))
    assert_size_stride(arg1046_1, (864, ), (1, ))
    assert_size_stride(arg1047_1, (864, ), (1, ))
    assert_size_stride(arg1048_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1049_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1050_1, (864, ), (1, ))
    assert_size_stride(arg1051_1, (864, ), (1, ))
    assert_size_stride(arg1052_1, (864, ), (1, ))
    assert_size_stride(arg1053_1, (864, ), (1, ))
    assert_size_stride(arg1054_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1055_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1056_1, (864, ), (1, ))
    assert_size_stride(arg1057_1, (864, ), (1, ))
    assert_size_stride(arg1058_1, (864, ), (1, ))
    assert_size_stride(arg1059_1, (864, ), (1, ))
    assert_size_stride(arg1060_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1061_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1062_1, (864, ), (1, ))
    assert_size_stride(arg1063_1, (864, ), (1, ))
    assert_size_stride(arg1064_1, (864, ), (1, ))
    assert_size_stride(arg1065_1, (864, ), (1, ))
    assert_size_stride(arg1066_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1067_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1068_1, (864, ), (1, ))
    assert_size_stride(arg1069_1, (864, ), (1, ))
    assert_size_stride(arg1070_1, (864, ), (1, ))
    assert_size_stride(arg1071_1, (864, ), (1, ))
    assert_size_stride(arg1072_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1073_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1074_1, (864, ), (1, ))
    assert_size_stride(arg1075_1, (864, ), (1, ))
    assert_size_stride(arg1076_1, (864, ), (1, ))
    assert_size_stride(arg1077_1, (864, ), (1, ))
    assert_size_stride(arg1078_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1079_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1080_1, (864, ), (1, ))
    assert_size_stride(arg1081_1, (864, ), (1, ))
    assert_size_stride(arg1082_1, (864, ), (1, ))
    assert_size_stride(arg1083_1, (864, ), (1, ))
    assert_size_stride(arg1084_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1085_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1086_1, (864, ), (1, ))
    assert_size_stride(arg1087_1, (864, ), (1, ))
    assert_size_stride(arg1088_1, (864, ), (1, ))
    assert_size_stride(arg1089_1, (864, ), (1, ))
    assert_size_stride(arg1090_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1091_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1092_1, (864, ), (1, ))
    assert_size_stride(arg1093_1, (864, ), (1, ))
    assert_size_stride(arg1094_1, (864, ), (1, ))
    assert_size_stride(arg1095_1, (864, ), (1, ))
    assert_size_stride(arg1096_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg1097_1, (864, ), (1, ))
    assert_size_stride(arg1098_1, (864, ), (1, ))
    assert_size_stride(arg1099_1, (864, ), (1, ))
    assert_size_stride(arg1100_1, (864, ), (1, ))
    assert_size_stride(arg1101_1, (864, 4320, 1, 1), (4320, 1, 1, 1))
    assert_size_stride(arg1102_1, (864, ), (1, ))
    assert_size_stride(arg1103_1, (864, ), (1, ))
    assert_size_stride(arg1104_1, (864, ), (1, ))
    assert_size_stride(arg1105_1, (864, ), (1, ))
    assert_size_stride(arg1106_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1107_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1108_1, (864, ), (1, ))
    assert_size_stride(arg1109_1, (864, ), (1, ))
    assert_size_stride(arg1110_1, (864, ), (1, ))
    assert_size_stride(arg1111_1, (864, ), (1, ))
    assert_size_stride(arg1112_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1113_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1114_1, (864, ), (1, ))
    assert_size_stride(arg1115_1, (864, ), (1, ))
    assert_size_stride(arg1116_1, (864, ), (1, ))
    assert_size_stride(arg1117_1, (864, ), (1, ))
    assert_size_stride(arg1118_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg1119_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1120_1, (864, ), (1, ))
    assert_size_stride(arg1121_1, (864, ), (1, ))
    assert_size_stride(arg1122_1, (864, ), (1, ))
    assert_size_stride(arg1123_1, (864, ), (1, ))
    assert_size_stride(arg1124_1, (864, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg1125_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1126_1, (864, ), (1, ))
    assert_size_stride(arg1127_1, (864, ), (1, ))
    assert_size_stride(arg1128_1, (864, ), (1, ))
    assert_size_stride(arg1129_1, (864, ), (1, ))
    assert_size_stride(arg1130_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1131_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1132_1, (864, ), (1, ))
    assert_size_stride(arg1133_1, (864, ), (1, ))
    assert_size_stride(arg1134_1, (864, ), (1, ))
    assert_size_stride(arg1135_1, (864, ), (1, ))
    assert_size_stride(arg1136_1, (864, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg1137_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1138_1, (864, ), (1, ))
    assert_size_stride(arg1139_1, (864, ), (1, ))
    assert_size_stride(arg1140_1, (864, ), (1, ))
    assert_size_stride(arg1141_1, (864, ), (1, ))
    assert_size_stride(arg1142_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1143_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1144_1, (864, ), (1, ))
    assert_size_stride(arg1145_1, (864, ), (1, ))
    assert_size_stride(arg1146_1, (864, ), (1, ))
    assert_size_stride(arg1147_1, (864, ), (1, ))
    assert_size_stride(arg1148_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1149_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1150_1, (864, ), (1, ))
    assert_size_stride(arg1151_1, (864, ), (1, ))
    assert_size_stride(arg1152_1, (864, ), (1, ))
    assert_size_stride(arg1153_1, (864, ), (1, ))
    assert_size_stride(arg1154_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1155_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1156_1, (864, ), (1, ))
    assert_size_stride(arg1157_1, (864, ), (1, ))
    assert_size_stride(arg1158_1, (864, ), (1, ))
    assert_size_stride(arg1159_1, (864, ), (1, ))
    assert_size_stride(arg1160_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1161_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1162_1, (864, ), (1, ))
    assert_size_stride(arg1163_1, (864, ), (1, ))
    assert_size_stride(arg1164_1, (864, ), (1, ))
    assert_size_stride(arg1165_1, (864, ), (1, ))
    assert_size_stride(arg1166_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1167_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1168_1, (864, ), (1, ))
    assert_size_stride(arg1169_1, (864, ), (1, ))
    assert_size_stride(arg1170_1, (864, ), (1, ))
    assert_size_stride(arg1171_1, (864, ), (1, ))
    assert_size_stride(arg1172_1, (864, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg1173_1, (864, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg1174_1, (864, ), (1, ))
    assert_size_stride(arg1175_1, (864, ), (1, ))
    assert_size_stride(arg1176_1, (864, ), (1, ))
    assert_size_stride(arg1177_1, (864, ), (1, ))
    assert_size_stride(arg1178_1, (1000, 4320), (4320, 1))
    assert_size_stride(arg1179_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 331, 331), (328683, 1, 993, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_800], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 109561, grid=grid(24, 109561), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((96, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_800], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 288, 9, grid=grid(288, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_800], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 96, 165, 165), (2613600, 1, 15840, 96))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf5 = empty_strided_cuda((8, 96, 165, 165), (2613600, 1, 15840, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_801, x_802], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, buf5, 20908800, grid=grid(20908800), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((8, 96, 83, 83), (661344, 1, 7968, 96), torch.float32)
        buf10 = empty_strided_cuda((8, 96, 83, 83), (661344, 1, 7968, 96), torch.float32)
        buf12 = empty_strided_cuda((8, 96, 83, 83), (661344, 1, 7968, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_814, input_24, x_865, input_27, input_29, input_30], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices, aten.relu, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_constant_pad_nd_max_pool2d_with_indices_relu_3.run(buf3, buf4, buf10, buf12, 5290752, grid=grid(5290752), stream=stream0)
        # Topologically Sorted Source Nodes: [x_802, x_803], Original ATen: [aten.relu, aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 54, 165, 165), (1470150, 1, 8910, 54))
        del arg6_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf55 = empty_strided_cuda((8, 54, 165, 165), (1470150, 1, 8910, 54), torch.float32)
        # Topologically Sorted Source Nodes: [x_804, x_861], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf7, arg7_1, arg8_1, arg9_1, arg10_1, buf55, 11761200, grid=grid(11761200), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf8 = empty_strided_cuda((8, 54, 83, 83), (372006, 1, 4482, 54), torch.float32)
        buf9 = empty_strided_cuda((8, 54, 83, 83), (372006, 1, 4482, 54), torch.float32)
        # Topologically Sorted Source Nodes: [x_824, x_comb_iter_1_right_14, x_851, x_comb_iter_3_right_14], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_5.run(buf7, buf8, buf9, 2976048, grid=grid(2976048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_865, input_27, input_28], Original ATen: [aten.relu, aten.avg_pool2d, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg93_1
        del buf10
        # Topologically Sorted Source Nodes: [x_865, input_29, input_30, input_31], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg94_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg94_1
        del buf12
        buf14 = empty_strided_cuda((8, 108, 83, 83), (746496, 6912, 83, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_19, out_4], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_6.run(buf11, buf13, arg95_1, arg96_1, arg97_1, arg98_1, buf14, 5952096, grid=grid(5952096), stream=stream0)
        del arg95_1
        del arg96_1
        del arg97_1
        del arg98_1
        del buf11
        del buf13
        buf15 = empty_strided_cuda((8, 108, 42, 42), (193536, 1792, 42, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_878, x_comb_iter_0_right_13], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_7.run(buf14, buf15, 1524096, grid=grid(1524096), stream=stream0)
        buf16 = empty_strided_cuda((8, 96, 169, 169), (2741856, 1, 16224, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_805, x_806], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_8.run(buf3, buf16, 21934848, grid=grid(21934848), stream=stream0)
        # Topologically Sorted Source Nodes: [x_805, x_806, x_807], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg11_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf17, (8, 96, 83, 83), (661344, 1, 7968, 96))
        del arg11_1
        del buf16
        # Topologically Sorted Source Nodes: [x_808], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg12_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_809, x_810], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf19, arg13_1, arg14_1, arg15_1, arg16_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        del arg16_1
        # Topologically Sorted Source Nodes: [x_809, x_810, x_811], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg17_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf20, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg17_1
        del buf19
        # Topologically Sorted Source Nodes: [x_812], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg18_1
        del buf20
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf4, arg23_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg23_1
        del buf4
        buf23 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_813, input_26, x_comb_iter_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf23, arg19_1, arg20_1, arg21_1, arg22_1, buf22, arg24_1, arg25_1, arg26_1, arg27_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        del arg22_1
        del arg24_1
        del arg25_1
        del arg26_1
        del arg27_1
        del buf22
        buf24 = empty_strided_cuda((8, 54, 171, 171), (1579014, 1, 9234, 54), torch.float32)
        # Topologically Sorted Source Nodes: [x_815, x_816], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_11.run(buf7, buf24, 12632112, grid=grid(12632112), stream=stream0)
        # Topologically Sorted Source Nodes: [x_815, x_816, x_817], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg28_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf25, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg28_1
        del buf24
        # Topologically Sorted Source Nodes: [x_818], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg29_1
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_819, x_820], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf27, arg30_1, arg31_1, arg32_1, arg33_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg30_1
        del arg31_1
        del arg32_1
        del arg33_1
        # Topologically Sorted Source Nodes: [x_819, x_820, x_821], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg34_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf28, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg34_1
        del buf27
        # Topologically Sorted Source Nodes: [x_822], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg35_1
        del buf28
        buf30 = empty_strided_cuda((8, 54, 169, 169), (1542294, 1, 9126, 54), torch.float32)
        # Topologically Sorted Source Nodes: [x_825, x_826], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_12.run(buf7, buf30, 12338352, grid=grid(12338352), stream=stream0)
        # Topologically Sorted Source Nodes: [x_825, x_826, x_827], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg40_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf31, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg40_1
        del buf30
        # Topologically Sorted Source Nodes: [x_828], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg41_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_829, x_830], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf33, arg42_1, arg43_1, arg44_1, arg45_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        # Topologically Sorted Source Nodes: [x_829, x_830, x_831], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg46_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf34, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg46_1
        del buf33
        # Topologically Sorted Source Nodes: [x_832], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg47_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg47_1
        del buf34
        buf36 = empty_strided_cuda((8, 54, 167, 167), (1506006, 1, 9018, 54), torch.float32)
        # Topologically Sorted Source Nodes: [x_834, x_835], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_13.run(buf7, buf36, 12048048, grid=grid(12048048), stream=stream0)
        del buf7
        # Topologically Sorted Source Nodes: [x_834, x_835, x_836], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg52_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf37, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg52_1
        del buf36
        # Topologically Sorted Source Nodes: [x_837], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg53_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg53_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [x_838, x_839], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf39, arg54_1, arg55_1, arg56_1, arg57_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg54_1
        del arg55_1
        del arg56_1
        del arg57_1
        # Topologically Sorted Source Nodes: [x_838, x_839, x_840], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg58_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf40, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg58_1
        del buf39
        # Topologically Sorted Source Nodes: [x_841], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg59_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg59_1
        buf42 = buf35; del buf35  # reuse
        buf43 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_833, x_842, x_comb_iter_72, x_843], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf42, arg48_1, arg49_1, arg50_1, arg51_1, buf41, arg60_1, arg61_1, arg62_1, arg63_1, buf43, 2976048, grid=grid(2976048), stream=stream0)
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        del arg60_1
        del arg61_1
        del arg62_1
        del arg63_1
        del buf41
        # Topologically Sorted Source Nodes: [x_843, x_844], Original ATen: [aten.relu, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg64_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf44, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg64_1
        del buf43
        # Topologically Sorted Source Nodes: [x_845], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg65_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg65_1
        del buf44
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [x_846, x_847], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf46, arg66_1, arg67_1, arg68_1, arg69_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg66_1
        del arg67_1
        del arg68_1
        del arg69_1
        # Topologically Sorted Source Nodes: [x_846, x_847, x_848], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg70_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf47, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg70_1
        del buf46
        # Topologically Sorted Source Nodes: [x_849], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg71_1
        del buf47
        buf49 = empty_strided_cuda((8, 96, 167, 167), (2677344, 1, 16032, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_852, x_853], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_15.run(buf3, buf49, 21418752, grid=grid(21418752), stream=stream0)
        del buf3
        # Topologically Sorted Source Nodes: [x_852, x_853, x_854], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg76_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf50, (8, 96, 83, 83), (661344, 1, 7968, 96))
        del arg76_1
        del buf49
        # Topologically Sorted Source Nodes: [x_855], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg77_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_856, x_857], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf52, arg78_1, arg79_1, arg80_1, arg81_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        # Topologically Sorted Source Nodes: [x_856, x_857, x_858], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf53 = extern_kernels.convolution(buf52, arg82_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=54, bias=None)
        assert_size_stride(buf53, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg82_1
        del buf52
        # Topologically Sorted Source Nodes: [x_859], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg83_1
        del buf53
        # Topologically Sorted Source Nodes: [x_861, x_863], Original ATen: [aten.relu, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg88_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 54, 83, 83), (372006, 1, 4482, 54))
        del arg88_1
        del buf55
        buf57 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_860, x_864, x_comb_iter_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf57, arg84_1, arg85_1, arg86_1, arg87_1, buf56, arg89_1, arg90_1, arg91_1, arg92_1, 2976048, grid=grid(2976048), stream=stream0)
        del arg84_1
        del arg85_1
        del arg86_1
        del arg87_1
        del arg89_1
        del arg90_1
        del arg91_1
        del arg92_1
        del buf56
        buf58 = empty_strided_cuda((8, 270, 83, 83), (1866240, 6912, 83, 1), torch.float32)
        buf59 = empty_strided_cuda((8, 270, 83, 83), (1860030, 1, 22410, 270), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_14, x_866], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_16.run(buf23, buf29, arg36_1, arg37_1, arg38_1, arg39_1, buf8, buf42, buf48, arg72_1, arg73_1, arg74_1, arg75_1, buf9, buf57, buf58, buf59, 2160, 6889, grid=grid(2160, 6889), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        del arg39_1
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf23
        del buf29
        del buf42
        del buf48
        del buf57
        del buf8
        del buf9
        # Topologically Sorted Source Nodes: [x_866, x_867], Original ATen: [aten.relu, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 108, 83, 83), (744012, 1, 8964, 108))
        del arg99_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        buf107 = empty_strided_cuda((8, 108, 83, 83), (744012, 1, 8964, 108), torch.float32)
        # Topologically Sorted Source Nodes: [x_868, x_925], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf61, arg100_1, arg101_1, arg102_1, arg103_1, buf107, 5952096, grid=grid(5952096), stream=stream0)
        del arg100_1
        del arg101_1
        del arg102_1
        del arg103_1
        buf62 = empty_strided_cuda((8, 108, 42, 42), (190512, 1, 4536, 108), torch.float32)
        buf63 = empty_strided_cuda((8, 108, 42, 42), (190512, 1, 4536, 108), torch.float32)
        # Topologically Sorted Source Nodes: [x_888, x_comb_iter_1_right_15, x_915, x_comb_iter_3_right_15], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_18.run(buf61, buf62, buf63, 1524096, grid=grid(1524096), stream=stream0)
        buf64 = empty_strided_cuda((8, 270, 42, 42), (476280, 1, 11340, 270), torch.float32)
        # Topologically Sorted Source Nodes: [x_929, input_32], Original ATen: [aten.relu, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_relu_19.run(buf58, buf64, 2160, 1764, grid=grid(2160, 1764), stream=stream0)
        # Topologically Sorted Source Nodes: [x_929, input_32, input_33], Original ATen: [aten.relu, aten.avg_pool2d, aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg181_1
        buf66 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [x_929, input_34, input_35], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_constant_pad_nd_relu_20.run(buf58, buf66, 2160, 1764, grid=grid(2160, 1764), stream=stream0)
        del buf58
        # Topologically Sorted Source Nodes: [x_929, input_34, input_35, input_36], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg182_1
        buf68 = empty_strided_cuda((8, 216, 42, 42), (387072, 1792, 42, 1), torch.float32)
        buf120 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        buf151 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        # Topologically Sorted Source Nodes: [cat_21, out_5, x_933, x_973], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_21.run(buf65, buf67, arg183_1, arg184_1, arg185_1, arg186_1, buf68, buf120, buf151, 1728, 1764, grid=grid(1728, 1764), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        del arg186_1
        del buf65
        del buf67
        buf69 = empty_strided_cuda((8, 216, 42, 42), (387072, 1792, 42, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_14], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_22.run(buf68, buf69, 3048192, grid=grid(3048192), stream=stream0)
        del buf68
        buf70 = empty_strided_cuda((8, 108, 87, 87), (817452, 1, 9396, 108), torch.float32)
        # Topologically Sorted Source Nodes: [x_869, x_870], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_23.run(buf14, buf70, 864, 7569, grid=grid(864, 7569), stream=stream0)
        # Topologically Sorted Source Nodes: [x_869, x_870, x_871], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg104_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf71, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg104_1
        # Topologically Sorted Source Nodes: [x_872], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg105_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_873, x_874], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf73, arg106_1, arg107_1, arg108_1, arg109_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg106_1
        del arg107_1
        del arg108_1
        del arg109_1
        # Topologically Sorted Source Nodes: [x_873, x_874, x_875], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg110_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf74, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg110_1
        del buf73
        # Topologically Sorted Source Nodes: [x_876], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg111_1
        del buf74
        buf76 = empty_strided_cuda((8, 108, 89, 89), (855468, 1, 9612, 108), torch.float32)
        # Topologically Sorted Source Nodes: [x_879, x_880], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_25.run(buf61, buf76, 6843744, grid=grid(6843744), stream=stream0)
        # Topologically Sorted Source Nodes: [x_879, x_880, x_881], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg116_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf77, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg116_1
        del buf76
        # Topologically Sorted Source Nodes: [x_882], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg117_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_883, x_884], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf79, arg118_1, arg119_1, arg120_1, arg121_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg118_1
        del arg119_1
        del arg120_1
        del arg121_1
        # Topologically Sorted Source Nodes: [x_883, x_884, x_885], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg122_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf80, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg122_1
        del buf79
        # Topologically Sorted Source Nodes: [x_886], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg123_1
        del buf80
        buf82 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_889, x_890], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_26.run(buf61, buf82, 6539616, grid=grid(6539616), stream=stream0)
        # Topologically Sorted Source Nodes: [x_889, x_890, x_891], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg128_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf83, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg128_1
        del buf82
        # Topologically Sorted Source Nodes: [x_892], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg129_1
        del buf83
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_893, x_894], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf85, arg130_1, arg131_1, arg132_1, arg133_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        # Topologically Sorted Source Nodes: [x_893, x_894, x_895], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg134_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf86, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg134_1
        del buf85
        # Topologically Sorted Source Nodes: [x_896], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg135_1
        del buf86
        buf88 = empty_strided_cuda((8, 108, 85, 85), (780300, 1, 9180, 108), torch.float32)
        # Topologically Sorted Source Nodes: [x_898, x_899], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_27.run(buf61, buf88, 6242400, grid=grid(6242400), stream=stream0)
        del buf61
        # Topologically Sorted Source Nodes: [x_898, x_899, x_900], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg140_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf89, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg140_1
        # Topologically Sorted Source Nodes: [x_901], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg141_1
        del buf89
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_902, x_903], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf91, arg142_1, arg143_1, arg144_1, arg145_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [x_902, x_903, x_904], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg146_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf92, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg146_1
        del buf91
        # Topologically Sorted Source Nodes: [x_905], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg147_1
        buf94 = buf87; del buf87  # reuse
        buf95 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [x_897, x_906, x_comb_iter_77, x_907], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_28.run(buf94, arg136_1, arg137_1, arg138_1, arg139_1, buf93, arg148_1, arg149_1, arg150_1, arg151_1, buf95, 1524096, grid=grid(1524096), stream=stream0)
        del arg136_1
        del arg137_1
        del arg138_1
        del arg139_1
        del arg148_1
        del arg149_1
        del arg150_1
        del arg151_1
        del buf93
        # Topologically Sorted Source Nodes: [x_907, x_908], Original ATen: [aten.relu, aten.convolution]
        buf96 = extern_kernels.convolution(buf95, arg152_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf96, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg152_1
        del buf95
        # Topologically Sorted Source Nodes: [x_909], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg153_1
        del buf96
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_910, x_911], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf98, arg154_1, arg155_1, arg156_1, arg157_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg154_1
        del arg155_1
        del arg156_1
        del arg157_1
        # Topologically Sorted Source Nodes: [x_910, x_911, x_912], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg158_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf99, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg158_1
        del buf98
        # Topologically Sorted Source Nodes: [x_913], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg159_1
        del buf99
        buf101 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_916, x_917], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_29.run(buf14, buf101, 864, 7225, grid=grid(864, 7225), stream=stream0)
        del buf14
        # Topologically Sorted Source Nodes: [x_916, x_917, x_918], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg164_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf102, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg164_1
        del buf101
        # Topologically Sorted Source Nodes: [x_919], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg165_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [x_920, x_921], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf104, arg166_1, arg167_1, arg168_1, arg169_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg166_1
        del arg167_1
        del arg168_1
        del arg169_1
        # Topologically Sorted Source Nodes: [x_920, x_921, x_922], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf105 = extern_kernels.convolution(buf104, arg170_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=108, bias=None)
        assert_size_stride(buf105, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg170_1
        del buf104
        # Topologically Sorted Source Nodes: [x_923], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg171_1
        del buf105
        # Topologically Sorted Source Nodes: [x_925, x_927], Original ATen: [aten.relu, aten.convolution]
        buf108 = extern_kernels.convolution(buf107, arg176_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 108, 42, 42), (190512, 1, 4536, 108))
        del arg176_1
        del buf107
        buf109 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_924, x_928, x_comb_iter_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_30.run(buf109, arg172_1, arg173_1, arg174_1, arg175_1, buf108, arg177_1, arg178_1, arg179_1, arg180_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf108
        buf111 = empty_strided_cuda((8, 540, 42, 42), (952560, 1, 22680, 540), torch.float32)
        buf116 = empty_strided_cuda((8, 540, 42, 42), (952560, 1, 22680, 540), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_15, x_930, x_981], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_31.run(buf75, arg112_1, arg113_1, arg114_1, arg115_1, buf15, buf81, arg124_1, arg125_1, arg126_1, arg127_1, buf62, buf94, buf100, arg160_1, arg161_1, arg162_1, arg163_1, buf63, buf109, buf111, buf116, 4320, 1764, grid=grid(4320, 1764), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        del arg124_1
        del arg125_1
        del arg126_1
        del arg127_1
        del arg160_1
        del arg161_1
        del arg162_1
        del arg163_1
        del buf100
        del buf109
        del buf15
        # Topologically Sorted Source Nodes: [x_930, x_931], Original ATen: [aten.relu, aten.convolution]
        buf112 = extern_kernels.convolution(buf111, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg187_1
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_932], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf113, arg188_1, arg189_1, arg190_1, arg191_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del arg191_1
        buf114 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        buf115 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        buf126 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        buf132 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        buf138 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_16, x_comb_iter_3_right_16, x_941, x_949, x_957], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_33.run(buf113, buf114, buf115, buf126, buf132, buf138, 3048192, grid=grid(3048192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_981, x_982], Original ATen: [aten.relu, aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg264_1
        buf118 = buf117; del buf117  # reuse
        buf167 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        buf198 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        # Topologically Sorted Source Nodes: [x_983, x_987, x_1027], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf118, arg265_1, arg266_1, arg267_1, arg268_1, buf167, buf198, 3048192, grid=grid(3048192), stream=stream0)
        del arg265_1
        del arg266_1
        del arg267_1
        del arg268_1
        buf119 = empty_strided_cuda((8, 216, 42, 42), (381024, 1, 9072, 216), torch.float32)
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_15], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_35.run(buf118, buf119, 3048192, grid=grid(3048192), stream=stream0)
        del buf118
        # Topologically Sorted Source Nodes: [x_933, x_934], Original ATen: [aten.relu, aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg192_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf121, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg192_1
        del buf120
        # Topologically Sorted Source Nodes: [x_935], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg193_1
        del buf121
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_936, x_937], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf123, arg194_1, arg195_1, arg196_1, arg197_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg194_1
        del arg195_1
        del arg196_1
        del arg197_1
        # Topologically Sorted Source Nodes: [x_936, x_937, x_938], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg198_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf124, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg198_1
        del buf123
        # Topologically Sorted Source Nodes: [x_939], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg199_1
        del buf124
        # Topologically Sorted Source Nodes: [x_941, x_942], Original ATen: [aten.relu, aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg204_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf127, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg204_1
        del buf126
        # Topologically Sorted Source Nodes: [x_943], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg205_1
        del buf127
        buf129 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_944, x_945], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf129, arg206_1, arg207_1, arg208_1, arg209_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        del arg209_1
        # Topologically Sorted Source Nodes: [x_944, x_945, x_946], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg210_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf130, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg210_1
        del buf129
        # Topologically Sorted Source Nodes: [x_947], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg211_1
        del buf130
        # Topologically Sorted Source Nodes: [x_949, x_950], Original ATen: [aten.relu, aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg216_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf133, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg216_1
        del buf132
        # Topologically Sorted Source Nodes: [x_951], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg217_1
        del buf133
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_952, x_953], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf135, arg218_1, arg219_1, arg220_1, arg221_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg218_1
        del arg219_1
        del arg220_1
        del arg221_1
        # Topologically Sorted Source Nodes: [x_952, x_953, x_954], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg222_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf136, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg222_1
        del buf135
        # Topologically Sorted Source Nodes: [x_955], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg223_1
        del buf136
        # Topologically Sorted Source Nodes: [x_957, x_958], Original ATen: [aten.relu, aten.convolution]
        buf139 = extern_kernels.convolution(buf138, arg228_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf139, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg228_1
        del buf138
        # Topologically Sorted Source Nodes: [x_959], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg229_1
        del buf139
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_960, x_961], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf141, arg230_1, arg231_1, arg232_1, arg233_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg230_1
        del arg231_1
        del arg232_1
        del arg233_1
        # Topologically Sorted Source Nodes: [x_960, x_961, x_962], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg234_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf142, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg234_1
        del buf141
        # Topologically Sorted Source Nodes: [x_963], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg235_1
        buf144 = buf137; del buf137  # reuse
        buf145 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_956, x_964, x_comb_iter_82, x_965], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf144, arg224_1, arg225_1, arg226_1, arg227_1, buf143, arg236_1, arg237_1, arg238_1, arg239_1, buf145, 3048192, grid=grid(3048192), stream=stream0)
        del arg224_1
        del arg225_1
        del arg226_1
        del arg227_1
        del arg236_1
        del arg237_1
        del arg238_1
        del arg239_1
        del buf143
        # Topologically Sorted Source Nodes: [x_965, x_966], Original ATen: [aten.relu, aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg240_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf146, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg240_1
        del buf145
        # Topologically Sorted Source Nodes: [x_967], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg241_1
        del buf146
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_968, x_969], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf148, arg242_1, arg243_1, arg244_1, arg245_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        # Topologically Sorted Source Nodes: [x_968, x_969, x_970], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg246_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf149, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg246_1
        del buf148
        # Topologically Sorted Source Nodes: [x_971], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg247_1
        del buf149
        # Topologically Sorted Source Nodes: [x_973, x_974], Original ATen: [aten.relu, aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg252_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf152, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg252_1
        del buf151
        # Topologically Sorted Source Nodes: [x_975], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, arg253_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg253_1
        del buf152
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_976, x_977], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf154, arg254_1, arg255_1, arg256_1, arg257_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg254_1
        del arg255_1
        del arg256_1
        del arg257_1
        # Topologically Sorted Source Nodes: [x_976, x_977, x_978], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf155 = extern_kernels.convolution(buf154, arg258_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf155, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg258_1
        del buf154
        # Topologically Sorted Source Nodes: [x_979], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg259_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg259_1
        del buf155
        buf158 = empty_strided_cuda((8, 1080, 42, 42), (1905120, 1, 45360, 1080), torch.float32)
        buf163 = empty_strided_cuda((8, 1080, 42, 42), (1905120, 1, 45360, 1080), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_16, x_984, x_1035], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_38.run(buf125, arg200_1, arg201_1, arg202_1, arg203_1, buf69, buf131, arg212_1, arg213_1, arg214_1, arg215_1, buf114, buf144, buf150, arg248_1, arg249_1, arg250_1, arg251_1, buf115, buf156, arg260_1, arg261_1, arg262_1, arg263_1, buf113, buf158, buf163, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del arg248_1
        del arg249_1
        del arg250_1
        del arg251_1
        del arg260_1
        del arg261_1
        del arg262_1
        del arg263_1
        del buf69
        # Topologically Sorted Source Nodes: [x_984, x_985], Original ATen: [aten.relu, aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg269_1
        buf160 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_986], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf160, arg270_1, arg271_1, arg272_1, arg273_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg270_1
        del arg271_1
        del arg272_1
        del arg273_1
        buf161 = buf156; del buf156  # reuse
        buf162 = buf150; del buf150  # reuse
        buf173 = buf144; del buf144  # reuse
        buf179 = buf131; del buf131  # reuse
        buf185 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_17, x_comb_iter_3_right_17, x_995, x_1003, x_1011], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_33.run(buf160, buf161, buf162, buf173, buf179, buf185, 3048192, grid=grid(3048192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1035, x_1036], Original ATen: [aten.relu, aten.convolution]
        buf164 = extern_kernels.convolution(buf163, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg346_1
        buf165 = buf164; del buf164  # reuse
        buf214 = buf115; del buf115  # reuse
        buf245 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [x_1037, x_1041, x_1081], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf165, arg347_1, arg348_1, arg349_1, arg350_1, buf214, buf245, 3048192, grid=grid(3048192), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        buf166 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_16], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_35.run(buf165, buf166, 3048192, grid=grid(3048192), stream=stream0)
        del buf165
        # Topologically Sorted Source Nodes: [x_987, x_988], Original ATen: [aten.relu, aten.convolution]
        buf168 = extern_kernels.convolution(buf167, arg274_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf168, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg274_1
        del buf167
        # Topologically Sorted Source Nodes: [x_989], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, arg275_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg275_1
        del buf168
        buf170 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_990, x_991], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf170, arg276_1, arg277_1, arg278_1, arg279_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg276_1
        del arg277_1
        del arg278_1
        del arg279_1
        # Topologically Sorted Source Nodes: [x_990, x_991, x_992], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg280_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf171, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg280_1
        del buf170
        # Topologically Sorted Source Nodes: [x_993], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg281_1
        del buf171
        # Topologically Sorted Source Nodes: [x_995, x_996], Original ATen: [aten.relu, aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg286_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf174, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg286_1
        del buf173
        # Topologically Sorted Source Nodes: [x_997], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg287_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg287_1
        del buf174
        buf176 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_998, x_999], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf176, arg288_1, arg289_1, arg290_1, arg291_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg288_1
        del arg289_1
        del arg290_1
        del arg291_1
        # Topologically Sorted Source Nodes: [x_998, x_999, x_1000], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf177 = extern_kernels.convolution(buf176, arg292_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf177, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg292_1
        del buf176
        # Topologically Sorted Source Nodes: [x_1001], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, arg293_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg293_1
        del buf177
        # Topologically Sorted Source Nodes: [x_1003, x_1004], Original ATen: [aten.relu, aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg298_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf180, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg298_1
        del buf179
        # Topologically Sorted Source Nodes: [x_1005], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, arg299_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg299_1
        del buf180
        buf182 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_1006, x_1007], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf182, arg300_1, arg301_1, arg302_1, arg303_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg300_1
        del arg301_1
        del arg302_1
        del arg303_1
        # Topologically Sorted Source Nodes: [x_1006, x_1007, x_1008], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg304_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf183, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg304_1
        del buf182
        # Topologically Sorted Source Nodes: [x_1009], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg305_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg305_1
        del buf183
        # Topologically Sorted Source Nodes: [x_1011, x_1012], Original ATen: [aten.relu, aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg310_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf186, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg310_1
        del buf185
        # Topologically Sorted Source Nodes: [x_1013], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, arg311_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg311_1
        del buf186
        buf188 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [x_1014, x_1015], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf188, arg312_1, arg313_1, arg314_1, arg315_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        # Topologically Sorted Source Nodes: [x_1014, x_1015, x_1016], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf189 = extern_kernels.convolution(buf188, arg316_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf189, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg316_1
        del buf188
        # Topologically Sorted Source Nodes: [x_1017], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, arg317_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg317_1
        buf191 = buf184; del buf184  # reuse
        buf192 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_1010, x_1018, x_comb_iter_87, x_1019], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf191, arg306_1, arg307_1, arg308_1, arg309_1, buf190, arg318_1, arg319_1, arg320_1, arg321_1, buf192, 3048192, grid=grid(3048192), stream=stream0)
        del arg306_1
        del arg307_1
        del arg308_1
        del arg309_1
        del arg318_1
        del arg319_1
        del arg320_1
        del arg321_1
        del buf190
        # Topologically Sorted Source Nodes: [x_1019, x_1020], Original ATen: [aten.relu, aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg322_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf193, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg322_1
        del buf192
        # Topologically Sorted Source Nodes: [x_1021], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, arg323_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg323_1
        del buf193
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_1022, x_1023], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf195, arg324_1, arg325_1, arg326_1, arg327_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg324_1
        del arg325_1
        del arg326_1
        del arg327_1
        # Topologically Sorted Source Nodes: [x_1022, x_1023, x_1024], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf196 = extern_kernels.convolution(buf195, arg328_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf196, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg328_1
        del buf195
        # Topologically Sorted Source Nodes: [x_1025], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg329_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg329_1
        del buf196
        # Topologically Sorted Source Nodes: [x_1027, x_1028], Original ATen: [aten.relu, aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg334_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf199, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg334_1
        del buf198
        # Topologically Sorted Source Nodes: [x_1029], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg335_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg335_1
        del buf199
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_1030, x_1031], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf201, arg336_1, arg337_1, arg338_1, arg339_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg336_1
        del arg337_1
        del arg338_1
        del arg339_1
        # Topologically Sorted Source Nodes: [x_1030, x_1031, x_1032], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf202 = extern_kernels.convolution(buf201, arg340_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf202, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg340_1
        del buf201
        # Topologically Sorted Source Nodes: [x_1033], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, arg341_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg341_1
        del buf202
        buf205 = buf163; del buf163  # reuse
        buf210 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_out_17, x_1038, x_1089], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_39.run(buf172, arg282_1, arg283_1, arg284_1, arg285_1, buf119, buf178, arg294_1, arg295_1, arg296_1, arg297_1, buf161, buf191, buf197, arg330_1, arg331_1, arg332_1, arg333_1, buf162, buf203, arg342_1, arg343_1, arg344_1, arg345_1, buf160, buf205, buf210, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del arg294_1
        del arg295_1
        del arg296_1
        del arg297_1
        del arg330_1
        del arg331_1
        del arg332_1
        del arg333_1
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del buf119
        # Topologically Sorted Source Nodes: [x_1038, x_1039], Original ATen: [aten.relu, aten.convolution]
        buf206 = extern_kernels.convolution(buf205, arg351_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg351_1
        buf207 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_1040], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf207, arg352_1, arg353_1, arg354_1, arg355_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        buf208 = buf203; del buf203  # reuse
        buf209 = buf197; del buf197  # reuse
        buf220 = buf191; del buf191  # reuse
        buf226 = buf178; del buf178  # reuse
        buf232 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_18, x_comb_iter_3_right_18, x_1049, x_1057, x_1065], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_33.run(buf207, buf208, buf209, buf220, buf226, buf232, 3048192, grid=grid(3048192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1089, x_1090], Original ATen: [aten.relu, aten.convolution]
        buf211 = extern_kernels.convolution(buf210, arg428_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg428_1
        buf212 = buf211; del buf211  # reuse
        buf261 = buf162; del buf162  # reuse
        buf292 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_1091, x_1095, x_1135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf212, arg429_1, arg430_1, arg431_1, arg432_1, buf261, buf292, 3048192, grid=grid(3048192), stream=stream0)
        del arg429_1
        del arg430_1
        del arg431_1
        del arg432_1
        buf213 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_17], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_35.run(buf212, buf213, 3048192, grid=grid(3048192), stream=stream0)
        del buf212
        # Topologically Sorted Source Nodes: [x_1041, x_1042], Original ATen: [aten.relu, aten.convolution]
        buf215 = extern_kernels.convolution(buf214, arg356_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf215, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg356_1
        del buf214
        # Topologically Sorted Source Nodes: [x_1043], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, arg357_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg357_1
        del buf215
        buf217 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [x_1044, x_1045], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf217, arg358_1, arg359_1, arg360_1, arg361_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg358_1
        del arg359_1
        del arg360_1
        del arg361_1
        # Topologically Sorted Source Nodes: [x_1044, x_1045, x_1046], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf218 = extern_kernels.convolution(buf217, arg362_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf218, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg362_1
        del buf217
        # Topologically Sorted Source Nodes: [x_1047], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg363_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg363_1
        del buf218
        # Topologically Sorted Source Nodes: [x_1049, x_1050], Original ATen: [aten.relu, aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg368_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf221, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg368_1
        del buf220
        # Topologically Sorted Source Nodes: [x_1051], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, arg369_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg369_1
        del buf221
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_1052, x_1053], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf223, arg370_1, arg371_1, arg372_1, arg373_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg370_1
        del arg371_1
        del arg372_1
        del arg373_1
        # Topologically Sorted Source Nodes: [x_1052, x_1053, x_1054], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf224 = extern_kernels.convolution(buf223, arg374_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf224, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg374_1
        del buf223
        # Topologically Sorted Source Nodes: [x_1055], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg375_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg375_1
        del buf224
        # Topologically Sorted Source Nodes: [x_1057, x_1058], Original ATen: [aten.relu, aten.convolution]
        buf227 = extern_kernels.convolution(buf226, arg380_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf227, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg380_1
        del buf226
        # Topologically Sorted Source Nodes: [x_1059], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, arg381_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg381_1
        del buf227
        buf229 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [x_1060, x_1061], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf229, arg382_1, arg383_1, arg384_1, arg385_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        # Topologically Sorted Source Nodes: [x_1060, x_1061, x_1062], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg386_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf230, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg386_1
        del buf229
        # Topologically Sorted Source Nodes: [x_1063], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, arg387_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg387_1
        del buf230
        # Topologically Sorted Source Nodes: [x_1065, x_1066], Original ATen: [aten.relu, aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg392_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf233, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg392_1
        del buf232
        # Topologically Sorted Source Nodes: [x_1067], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, arg393_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg393_1
        del buf233
        buf235 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [x_1068, x_1069], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf235, arg394_1, arg395_1, arg396_1, arg397_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg394_1
        del arg395_1
        del arg396_1
        del arg397_1
        # Topologically Sorted Source Nodes: [x_1068, x_1069, x_1070], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf236 = extern_kernels.convolution(buf235, arg398_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf236, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg398_1
        del buf235
        # Topologically Sorted Source Nodes: [x_1071], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg399_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg399_1
        buf238 = buf231; del buf231  # reuse
        buf239 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [x_1064, x_1072, x_comb_iter_92, x_1073], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf238, arg388_1, arg389_1, arg390_1, arg391_1, buf237, arg400_1, arg401_1, arg402_1, arg403_1, buf239, 3048192, grid=grid(3048192), stream=stream0)
        del arg388_1
        del arg389_1
        del arg390_1
        del arg391_1
        del arg400_1
        del arg401_1
        del arg402_1
        del arg403_1
        del buf237
        # Topologically Sorted Source Nodes: [x_1073, x_1074], Original ATen: [aten.relu, aten.convolution]
        buf240 = extern_kernels.convolution(buf239, arg404_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf240, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg404_1
        del buf239
        # Topologically Sorted Source Nodes: [x_1075], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, arg405_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg405_1
        del buf240
        buf242 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_1076, x_1077], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf242, arg406_1, arg407_1, arg408_1, arg409_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg406_1
        del arg407_1
        del arg408_1
        del arg409_1
        # Topologically Sorted Source Nodes: [x_1076, x_1077, x_1078], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf243 = extern_kernels.convolution(buf242, arg410_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf243, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg410_1
        del buf242
        # Topologically Sorted Source Nodes: [x_1079], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, arg411_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg411_1
        del buf243
        # Topologically Sorted Source Nodes: [x_1081, x_1082], Original ATen: [aten.relu, aten.convolution]
        buf246 = extern_kernels.convolution(buf245, arg416_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf246, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg416_1
        del buf245
        # Topologically Sorted Source Nodes: [x_1083], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, arg417_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg417_1
        del buf246
        buf248 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [x_1084, x_1085], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf248, arg418_1, arg419_1, arg420_1, arg421_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg418_1
        del arg419_1
        del arg420_1
        del arg421_1
        # Topologically Sorted Source Nodes: [x_1084, x_1085, x_1086], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf249 = extern_kernels.convolution(buf248, arg422_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf249, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg422_1
        del buf248
        # Topologically Sorted Source Nodes: [x_1087], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, arg423_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg423_1
        del buf249
        buf252 = buf210; del buf210  # reuse
        buf257 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_out_18, x_1092, x_1143], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_39.run(buf219, arg364_1, arg365_1, arg366_1, arg367_1, buf166, buf225, arg376_1, arg377_1, arg378_1, arg379_1, buf208, buf238, buf244, arg412_1, arg413_1, arg414_1, arg415_1, buf209, buf250, arg424_1, arg425_1, arg426_1, arg427_1, buf207, buf252, buf257, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del arg364_1
        del arg365_1
        del arg366_1
        del arg367_1
        del arg376_1
        del arg377_1
        del arg378_1
        del arg379_1
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        del arg424_1
        del arg425_1
        del arg426_1
        del arg427_1
        del buf166
        del buf207
        del buf208
        del buf209
        # Topologically Sorted Source Nodes: [x_1092, x_1093], Original ATen: [aten.relu, aten.convolution]
        buf253 = extern_kernels.convolution(buf252, arg433_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg433_1
        del buf252
        buf254 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_1094], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf254, arg434_1, arg435_1, arg436_1, arg437_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg434_1
        del arg435_1
        del arg436_1
        del arg437_1
        buf255 = buf250; del buf250  # reuse
        buf256 = buf244; del buf244  # reuse
        buf267 = buf238; del buf238  # reuse
        buf273 = buf225; del buf225  # reuse
        buf279 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_19, x_comb_iter_3_right_19, x_1103, x_1111, x_1119], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_33.run(buf254, buf255, buf256, buf267, buf273, buf279, 3048192, grid=grid(3048192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1143, x_1144], Original ATen: [aten.relu, aten.convolution]
        buf258 = extern_kernels.convolution(buf257, arg510_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 432, 42, 42), (762048, 1, 18144, 432))
        del arg510_1
        buf259 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_1145], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_40.run(buf259, arg511_1, arg512_1, arg513_1, arg514_1, 6096384, grid=grid(6096384), stream=stream0)
        del arg511_1
        del arg512_1
        del arg513_1
        del arg514_1
        buf260 = reinterpret_tensor(buf94, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_1158, x_comb_iter_0_right_18], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_41.run(buf259, buf260, 1524096, grid=grid(1524096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1095, x_1096], Original ATen: [aten.relu, aten.convolution]
        buf262 = extern_kernels.convolution(buf261, arg438_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf262, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg438_1
        del buf261
        # Topologically Sorted Source Nodes: [x_1097], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, arg439_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg439_1
        del buf262
        buf264 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [x_1098, x_1099], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf264, arg440_1, arg441_1, arg442_1, arg443_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg440_1
        del arg441_1
        del arg442_1
        del arg443_1
        # Topologically Sorted Source Nodes: [x_1098, x_1099, x_1100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf265 = extern_kernels.convolution(buf264, arg444_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf265, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg444_1
        del buf264
        # Topologically Sorted Source Nodes: [x_1101], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, arg445_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg445_1
        del buf265
        # Topologically Sorted Source Nodes: [x_1103, x_1104], Original ATen: [aten.relu, aten.convolution]
        buf268 = extern_kernels.convolution(buf267, arg450_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf268, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg450_1
        del buf267
        # Topologically Sorted Source Nodes: [x_1105], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, arg451_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg451_1
        del buf268
        buf270 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_1106, x_1107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf270, arg452_1, arg453_1, arg454_1, arg455_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        # Topologically Sorted Source Nodes: [x_1106, x_1107, x_1108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf271 = extern_kernels.convolution(buf270, arg456_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf271, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg456_1
        del buf270
        # Topologically Sorted Source Nodes: [x_1109], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, arg457_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg457_1
        del buf271
        # Topologically Sorted Source Nodes: [x_1111, x_1112], Original ATen: [aten.relu, aten.convolution]
        buf274 = extern_kernels.convolution(buf273, arg462_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf274, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg462_1
        del buf273
        # Topologically Sorted Source Nodes: [x_1113], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, arg463_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg463_1
        del buf274
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [x_1114, x_1115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf276, arg464_1, arg465_1, arg466_1, arg467_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg464_1
        del arg465_1
        del arg466_1
        del arg467_1
        # Topologically Sorted Source Nodes: [x_1114, x_1115, x_1116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf277 = extern_kernels.convolution(buf276, arg468_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf277, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg468_1
        del buf276
        # Topologically Sorted Source Nodes: [x_1117], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, arg469_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg469_1
        del buf277
        # Topologically Sorted Source Nodes: [x_1119, x_1120], Original ATen: [aten.relu, aten.convolution]
        buf280 = extern_kernels.convolution(buf279, arg474_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf280, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg474_1
        del buf279
        # Topologically Sorted Source Nodes: [x_1121], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, arg475_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg475_1
        del buf280
        buf282 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [x_1122, x_1123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf282, arg476_1, arg477_1, arg478_1, arg479_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg476_1
        del arg477_1
        del arg478_1
        del arg479_1
        # Topologically Sorted Source Nodes: [x_1122, x_1123, x_1124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf283 = extern_kernels.convolution(buf282, arg480_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf283, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg480_1
        del buf282
        # Topologically Sorted Source Nodes: [x_1125], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, arg481_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg481_1
        buf285 = buf278; del buf278  # reuse
        buf286 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_1118, x_1126, x_comb_iter_97, x_1127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf285, arg470_1, arg471_1, arg472_1, arg473_1, buf284, arg482_1, arg483_1, arg484_1, arg485_1, buf286, 3048192, grid=grid(3048192), stream=stream0)
        del arg470_1
        del arg471_1
        del arg472_1
        del arg473_1
        del arg482_1
        del arg483_1
        del arg484_1
        del arg485_1
        del buf284
        # Topologically Sorted Source Nodes: [x_1127, x_1128], Original ATen: [aten.relu, aten.convolution]
        buf287 = extern_kernels.convolution(buf286, arg486_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf287, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg486_1
        del buf286
        # Topologically Sorted Source Nodes: [x_1129], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, arg487_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg487_1
        del buf287
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_1130, x_1131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf289, arg488_1, arg489_1, arg490_1, arg491_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg488_1
        del arg489_1
        del arg490_1
        del arg491_1
        # Topologically Sorted Source Nodes: [x_1130, x_1131, x_1132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf290 = extern_kernels.convolution(buf289, arg492_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf290, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg492_1
        del buf289
        # Topologically Sorted Source Nodes: [x_1133], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, arg493_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg493_1
        del buf290
        # Topologically Sorted Source Nodes: [x_1135, x_1136], Original ATen: [aten.relu, aten.convolution]
        buf293 = extern_kernels.convolution(buf292, arg498_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf293, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg498_1
        del buf292
        # Topologically Sorted Source Nodes: [x_1137], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, arg499_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg499_1
        del buf293
        buf295 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [x_1138, x_1139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf295, arg500_1, arg501_1, arg502_1, arg503_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg500_1
        del arg501_1
        del arg502_1
        del arg503_1
        # Topologically Sorted Source Nodes: [x_1138, x_1139, x_1140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf296 = extern_kernels.convolution(buf295, arg504_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=216, bias=None)
        assert_size_stride(buf296, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg504_1
        del buf295
        # Topologically Sorted Source Nodes: [x_1141], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, arg505_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (8, 216, 42, 42), (381024, 1, 9072, 216))
        del arg505_1
        del buf296
        buf298 = empty_strided_cuda((8, 1080, 42, 42), (1935360, 1792, 42, 1), torch.float32)
        buf299 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_out_19, x_1146], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_42.run(buf266, arg446_1, arg447_1, arg448_1, arg449_1, buf213, buf272, arg458_1, arg459_1, arg460_1, arg461_1, buf255, buf285, buf291, arg494_1, arg495_1, arg496_1, arg497_1, buf256, buf297, arg506_1, arg507_1, arg508_1, arg509_1, buf254, buf298, buf299, 8640, 1764, grid=grid(8640, 1764), stream=stream0)
        del arg446_1
        del arg447_1
        del arg448_1
        del arg449_1
        del arg458_1
        del arg459_1
        del arg460_1
        del arg461_1
        del arg494_1
        del arg495_1
        del arg496_1
        del arg497_1
        del arg506_1
        del arg507_1
        del arg508_1
        del arg509_1
        del buf213
        del buf254
        del buf255
        del buf256
        del buf266
        del buf272
        del buf285
        del buf291
        # Topologically Sorted Source Nodes: [x_1146, x_1147], Original ATen: [aten.relu, aten.convolution]
        buf300 = extern_kernels.convolution(buf299, arg515_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 432, 42, 42), (762048, 1, 18144, 432))
        del arg515_1
        del buf299
        buf301 = buf300; del buf300  # reuse
        buf347 = empty_strided_cuda((8, 432, 42, 42), (762048, 1, 18144, 432), torch.float32)
        # Topologically Sorted Source Nodes: [x_1148, x_1205], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf301, arg516_1, arg517_1, arg518_1, arg519_1, buf347, 6096384, grid=grid(6096384), stream=stream0)
        del arg516_1
        del arg517_1
        del arg518_1
        del arg519_1
        buf302 = reinterpret_tensor(buf81, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf81  # reuse
        buf303 = reinterpret_tensor(buf75, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_1168, x_comb_iter_1_right_20, x_1195, x_comb_iter_3_right_20], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_44.run(buf301, buf302, buf303, 1524096, grid=grid(1524096), stream=stream0)
        buf304 = reinterpret_tensor(buf66, (8, 1080, 21, 21), (476280, 1, 22680, 1080), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_1209, input_37], Original ATen: [aten.relu, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_relu_45.run(buf298, buf304, 8640, 441, grid=grid(8640, 441), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1209, input_37, input_38], Original ATen: [aten.relu, aten.avg_pool2d, aten.convolution]
        buf305 = extern_kernels.convolution(buf304, arg597_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (8, 216, 21, 21), (95256, 1, 4536, 216))
        del arg597_1
        buf306 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_1209, input_39, input_40], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_constant_pad_nd_relu_46.run(buf298, buf306, 8640, 441, grid=grid(8640, 441), stream=stream0)
        del buf298
        # Topologically Sorted Source Nodes: [x_1209, input_39, input_40, input_41], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d, aten.convolution]
        buf307 = extern_kernels.convolution(buf306, arg598_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (8, 216, 21, 21), (95256, 1, 4536, 216))
        del arg598_1
        del buf306
        buf308 = empty_strided_cuda((8, 432, 21, 21), (190528, 441, 21, 1), torch.float32)
        buf360 = reinterpret_tensor(buf63, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf63  # reuse
        buf391 = reinterpret_tensor(buf62, (8, 432, 21, 21), (190512, 1, 9072, 432), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [cat_27, out_6, x_1213, x_1253], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_47.run(buf305, buf307, arg599_1, arg600_1, arg601_1, arg602_1, buf308, buf360, buf391, 3456, 441, grid=grid(3456, 441), stream=stream0)
        del arg599_1
        del arg600_1
        del arg601_1
        del arg602_1
        del buf305
        del buf307
        buf309 = empty_strided_cuda((8, 432, 21, 21), (190528, 441, 21, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_19], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_48.run(buf308, buf309, 1524096, grid=grid(1524096), stream=stream0)
        del buf308
        buf310 = empty_strided_cuda((8, 432, 45, 45), (874800, 1, 19440, 432), torch.float32)
        # Topologically Sorted Source Nodes: [x_1149, x_1150], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_49.run(buf259, buf310, 6998400, grid=grid(6998400), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1149, x_1150, x_1151], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf311 = extern_kernels.convolution(buf310, arg520_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf311, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg520_1
        # Topologically Sorted Source Nodes: [x_1152], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, arg521_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg521_1
        del buf311
        buf313 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [x_1153, x_1154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf313, arg522_1, arg523_1, arg524_1, arg525_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg522_1
        del arg523_1
        del arg524_1
        del arg525_1
        # Topologically Sorted Source Nodes: [x_1153, x_1154, x_1155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf314 = extern_kernels.convolution(buf313, arg526_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf314, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg526_1
        del buf313
        # Topologically Sorted Source Nodes: [x_1156], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, arg527_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg527_1
        del buf314
        buf316 = empty_strided_cuda((8, 432, 47, 47), (954288, 1, 20304, 432), torch.float32)
        # Topologically Sorted Source Nodes: [x_1159, x_1160], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_51.run(buf301, buf316, 7634304, grid=grid(7634304), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1159, x_1160, x_1161], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf317 = extern_kernels.convolution(buf316, arg532_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf317, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg532_1
        del buf316
        # Topologically Sorted Source Nodes: [x_1162], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, arg533_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg533_1
        del buf317
        buf319 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [x_1163, x_1164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf319, arg534_1, arg535_1, arg536_1, arg537_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg534_1
        del arg535_1
        del arg536_1
        del arg537_1
        # Topologically Sorted Source Nodes: [x_1163, x_1164, x_1165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf320 = extern_kernels.convolution(buf319, arg538_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf320, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg538_1
        del buf319
        # Topologically Sorted Source Nodes: [x_1166], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, arg539_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg539_1
        del buf320
        buf322 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [x_1169, x_1170], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_49.run(buf301, buf322, 6998400, grid=grid(6998400), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1169, x_1170, x_1171], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf323 = extern_kernels.convolution(buf322, arg544_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf323, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg544_1
        del buf322
        # Topologically Sorted Source Nodes: [x_1172], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, arg545_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg545_1
        del buf323
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [x_1173, x_1174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf325, arg546_1, arg547_1, arg548_1, arg549_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg546_1
        del arg547_1
        del arg548_1
        del arg549_1
        # Topologically Sorted Source Nodes: [x_1173, x_1174, x_1175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf326 = extern_kernels.convolution(buf325, arg550_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf326, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg550_1
        del buf325
        # Topologically Sorted Source Nodes: [x_1176], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, arg551_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg551_1
        del buf326
        buf328 = empty_strided_cuda((8, 432, 43, 43), (798768, 1, 18576, 432), torch.float32)
        # Topologically Sorted Source Nodes: [x_1178, x_1179], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_52.run(buf301, buf328, 6390144, grid=grid(6390144), stream=stream0)
        del buf301
        # Topologically Sorted Source Nodes: [x_1178, x_1179, x_1180], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf329 = extern_kernels.convolution(buf328, arg556_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf329, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg556_1
        # Topologically Sorted Source Nodes: [x_1181], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, arg557_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg557_1
        del buf329
        buf331 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [x_1182, x_1183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf331, arg558_1, arg559_1, arg560_1, arg561_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg558_1
        del arg559_1
        del arg560_1
        del arg561_1
        # Topologically Sorted Source Nodes: [x_1182, x_1183, x_1184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf332 = extern_kernels.convolution(buf331, arg562_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf332, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg562_1
        del buf331
        # Topologically Sorted Source Nodes: [x_1185], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, arg563_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg563_1
        buf334 = buf327; del buf327  # reuse
        buf335 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [x_1177, x_1186, x_comb_iter_102, x_1187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53.run(buf334, arg552_1, arg553_1, arg554_1, arg555_1, buf333, arg564_1, arg565_1, arg566_1, arg567_1, buf335, 1524096, grid=grid(1524096), stream=stream0)
        del arg552_1
        del arg553_1
        del arg554_1
        del arg555_1
        del arg564_1
        del arg565_1
        del arg566_1
        del arg567_1
        del buf333
        # Topologically Sorted Source Nodes: [x_1187, x_1188], Original ATen: [aten.relu, aten.convolution]
        buf336 = extern_kernels.convolution(buf335, arg568_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf336, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg568_1
        del buf335
        # Topologically Sorted Source Nodes: [x_1189], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, arg569_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg569_1
        del buf336
        buf338 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [x_1190, x_1191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf338, arg570_1, arg571_1, arg572_1, arg573_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg570_1
        del arg571_1
        del arg572_1
        del arg573_1
        # Topologically Sorted Source Nodes: [x_1190, x_1191, x_1192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf339 = extern_kernels.convolution(buf338, arg574_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf339, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg574_1
        del buf338
        # Topologically Sorted Source Nodes: [x_1193], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, arg575_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg575_1
        del buf339
        buf341 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [x_1196, x_1197], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_52.run(buf259, buf341, 6390144, grid=grid(6390144), stream=stream0)
        del buf259
        # Topologically Sorted Source Nodes: [x_1196, x_1197, x_1198], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf342 = extern_kernels.convolution(buf341, arg580_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf342, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg580_1
        del buf341
        # Topologically Sorted Source Nodes: [x_1199], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, arg581_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg581_1
        del buf342
        buf344 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [x_1200, x_1201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf344, arg582_1, arg583_1, arg584_1, arg585_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg582_1
        del arg583_1
        del arg584_1
        del arg585_1
        # Topologically Sorted Source Nodes: [x_1200, x_1201, x_1202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf345 = extern_kernels.convolution(buf344, arg586_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf345, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg586_1
        del buf344
        # Topologically Sorted Source Nodes: [x_1203], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, arg587_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg587_1
        del buf345
        # Topologically Sorted Source Nodes: [x_1205, x_1207], Original ATen: [aten.relu, aten.convolution]
        buf348 = extern_kernels.convolution(buf347, arg592_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg592_1
        del buf347
        buf349 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_1204, x_1208, x_comb_iter_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_54.run(buf349, arg588_1, arg589_1, arg590_1, arg591_1, buf348, arg593_1, arg594_1, arg595_1, arg596_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg588_1
        del arg589_1
        del arg590_1
        del arg591_1
        del arg593_1
        del arg594_1
        del arg595_1
        del arg596_1
        del buf348
        buf351 = reinterpret_tensor(buf116, (8, 2160, 21, 21), (952560, 1, 45360, 2160), 0); del buf116  # reuse
        buf356 = reinterpret_tensor(buf111, (8, 2160, 21, 21), (952560, 1, 45360, 2160), 0); del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_out_20, x_1210, x_1261], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_55.run(buf315, arg528_1, arg529_1, arg530_1, arg531_1, buf260, buf321, arg540_1, arg541_1, arg542_1, arg543_1, buf302, buf334, buf340, arg576_1, arg577_1, arg578_1, arg579_1, buf303, buf349, buf351, buf356, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del arg528_1
        del arg529_1
        del arg530_1
        del arg531_1
        del arg540_1
        del arg541_1
        del arg542_1
        del arg543_1
        del arg576_1
        del arg577_1
        del arg578_1
        del arg579_1
        # Topologically Sorted Source Nodes: [x_1210, x_1211], Original ATen: [aten.relu, aten.convolution]
        buf352 = extern_kernels.convolution(buf351, arg603_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg603_1
        buf353 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [x_1212], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf353, arg604_1, arg605_1, arg606_1, arg607_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg604_1
        del arg605_1
        del arg606_1
        del arg607_1
        buf354 = buf349; del buf349  # reuse
        buf355 = buf340; del buf340  # reuse
        buf366 = buf334; del buf334  # reuse
        buf372 = buf321; del buf321  # reuse
        buf378 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_21, x_comb_iter_3_right_21, x_1221, x_1229, x_1237], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_57.run(buf353, buf354, buf355, buf366, buf372, buf378, 1524096, grid=grid(1524096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1261, x_1262], Original ATen: [aten.relu, aten.convolution]
        buf357 = extern_kernels.convolution(buf356, arg680_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg680_1
        buf358 = buf357; del buf357  # reuse
        buf407 = buf303; del buf303  # reuse
        buf438 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [x_1263, x_1267, x_1307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf358, arg681_1, arg682_1, arg683_1, arg684_1, buf407, buf438, 1524096, grid=grid(1524096), stream=stream0)
        del arg681_1
        del arg682_1
        del arg683_1
        del arg684_1
        buf359 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_20], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_59.run(buf358, buf359, 1524096, grid=grid(1524096), stream=stream0)
        del buf358
        # Topologically Sorted Source Nodes: [x_1213, x_1214], Original ATen: [aten.relu, aten.convolution]
        buf361 = extern_kernels.convolution(buf360, arg608_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf361, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg608_1
        del buf360
        # Topologically Sorted Source Nodes: [x_1215], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, arg609_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg609_1
        del buf361
        buf363 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [x_1216, x_1217], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf363, arg610_1, arg611_1, arg612_1, arg613_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg610_1
        del arg611_1
        del arg612_1
        del arg613_1
        # Topologically Sorted Source Nodes: [x_1216, x_1217, x_1218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf364 = extern_kernels.convolution(buf363, arg614_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf364, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg614_1
        del buf363
        # Topologically Sorted Source Nodes: [x_1219], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, arg615_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg615_1
        del buf364
        # Topologically Sorted Source Nodes: [x_1221, x_1222], Original ATen: [aten.relu, aten.convolution]
        buf367 = extern_kernels.convolution(buf366, arg620_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf367, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg620_1
        del buf366
        # Topologically Sorted Source Nodes: [x_1223], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, arg621_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg621_1
        del buf367
        buf369 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [x_1224, x_1225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf369, arg622_1, arg623_1, arg624_1, arg625_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg622_1
        del arg623_1
        del arg624_1
        del arg625_1
        # Topologically Sorted Source Nodes: [x_1224, x_1225, x_1226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf370 = extern_kernels.convolution(buf369, arg626_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf370, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg626_1
        del buf369
        # Topologically Sorted Source Nodes: [x_1227], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, arg627_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg627_1
        del buf370
        # Topologically Sorted Source Nodes: [x_1229, x_1230], Original ATen: [aten.relu, aten.convolution]
        buf373 = extern_kernels.convolution(buf372, arg632_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf373, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg632_1
        del buf372
        # Topologically Sorted Source Nodes: [x_1231], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, arg633_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg633_1
        del buf373
        buf375 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [x_1232, x_1233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf375, arg634_1, arg635_1, arg636_1, arg637_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg634_1
        del arg635_1
        del arg636_1
        del arg637_1
        # Topologically Sorted Source Nodes: [x_1232, x_1233, x_1234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf376 = extern_kernels.convolution(buf375, arg638_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf376, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg638_1
        del buf375
        # Topologically Sorted Source Nodes: [x_1235], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, arg639_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg639_1
        del buf376
        # Topologically Sorted Source Nodes: [x_1237, x_1238], Original ATen: [aten.relu, aten.convolution]
        buf379 = extern_kernels.convolution(buf378, arg644_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf379, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg644_1
        del buf378
        # Topologically Sorted Source Nodes: [x_1239], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, arg645_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg645_1
        del buf379
        buf381 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [x_1240, x_1241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf381, arg646_1, arg647_1, arg648_1, arg649_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg646_1
        del arg647_1
        del arg648_1
        del arg649_1
        # Topologically Sorted Source Nodes: [x_1240, x_1241, x_1242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf382 = extern_kernels.convolution(buf381, arg650_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf382, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg650_1
        del buf381
        # Topologically Sorted Source Nodes: [x_1243], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, arg651_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg651_1
        buf384 = buf377; del buf377  # reuse
        buf385 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [x_1236, x_1244, x_comb_iter_107, x_1245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53.run(buf384, arg640_1, arg641_1, arg642_1, arg643_1, buf383, arg652_1, arg653_1, arg654_1, arg655_1, buf385, 1524096, grid=grid(1524096), stream=stream0)
        del arg640_1
        del arg641_1
        del arg642_1
        del arg643_1
        del arg652_1
        del arg653_1
        del arg654_1
        del arg655_1
        del buf383
        # Topologically Sorted Source Nodes: [x_1245, x_1246], Original ATen: [aten.relu, aten.convolution]
        buf386 = extern_kernels.convolution(buf385, arg656_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf386, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg656_1
        del buf385
        # Topologically Sorted Source Nodes: [x_1247], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, arg657_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg657_1
        del buf386
        buf388 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_1248, x_1249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf388, arg658_1, arg659_1, arg660_1, arg661_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg658_1
        del arg659_1
        del arg660_1
        del arg661_1
        # Topologically Sorted Source Nodes: [x_1248, x_1249, x_1250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf389 = extern_kernels.convolution(buf388, arg662_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf389, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg662_1
        del buf388
        # Topologically Sorted Source Nodes: [x_1251], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, arg663_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg663_1
        del buf389
        # Topologically Sorted Source Nodes: [x_1253, x_1254], Original ATen: [aten.relu, aten.convolution]
        buf392 = extern_kernels.convolution(buf391, arg668_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf392, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg668_1
        del buf391
        # Topologically Sorted Source Nodes: [x_1255], Original ATen: [aten.convolution]
        buf393 = extern_kernels.convolution(buf392, arg669_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg669_1
        del buf392
        buf394 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [x_1256, x_1257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf394, arg670_1, arg671_1, arg672_1, arg673_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg670_1
        del arg671_1
        del arg672_1
        del arg673_1
        # Topologically Sorted Source Nodes: [x_1256, x_1257, x_1258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf395 = extern_kernels.convolution(buf394, arg674_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf395, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg674_1
        del buf394
        # Topologically Sorted Source Nodes: [x_1259], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, arg675_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg675_1
        del buf395
        buf398 = buf356; del buf356  # reuse
        buf403 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [x_out_21, x_1264, x_1315], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_60.run(buf365, arg616_1, arg617_1, arg618_1, arg619_1, buf309, buf371, arg628_1, arg629_1, arg630_1, arg631_1, buf354, buf384, buf390, arg664_1, arg665_1, arg666_1, arg667_1, buf355, buf396, arg676_1, arg677_1, arg678_1, arg679_1, buf353, buf398, buf403, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del arg616_1
        del arg617_1
        del arg618_1
        del arg619_1
        del arg628_1
        del arg629_1
        del arg630_1
        del arg631_1
        del arg664_1
        del arg665_1
        del arg666_1
        del arg667_1
        del arg676_1
        del arg677_1
        del arg678_1
        del arg679_1
        del buf309
        # Topologically Sorted Source Nodes: [x_1264, x_1265], Original ATen: [aten.relu, aten.convolution]
        buf399 = extern_kernels.convolution(buf398, arg685_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg685_1
        buf400 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [x_1266], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf400, arg686_1, arg687_1, arg688_1, arg689_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg686_1
        del arg687_1
        del arg688_1
        del arg689_1
        buf401 = buf396; del buf396  # reuse
        buf402 = buf390; del buf390  # reuse
        buf413 = buf384; del buf384  # reuse
        buf419 = buf371; del buf371  # reuse
        buf425 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_22, x_comb_iter_3_right_22, x_1275, x_1283, x_1291], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_57.run(buf400, buf401, buf402, buf413, buf419, buf425, 1524096, grid=grid(1524096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1315, x_1316], Original ATen: [aten.relu, aten.convolution]
        buf404 = extern_kernels.convolution(buf403, arg762_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg762_1
        buf405 = buf404; del buf404  # reuse
        buf454 = buf355; del buf355  # reuse
        buf485 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [x_1317, x_1321, x_1361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf405, arg763_1, arg764_1, arg765_1, arg766_1, buf454, buf485, 1524096, grid=grid(1524096), stream=stream0)
        del arg763_1
        del arg764_1
        del arg765_1
        del arg766_1
        buf406 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_21], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_59.run(buf405, buf406, 1524096, grid=grid(1524096), stream=stream0)
        del buf405
        # Topologically Sorted Source Nodes: [x_1267, x_1268], Original ATen: [aten.relu, aten.convolution]
        buf408 = extern_kernels.convolution(buf407, arg690_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf408, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg690_1
        del buf407
        # Topologically Sorted Source Nodes: [x_1269], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, arg691_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg691_1
        del buf408
        buf410 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [x_1270, x_1271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf410, arg692_1, arg693_1, arg694_1, arg695_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg692_1
        del arg693_1
        del arg694_1
        del arg695_1
        # Topologically Sorted Source Nodes: [x_1270, x_1271, x_1272], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf411 = extern_kernels.convolution(buf410, arg696_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf411, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg696_1
        del buf410
        # Topologically Sorted Source Nodes: [x_1273], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, arg697_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg697_1
        del buf411
        # Topologically Sorted Source Nodes: [x_1275, x_1276], Original ATen: [aten.relu, aten.convolution]
        buf414 = extern_kernels.convolution(buf413, arg702_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf414, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg702_1
        del buf413
        # Topologically Sorted Source Nodes: [x_1277], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf414, arg703_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg703_1
        del buf414
        buf416 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [x_1278, x_1279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf416, arg704_1, arg705_1, arg706_1, arg707_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg704_1
        del arg705_1
        del arg706_1
        del arg707_1
        # Topologically Sorted Source Nodes: [x_1278, x_1279, x_1280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf417 = extern_kernels.convolution(buf416, arg708_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf417, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg708_1
        del buf416
        # Topologically Sorted Source Nodes: [x_1281], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, arg709_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg709_1
        del buf417
        # Topologically Sorted Source Nodes: [x_1283, x_1284], Original ATen: [aten.relu, aten.convolution]
        buf420 = extern_kernels.convolution(buf419, arg714_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf420, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg714_1
        del buf419
        # Topologically Sorted Source Nodes: [x_1285], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf420, arg715_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg715_1
        del buf420
        buf422 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [x_1286, x_1287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf422, arg716_1, arg717_1, arg718_1, arg719_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg716_1
        del arg717_1
        del arg718_1
        del arg719_1
        # Topologically Sorted Source Nodes: [x_1286, x_1287, x_1288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf423 = extern_kernels.convolution(buf422, arg720_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf423, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg720_1
        del buf422
        # Topologically Sorted Source Nodes: [x_1289], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, arg721_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg721_1
        del buf423
        # Topologically Sorted Source Nodes: [x_1291, x_1292], Original ATen: [aten.relu, aten.convolution]
        buf426 = extern_kernels.convolution(buf425, arg726_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf426, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg726_1
        del buf425
        # Topologically Sorted Source Nodes: [x_1293], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, arg727_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg727_1
        del buf426
        buf428 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [x_1294, x_1295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf428, arg728_1, arg729_1, arg730_1, arg731_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg728_1
        del arg729_1
        del arg730_1
        del arg731_1
        # Topologically Sorted Source Nodes: [x_1294, x_1295, x_1296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf429 = extern_kernels.convolution(buf428, arg732_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf429, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg732_1
        del buf428
        # Topologically Sorted Source Nodes: [x_1297], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, arg733_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg733_1
        buf431 = buf424; del buf424  # reuse
        buf432 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [x_1290, x_1298, x_comb_iter_112, x_1299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53.run(buf431, arg722_1, arg723_1, arg724_1, arg725_1, buf430, arg734_1, arg735_1, arg736_1, arg737_1, buf432, 1524096, grid=grid(1524096), stream=stream0)
        del arg722_1
        del arg723_1
        del arg724_1
        del arg725_1
        del arg734_1
        del arg735_1
        del arg736_1
        del arg737_1
        del buf430
        # Topologically Sorted Source Nodes: [x_1299, x_1300], Original ATen: [aten.relu, aten.convolution]
        buf433 = extern_kernels.convolution(buf432, arg738_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf433, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg738_1
        del buf432
        # Topologically Sorted Source Nodes: [x_1301], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf433, arg739_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg739_1
        del buf433
        buf435 = buf434; del buf434  # reuse
        # Topologically Sorted Source Nodes: [x_1302, x_1303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf435, arg740_1, arg741_1, arg742_1, arg743_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg740_1
        del arg741_1
        del arg742_1
        del arg743_1
        # Topologically Sorted Source Nodes: [x_1302, x_1303, x_1304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf436 = extern_kernels.convolution(buf435, arg744_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf436, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg744_1
        del buf435
        # Topologically Sorted Source Nodes: [x_1305], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, arg745_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg745_1
        del buf436
        # Topologically Sorted Source Nodes: [x_1307, x_1308], Original ATen: [aten.relu, aten.convolution]
        buf439 = extern_kernels.convolution(buf438, arg750_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf439, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg750_1
        del buf438
        # Topologically Sorted Source Nodes: [x_1309], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, arg751_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg751_1
        del buf439
        buf441 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [x_1310, x_1311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf441, arg752_1, arg753_1, arg754_1, arg755_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg752_1
        del arg753_1
        del arg754_1
        del arg755_1
        # Topologically Sorted Source Nodes: [x_1310, x_1311, x_1312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf442 = extern_kernels.convolution(buf441, arg756_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf442, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg756_1
        del buf441
        # Topologically Sorted Source Nodes: [x_1313], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, arg757_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg757_1
        del buf442
        buf445 = buf403; del buf403  # reuse
        buf450 = buf398; del buf398  # reuse
        # Topologically Sorted Source Nodes: [x_out_22, x_1318, x_1369], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_61.run(buf412, arg698_1, arg699_1, arg700_1, arg701_1, buf359, buf418, arg710_1, arg711_1, arg712_1, arg713_1, buf401, buf431, buf437, arg746_1, arg747_1, arg748_1, arg749_1, buf402, buf443, arg758_1, arg759_1, arg760_1, arg761_1, buf400, buf445, buf450, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del arg698_1
        del arg699_1
        del arg700_1
        del arg701_1
        del arg710_1
        del arg711_1
        del arg712_1
        del arg713_1
        del arg746_1
        del arg747_1
        del arg748_1
        del arg749_1
        del arg758_1
        del arg759_1
        del arg760_1
        del arg761_1
        del buf359
        del buf400
        del buf401
        del buf402
        # Topologically Sorted Source Nodes: [x_1318, x_1319], Original ATen: [aten.relu, aten.convolution]
        buf446 = extern_kernels.convolution(buf445, arg767_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg767_1
        del buf445
        buf447 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [x_1320], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_56.run(buf447, arg768_1, arg769_1, arg770_1, arg771_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg768_1
        del arg769_1
        del arg770_1
        del arg771_1
        buf448 = buf443; del buf443  # reuse
        buf449 = buf437; del buf437  # reuse
        buf460 = buf431; del buf431  # reuse
        buf466 = buf418; del buf418  # reuse
        buf472 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_23, x_comb_iter_3_right_23, x_1329, x_1337, x_1345], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_57.run(buf447, buf448, buf449, buf460, buf466, buf472, 1524096, grid=grid(1524096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1369, x_1370], Original ATen: [aten.relu, aten.convolution]
        buf451 = extern_kernels.convolution(buf450, arg844_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (8, 864, 21, 21), (381024, 1, 18144, 864))
        del arg844_1
        buf452 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [x_1371], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_62.run(buf452, arg845_1, arg846_1, arg847_1, arg848_1, 3048192, grid=grid(3048192), stream=stream0)
        del arg845_1
        del arg846_1
        del arg847_1
        del arg848_1
        buf453 = empty_strided_cuda((8, 864, 11, 11), (104544, 1, 9504, 864), torch.float32)
        # Topologically Sorted Source Nodes: [x_1384, x_comb_iter_0_right_22], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_63.run(buf452, buf453, 836352, grid=grid(836352), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1321, x_1322], Original ATen: [aten.relu, aten.convolution]
        buf455 = extern_kernels.convolution(buf454, arg772_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf455, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg772_1
        del buf454
        # Topologically Sorted Source Nodes: [x_1323], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, arg773_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg773_1
        del buf455
        buf457 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [x_1324, x_1325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf457, arg774_1, arg775_1, arg776_1, arg777_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg774_1
        del arg775_1
        del arg776_1
        del arg777_1
        # Topologically Sorted Source Nodes: [x_1324, x_1325, x_1326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf458 = extern_kernels.convolution(buf457, arg778_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf458, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg778_1
        del buf457
        # Topologically Sorted Source Nodes: [x_1327], Original ATen: [aten.convolution]
        buf459 = extern_kernels.convolution(buf458, arg779_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf459, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg779_1
        del buf458
        # Topologically Sorted Source Nodes: [x_1329, x_1330], Original ATen: [aten.relu, aten.convolution]
        buf461 = extern_kernels.convolution(buf460, arg784_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf461, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg784_1
        del buf460
        # Topologically Sorted Source Nodes: [x_1331], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, arg785_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg785_1
        del buf461
        buf463 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [x_1332, x_1333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf463, arg786_1, arg787_1, arg788_1, arg789_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg786_1
        del arg787_1
        del arg788_1
        del arg789_1
        # Topologically Sorted Source Nodes: [x_1332, x_1333, x_1334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf464 = extern_kernels.convolution(buf463, arg790_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf464, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg790_1
        del buf463
        # Topologically Sorted Source Nodes: [x_1335], Original ATen: [aten.convolution]
        buf465 = extern_kernels.convolution(buf464, arg791_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg791_1
        del buf464
        # Topologically Sorted Source Nodes: [x_1337, x_1338], Original ATen: [aten.relu, aten.convolution]
        buf467 = extern_kernels.convolution(buf466, arg796_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf467, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg796_1
        del buf466
        # Topologically Sorted Source Nodes: [x_1339], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(buf467, arg797_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg797_1
        del buf467
        buf469 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [x_1340, x_1341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf469, arg798_1, arg799_1, arg800_1, arg801_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg798_1
        del arg799_1
        del arg800_1
        del arg801_1
        # Topologically Sorted Source Nodes: [x_1340, x_1341, x_1342], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf470 = extern_kernels.convolution(buf469, arg802_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf470, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg802_1
        del buf469
        # Topologically Sorted Source Nodes: [x_1343], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, arg803_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg803_1
        del buf470
        # Topologically Sorted Source Nodes: [x_1345, x_1346], Original ATen: [aten.relu, aten.convolution]
        buf473 = extern_kernels.convolution(buf472, arg808_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf473, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg808_1
        del buf472
        # Topologically Sorted Source Nodes: [x_1347], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, arg809_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg809_1
        del buf473
        buf475 = buf474; del buf474  # reuse
        # Topologically Sorted Source Nodes: [x_1348, x_1349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf475, arg810_1, arg811_1, arg812_1, arg813_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg810_1
        del arg811_1
        del arg812_1
        del arg813_1
        # Topologically Sorted Source Nodes: [x_1348, x_1349, x_1350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf476 = extern_kernels.convolution(buf475, arg814_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf476, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg814_1
        del buf475
        # Topologically Sorted Source Nodes: [x_1351], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, arg815_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg815_1
        buf478 = buf471; del buf471  # reuse
        buf479 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [x_1344, x_1352, x_comb_iter_117, x_1353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_53.run(buf478, arg804_1, arg805_1, arg806_1, arg807_1, buf477, arg816_1, arg817_1, arg818_1, arg819_1, buf479, 1524096, grid=grid(1524096), stream=stream0)
        del arg804_1
        del arg805_1
        del arg806_1
        del arg807_1
        del arg816_1
        del arg817_1
        del arg818_1
        del arg819_1
        del buf477
        # Topologically Sorted Source Nodes: [x_1353, x_1354], Original ATen: [aten.relu, aten.convolution]
        buf480 = extern_kernels.convolution(buf479, arg820_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf480, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg820_1
        del buf479
        # Topologically Sorted Source Nodes: [x_1355], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, arg821_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg821_1
        del buf480
        buf482 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [x_1356, x_1357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf482, arg822_1, arg823_1, arg824_1, arg825_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg822_1
        del arg823_1
        del arg824_1
        del arg825_1
        # Topologically Sorted Source Nodes: [x_1356, x_1357, x_1358], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf483 = extern_kernels.convolution(buf482, arg826_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf483, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg826_1
        del buf482
        # Topologically Sorted Source Nodes: [x_1359], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, arg827_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg827_1
        del buf483
        # Topologically Sorted Source Nodes: [x_1361, x_1362], Original ATen: [aten.relu, aten.convolution]
        buf486 = extern_kernels.convolution(buf485, arg832_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf486, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg832_1
        del buf485
        # Topologically Sorted Source Nodes: [x_1363], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, arg833_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg833_1
        del buf486
        buf488 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [x_1364, x_1365], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf488, arg834_1, arg835_1, arg836_1, arg837_1, 1524096, grid=grid(1524096), stream=stream0)
        del arg834_1
        del arg835_1
        del arg836_1
        del arg837_1
        # Topologically Sorted Source Nodes: [x_1364, x_1365, x_1366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf489 = extern_kernels.convolution(buf488, arg838_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=432, bias=None)
        assert_size_stride(buf489, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg838_1
        del buf488
        # Topologically Sorted Source Nodes: [x_1367], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, arg839_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (8, 432, 21, 21), (190512, 1, 9072, 432))
        del arg839_1
        del buf489
        buf491 = empty_strided_cuda((8, 2160, 21, 21), (952576, 441, 21, 1), torch.float32)
        buf492 = buf450; del buf450  # reuse
        # Topologically Sorted Source Nodes: [x_out_23, x_1372], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_64.run(buf459, arg780_1, arg781_1, arg782_1, arg783_1, buf406, buf465, arg792_1, arg793_1, arg794_1, arg795_1, buf448, buf478, buf484, arg828_1, arg829_1, arg830_1, arg831_1, buf449, buf490, arg840_1, arg841_1, arg842_1, arg843_1, buf447, buf491, buf492, 17280, 441, grid=grid(17280, 441), stream=stream0)
        del arg780_1
        del arg781_1
        del arg782_1
        del arg783_1
        del arg792_1
        del arg793_1
        del arg794_1
        del arg795_1
        del arg828_1
        del arg829_1
        del arg830_1
        del arg831_1
        del arg840_1
        del arg841_1
        del arg842_1
        del arg843_1
        del buf406
        del buf447
        del buf448
        del buf449
        del buf459
        del buf465
        del buf478
        del buf484
        del buf490
        # Topologically Sorted Source Nodes: [x_1372, x_1373], Original ATen: [aten.relu, aten.convolution]
        buf493 = extern_kernels.convolution(buf492, arg849_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf493, (8, 864, 21, 21), (381024, 1, 18144, 864))
        del arg849_1
        del buf492
        buf494 = buf493; del buf493  # reuse
        buf540 = reinterpret_tensor(buf297, (8, 864, 21, 21), (381024, 1, 18144, 864), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [x_1374, x_1431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_65.run(buf494, arg850_1, arg851_1, arg852_1, arg853_1, buf540, 3048192, grid=grid(3048192), stream=stream0)
        del arg850_1
        del arg851_1
        del arg852_1
        del arg853_1
        buf495 = empty_strided_cuda((8, 864, 11, 11), (104544, 1, 9504, 864), torch.float32)
        buf496 = empty_strided_cuda((8, 864, 11, 11), (104544, 1, 9504, 864), torch.float32)
        # Topologically Sorted Source Nodes: [x_1394, x_comb_iter_1_right_24, x_1421, x_comb_iter_3_right_24], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
        triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_66.run(buf494, buf495, buf496, 836352, grid=grid(836352), stream=stream0)
        buf497 = empty_strided_cuda((8, 2160, 11, 11), (261360, 1, 23760, 2160), torch.float32)
        # Topologically Sorted Source Nodes: [x_1435, input_42], Original ATen: [aten.relu, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_relu_67.run(buf491, buf497, 17280, 121, grid=grid(17280, 121), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1435, input_42, input_43], Original ATen: [aten.relu, aten.avg_pool2d, aten.convolution]
        buf498 = extern_kernels.convolution(buf497, arg931_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (8, 432, 11, 11), (52272, 1, 4752, 432))
        del arg931_1
        buf499 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [x_1435, input_44, input_45], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_constant_pad_nd_relu_68.run(buf491, buf499, 17280, 121, grid=grid(17280, 121), stream=stream0)
        del buf491
        # Topologically Sorted Source Nodes: [x_1435, input_44, input_45, input_46], Original ATen: [aten.relu, aten.constant_pad_nd, aten.avg_pool2d, aten.convolution]
        buf500 = extern_kernels.convolution(buf499, arg932_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (8, 432, 11, 11), (52272, 1, 4752, 432))
        del arg932_1
        del buf499
        buf501 = empty_strided_cuda((8, 864, 11, 11), (104544, 121, 11, 1), torch.float32)
        buf553 = empty_strided_cuda((8, 864, 11, 11), (104544, 1, 9504, 864), torch.float32)
        buf584 = empty_strided_cuda((8, 864, 11, 11), (104544, 1, 9504, 864), torch.float32)
        # Topologically Sorted Source Nodes: [cat_32, out_7, x_1439, x_1479], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_69.run(buf498, buf500, arg933_1, arg934_1, arg935_1, arg936_1, buf501, buf553, buf584, 6912, 121, grid=grid(6912, 121), stream=stream0)
        del arg933_1
        del arg934_1
        del arg935_1
        del arg936_1
        del buf498
        del buf500
        buf502 = empty_strided_cuda((8, 864, 11, 11), (104544, 121, 11, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_23], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_70.run(buf501, buf502, 836352, grid=grid(836352), stream=stream0)
        del buf501
        buf503 = empty_strided_cuda((8, 864, 25, 25), (540000, 1, 21600, 864), torch.float32)
        # Topologically Sorted Source Nodes: [x_1375, x_1376], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_71.run(buf452, buf503, 4320000, grid=grid(4320000), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1375, x_1376, x_1377], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf504 = extern_kernels.convolution(buf503, arg854_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf504, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg854_1
        # Topologically Sorted Source Nodes: [x_1378], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, arg855_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg855_1
        del buf504
        buf506 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [x_1379, x_1380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf506, arg856_1, arg857_1, arg858_1, arg859_1, 836352, grid=grid(836352), stream=stream0)
        del arg856_1
        del arg857_1
        del arg858_1
        del arg859_1
        # Topologically Sorted Source Nodes: [x_1379, x_1380, x_1381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf507 = extern_kernels.convolution(buf506, arg860_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf507, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg860_1
        del buf506
        # Topologically Sorted Source Nodes: [x_1382], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, arg861_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg861_1
        del buf507
        buf509 = empty_strided_cuda((8, 864, 27, 27), (629856, 1, 23328, 864), torch.float32)
        # Topologically Sorted Source Nodes: [x_1385, x_1386], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_73.run(buf494, buf509, 5038848, grid=grid(5038848), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1385, x_1386, x_1387], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf510 = extern_kernels.convolution(buf509, arg866_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf510, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg866_1
        del buf509
        # Topologically Sorted Source Nodes: [x_1388], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf510, arg867_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg867_1
        del buf510
        buf512 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [x_1389, x_1390], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf512, arg868_1, arg869_1, arg870_1, arg871_1, 836352, grid=grid(836352), stream=stream0)
        del arg868_1
        del arg869_1
        del arg870_1
        del arg871_1
        # Topologically Sorted Source Nodes: [x_1389, x_1390, x_1391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf513 = extern_kernels.convolution(buf512, arg872_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf513, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg872_1
        del buf512
        # Topologically Sorted Source Nodes: [x_1392], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf513, arg873_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg873_1
        del buf513
        buf515 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [x_1395, x_1396], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_71.run(buf494, buf515, 4320000, grid=grid(4320000), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1395, x_1396, x_1397], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf516 = extern_kernels.convolution(buf515, arg878_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf516, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg878_1
        del buf515
        # Topologically Sorted Source Nodes: [x_1398], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, arg879_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg879_1
        del buf516
        buf518 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [x_1399, x_1400], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf518, arg880_1, arg881_1, arg882_1, arg883_1, 836352, grid=grid(836352), stream=stream0)
        del arg880_1
        del arg881_1
        del arg882_1
        del arg883_1
        # Topologically Sorted Source Nodes: [x_1399, x_1400, x_1401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf519 = extern_kernels.convolution(buf518, arg884_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf519, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg884_1
        del buf518
        # Topologically Sorted Source Nodes: [x_1402], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, arg885_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg885_1
        del buf519
        buf521 = empty_strided_cuda((8, 864, 23, 23), (457056, 1, 19872, 864), torch.float32)
        # Topologically Sorted Source Nodes: [x_1404, x_1405], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_74.run(buf494, buf521, 3656448, grid=grid(3656448), stream=stream0)
        del buf494
        # Topologically Sorted Source Nodes: [x_1404, x_1405, x_1406], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf522 = extern_kernels.convolution(buf521, arg890_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf522, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg890_1
        # Topologically Sorted Source Nodes: [x_1407], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, arg891_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg891_1
        del buf522
        buf524 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [x_1408, x_1409], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf524, arg892_1, arg893_1, arg894_1, arg895_1, 836352, grid=grid(836352), stream=stream0)
        del arg892_1
        del arg893_1
        del arg894_1
        del arg895_1
        # Topologically Sorted Source Nodes: [x_1408, x_1409, x_1410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf525 = extern_kernels.convolution(buf524, arg896_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf525, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg896_1
        del buf524
        # Topologically Sorted Source Nodes: [x_1411], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, arg897_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg897_1
        buf527 = buf520; del buf520  # reuse
        buf528 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [x_1403, x_1412, x_comb_iter_122, x_1413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75.run(buf527, arg886_1, arg887_1, arg888_1, arg889_1, buf526, arg898_1, arg899_1, arg900_1, arg901_1, buf528, 836352, grid=grid(836352), stream=stream0)
        del arg886_1
        del arg887_1
        del arg888_1
        del arg889_1
        del arg898_1
        del arg899_1
        del arg900_1
        del arg901_1
        del buf526
        # Topologically Sorted Source Nodes: [x_1413, x_1414], Original ATen: [aten.relu, aten.convolution]
        buf529 = extern_kernels.convolution(buf528, arg902_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf529, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg902_1
        del buf528
        # Topologically Sorted Source Nodes: [x_1415], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf529, arg903_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg903_1
        del buf529
        buf531 = buf530; del buf530  # reuse
        # Topologically Sorted Source Nodes: [x_1416, x_1417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf531, arg904_1, arg905_1, arg906_1, arg907_1, 836352, grid=grid(836352), stream=stream0)
        del arg904_1
        del arg905_1
        del arg906_1
        del arg907_1
        # Topologically Sorted Source Nodes: [x_1416, x_1417, x_1418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf532 = extern_kernels.convolution(buf531, arg908_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf532, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg908_1
        del buf531
        # Topologically Sorted Source Nodes: [x_1419], Original ATen: [aten.convolution]
        buf533 = extern_kernels.convolution(buf532, arg909_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg909_1
        del buf532
        buf534 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [x_1422, x_1423], Original ATen: [aten.relu, aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_relu_74.run(buf452, buf534, 3656448, grid=grid(3656448), stream=stream0)
        del buf452
        # Topologically Sorted Source Nodes: [x_1422, x_1423, x_1424], Original ATen: [aten.relu, aten.constant_pad_nd, aten.convolution]
        buf535 = extern_kernels.convolution(buf534, arg914_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf535, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg914_1
        del buf534
        # Topologically Sorted Source Nodes: [x_1425], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, arg915_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg915_1
        del buf535
        buf537 = buf536; del buf536  # reuse
        # Topologically Sorted Source Nodes: [x_1426, x_1427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf537, arg916_1, arg917_1, arg918_1, arg919_1, 836352, grid=grid(836352), stream=stream0)
        del arg916_1
        del arg917_1
        del arg918_1
        del arg919_1
        # Topologically Sorted Source Nodes: [x_1426, x_1427, x_1428], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf538 = extern_kernels.convolution(buf537, arg920_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf538, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg920_1
        del buf537
        # Topologically Sorted Source Nodes: [x_1429], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, arg921_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg921_1
        del buf538
        # Topologically Sorted Source Nodes: [x_1431, x_1433], Original ATen: [aten.relu, aten.convolution]
        buf541 = extern_kernels.convolution(buf540, arg926_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg926_1
        del buf540
        buf542 = buf539; del buf539  # reuse
        # Topologically Sorted Source Nodes: [x_1430, x_1434, x_comb_iter_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_76.run(buf542, arg922_1, arg923_1, arg924_1, arg925_1, buf541, arg927_1, arg928_1, arg929_1, arg930_1, 836352, grid=grid(836352), stream=stream0)
        del arg922_1
        del arg923_1
        del arg924_1
        del arg925_1
        del arg927_1
        del arg928_1
        del arg929_1
        del arg930_1
        del buf541
        buf544 = empty_strided_cuda((8, 4320, 11, 11), (522720, 1, 47520, 4320), torch.float32)
        buf549 = empty_strided_cuda((8, 4320, 11, 11), (522720, 1, 47520, 4320), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_24, x_1436, x_1487], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_77.run(buf508, arg862_1, arg863_1, arg864_1, arg865_1, buf453, buf514, arg874_1, arg875_1, arg876_1, arg877_1, buf495, buf527, buf533, arg910_1, arg911_1, arg912_1, arg913_1, buf496, buf542, buf544, buf549, 34560, 121, grid=grid(34560, 121), stream=stream0)
        del arg862_1
        del arg863_1
        del arg864_1
        del arg865_1
        del arg874_1
        del arg875_1
        del arg876_1
        del arg877_1
        del arg910_1
        del arg911_1
        del arg912_1
        del arg913_1
        # Topologically Sorted Source Nodes: [x_1436, x_1437], Original ATen: [aten.relu, aten.convolution]
        buf545 = extern_kernels.convolution(buf544, arg937_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg937_1
        buf546 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [x_1438], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_78.run(buf546, arg938_1, arg939_1, arg940_1, arg941_1, 836352, grid=grid(836352), stream=stream0)
        del arg938_1
        del arg939_1
        del arg940_1
        del arg941_1
        buf547 = buf542; del buf542  # reuse
        buf548 = buf533; del buf533  # reuse
        buf559 = buf527; del buf527  # reuse
        buf565 = buf514; del buf514  # reuse
        buf571 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_25, x_comb_iter_3_right_25, x_1447, x_1455, x_1463], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_79.run(buf546, buf547, buf548, buf559, buf565, buf571, 836352, grid=grid(836352), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1487, x_1488], Original ATen: [aten.relu, aten.convolution]
        buf550 = extern_kernels.convolution(buf549, arg1014_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1014_1
        buf551 = buf550; del buf550  # reuse
        buf600 = buf496; del buf496  # reuse
        buf631 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [x_1489, x_1493, x_1533], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_80.run(buf551, arg1015_1, arg1016_1, arg1017_1, arg1018_1, buf600, buf631, 836352, grid=grid(836352), stream=stream0)
        del arg1015_1
        del arg1016_1
        del arg1017_1
        del arg1018_1
        buf552 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_24], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_81.run(buf551, buf552, 836352, grid=grid(836352), stream=stream0)
        del buf551
        # Topologically Sorted Source Nodes: [x_1439, x_1440], Original ATen: [aten.relu, aten.convolution]
        buf554 = extern_kernels.convolution(buf553, arg942_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf554, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg942_1
        del buf553
        # Topologically Sorted Source Nodes: [x_1441], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(buf554, arg943_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf555, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg943_1
        del buf554
        buf556 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [x_1442, x_1443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf556, arg944_1, arg945_1, arg946_1, arg947_1, 836352, grid=grid(836352), stream=stream0)
        del arg944_1
        del arg945_1
        del arg946_1
        del arg947_1
        # Topologically Sorted Source Nodes: [x_1442, x_1443, x_1444], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf557 = extern_kernels.convolution(buf556, arg948_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf557, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg948_1
        del buf556
        # Topologically Sorted Source Nodes: [x_1445], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, arg949_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg949_1
        del buf557
        # Topologically Sorted Source Nodes: [x_1447, x_1448], Original ATen: [aten.relu, aten.convolution]
        buf560 = extern_kernels.convolution(buf559, arg954_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf560, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg954_1
        del buf559
        # Topologically Sorted Source Nodes: [x_1449], Original ATen: [aten.convolution]
        buf561 = extern_kernels.convolution(buf560, arg955_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf561, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg955_1
        del buf560
        buf562 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [x_1450, x_1451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf562, arg956_1, arg957_1, arg958_1, arg959_1, 836352, grid=grid(836352), stream=stream0)
        del arg956_1
        del arg957_1
        del arg958_1
        del arg959_1
        # Topologically Sorted Source Nodes: [x_1450, x_1451, x_1452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf563 = extern_kernels.convolution(buf562, arg960_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf563, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg960_1
        del buf562
        # Topologically Sorted Source Nodes: [x_1453], Original ATen: [aten.convolution]
        buf564 = extern_kernels.convolution(buf563, arg961_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf564, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg961_1
        del buf563
        # Topologically Sorted Source Nodes: [x_1455, x_1456], Original ATen: [aten.relu, aten.convolution]
        buf566 = extern_kernels.convolution(buf565, arg966_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf566, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg966_1
        del buf565
        # Topologically Sorted Source Nodes: [x_1457], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, arg967_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg967_1
        del buf566
        buf568 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [x_1458, x_1459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf568, arg968_1, arg969_1, arg970_1, arg971_1, 836352, grid=grid(836352), stream=stream0)
        del arg968_1
        del arg969_1
        del arg970_1
        del arg971_1
        # Topologically Sorted Source Nodes: [x_1458, x_1459, x_1460], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf569 = extern_kernels.convolution(buf568, arg972_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf569, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg972_1
        del buf568
        # Topologically Sorted Source Nodes: [x_1461], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, arg973_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg973_1
        del buf569
        # Topologically Sorted Source Nodes: [x_1463, x_1464], Original ATen: [aten.relu, aten.convolution]
        buf572 = extern_kernels.convolution(buf571, arg978_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf572, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg978_1
        del buf571
        # Topologically Sorted Source Nodes: [x_1465], Original ATen: [aten.convolution]
        buf573 = extern_kernels.convolution(buf572, arg979_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf573, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg979_1
        del buf572
        buf574 = buf573; del buf573  # reuse
        # Topologically Sorted Source Nodes: [x_1466, x_1467], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf574, arg980_1, arg981_1, arg982_1, arg983_1, 836352, grid=grid(836352), stream=stream0)
        del arg980_1
        del arg981_1
        del arg982_1
        del arg983_1
        # Topologically Sorted Source Nodes: [x_1466, x_1467, x_1468], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf575 = extern_kernels.convolution(buf574, arg984_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf575, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg984_1
        del buf574
        # Topologically Sorted Source Nodes: [x_1469], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, arg985_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg985_1
        buf577 = buf570; del buf570  # reuse
        buf578 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [x_1462, x_1470, x_comb_iter_127, x_1471], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75.run(buf577, arg974_1, arg975_1, arg976_1, arg977_1, buf576, arg986_1, arg987_1, arg988_1, arg989_1, buf578, 836352, grid=grid(836352), stream=stream0)
        del arg974_1
        del arg975_1
        del arg976_1
        del arg977_1
        del arg986_1
        del arg987_1
        del arg988_1
        del arg989_1
        del buf576
        # Topologically Sorted Source Nodes: [x_1471, x_1472], Original ATen: [aten.relu, aten.convolution]
        buf579 = extern_kernels.convolution(buf578, arg990_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf579, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg990_1
        del buf578
        # Topologically Sorted Source Nodes: [x_1473], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf579, arg991_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg991_1
        del buf579
        buf581 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [x_1474, x_1475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf581, arg992_1, arg993_1, arg994_1, arg995_1, 836352, grid=grid(836352), stream=stream0)
        del arg992_1
        del arg993_1
        del arg994_1
        del arg995_1
        # Topologically Sorted Source Nodes: [x_1474, x_1475, x_1476], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf582 = extern_kernels.convolution(buf581, arg996_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf582, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg996_1
        del buf581
        # Topologically Sorted Source Nodes: [x_1477], Original ATen: [aten.convolution]
        buf583 = extern_kernels.convolution(buf582, arg997_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf583, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg997_1
        del buf582
        # Topologically Sorted Source Nodes: [x_1479, x_1480], Original ATen: [aten.relu, aten.convolution]
        buf585 = extern_kernels.convolution(buf584, arg1002_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf585, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1002_1
        del buf584
        # Topologically Sorted Source Nodes: [x_1481], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, arg1003_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1003_1
        del buf585
        buf587 = buf586; del buf586  # reuse
        # Topologically Sorted Source Nodes: [x_1482, x_1483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf587, arg1004_1, arg1005_1, arg1006_1, arg1007_1, 836352, grid=grid(836352), stream=stream0)
        del arg1004_1
        del arg1005_1
        del arg1006_1
        del arg1007_1
        # Topologically Sorted Source Nodes: [x_1482, x_1483, x_1484], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf588 = extern_kernels.convolution(buf587, arg1008_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf588, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1008_1
        del buf587
        # Topologically Sorted Source Nodes: [x_1485], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, arg1009_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1009_1
        del buf588
        buf591 = buf549; del buf549  # reuse
        buf596 = buf544; del buf544  # reuse
        # Topologically Sorted Source Nodes: [x_out_25, x_1490, x_1541], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_82.run(buf558, arg950_1, arg951_1, arg952_1, arg953_1, buf502, buf564, arg962_1, arg963_1, arg964_1, arg965_1, buf547, buf577, buf583, arg998_1, arg999_1, arg1000_1, arg1001_1, buf548, buf589, arg1010_1, arg1011_1, arg1012_1, arg1013_1, buf546, buf591, buf596, 34560, 121, grid=grid(34560, 121), stream=stream0)
        del arg1000_1
        del arg1001_1
        del arg1010_1
        del arg1011_1
        del arg1012_1
        del arg1013_1
        del arg950_1
        del arg951_1
        del arg952_1
        del arg953_1
        del arg962_1
        del arg963_1
        del arg964_1
        del arg965_1
        del arg998_1
        del arg999_1
        del buf502
        # Topologically Sorted Source Nodes: [x_1490, x_1491], Original ATen: [aten.relu, aten.convolution]
        buf592 = extern_kernels.convolution(buf591, arg1019_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1019_1
        del buf591
        buf593 = buf592; del buf592  # reuse
        # Topologically Sorted Source Nodes: [x_1492], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_78.run(buf593, arg1020_1, arg1021_1, arg1022_1, arg1023_1, 836352, grid=grid(836352), stream=stream0)
        del arg1020_1
        del arg1021_1
        del arg1022_1
        del arg1023_1
        buf594 = buf589; del buf589  # reuse
        buf595 = buf583; del buf583  # reuse
        buf606 = buf577; del buf577  # reuse
        buf612 = buf564; del buf564  # reuse
        buf618 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_26, x_comb_iter_3_right_26, x_1501, x_1509, x_1517], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_79.run(buf593, buf594, buf595, buf606, buf612, buf618, 836352, grid=grid(836352), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1541, x_1542], Original ATen: [aten.relu, aten.convolution]
        buf597 = extern_kernels.convolution(buf596, arg1096_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf597, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1096_1
        buf598 = buf597; del buf597  # reuse
        buf643 = buf548; del buf548  # reuse
        buf674 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [x_1543, x_1547, x_1587], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_80.run(buf598, arg1097_1, arg1098_1, arg1099_1, arg1100_1, buf643, buf674, 836352, grid=grid(836352), stream=stream0)
        del arg1097_1
        del arg1098_1
        del arg1099_1
        del arg1100_1
        buf599 = buf546; del buf546  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_0_right_25], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_81.run(buf598, buf599, 836352, grid=grid(836352), stream=stream0)
        del buf598
        # Topologically Sorted Source Nodes: [x_1493, x_1494], Original ATen: [aten.relu, aten.convolution]
        buf601 = extern_kernels.convolution(buf600, arg1024_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf601, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1024_1
        del buf600
        # Topologically Sorted Source Nodes: [x_1495], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, arg1025_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1025_1
        del buf601
        buf603 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [x_1496, x_1497], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf603, arg1026_1, arg1027_1, arg1028_1, arg1029_1, 836352, grid=grid(836352), stream=stream0)
        del arg1026_1
        del arg1027_1
        del arg1028_1
        del arg1029_1
        # Topologically Sorted Source Nodes: [x_1496, x_1497, x_1498], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf604 = extern_kernels.convolution(buf603, arg1030_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf604, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1030_1
        del buf603
        # Topologically Sorted Source Nodes: [x_1499], Original ATen: [aten.convolution]
        buf605 = extern_kernels.convolution(buf604, arg1031_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf605, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1031_1
        del buf604
        # Topologically Sorted Source Nodes: [x_1501, x_1502], Original ATen: [aten.relu, aten.convolution]
        buf607 = extern_kernels.convolution(buf606, arg1036_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf607, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1036_1
        del buf606
        # Topologically Sorted Source Nodes: [x_1503], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(buf607, arg1037_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1037_1
        del buf607
        buf609 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [x_1504, x_1505], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf609, arg1038_1, arg1039_1, arg1040_1, arg1041_1, 836352, grid=grid(836352), stream=stream0)
        del arg1038_1
        del arg1039_1
        del arg1040_1
        del arg1041_1
        # Topologically Sorted Source Nodes: [x_1504, x_1505, x_1506], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf610 = extern_kernels.convolution(buf609, arg1042_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf610, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1042_1
        del buf609
        # Topologically Sorted Source Nodes: [x_1507], Original ATen: [aten.convolution]
        buf611 = extern_kernels.convolution(buf610, arg1043_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf611, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1043_1
        del buf610
        # Topologically Sorted Source Nodes: [x_1509, x_1510], Original ATen: [aten.relu, aten.convolution]
        buf613 = extern_kernels.convolution(buf612, arg1048_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf613, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1048_1
        del buf612
        # Topologically Sorted Source Nodes: [x_1511], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf613, arg1049_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1049_1
        del buf613
        buf615 = buf614; del buf614  # reuse
        # Topologically Sorted Source Nodes: [x_1512, x_1513], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf615, arg1050_1, arg1051_1, arg1052_1, arg1053_1, 836352, grid=grid(836352), stream=stream0)
        del arg1050_1
        del arg1051_1
        del arg1052_1
        del arg1053_1
        # Topologically Sorted Source Nodes: [x_1512, x_1513, x_1514], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf616 = extern_kernels.convolution(buf615, arg1054_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf616, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1054_1
        del buf615
        # Topologically Sorted Source Nodes: [x_1515], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, arg1055_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1055_1
        del buf616
        # Topologically Sorted Source Nodes: [x_1517, x_1518], Original ATen: [aten.relu, aten.convolution]
        buf619 = extern_kernels.convolution(buf618, arg1060_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf619, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1060_1
        del buf618
        # Topologically Sorted Source Nodes: [x_1519], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, arg1061_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1061_1
        del buf619
        buf621 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [x_1520, x_1521], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf621, arg1062_1, arg1063_1, arg1064_1, arg1065_1, 836352, grid=grid(836352), stream=stream0)
        del arg1062_1
        del arg1063_1
        del arg1064_1
        del arg1065_1
        # Topologically Sorted Source Nodes: [x_1520, x_1521, x_1522], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf622 = extern_kernels.convolution(buf621, arg1066_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf622, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1066_1
        del buf621
        # Topologically Sorted Source Nodes: [x_1523], Original ATen: [aten.convolution]
        buf623 = extern_kernels.convolution(buf622, arg1067_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf623, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1067_1
        buf624 = buf617; del buf617  # reuse
        buf625 = buf622; del buf622  # reuse
        # Topologically Sorted Source Nodes: [x_1516, x_1524, x_comb_iter_132, x_1525], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75.run(buf624, arg1056_1, arg1057_1, arg1058_1, arg1059_1, buf623, arg1068_1, arg1069_1, arg1070_1, arg1071_1, buf625, 836352, grid=grid(836352), stream=stream0)
        del arg1056_1
        del arg1057_1
        del arg1058_1
        del arg1059_1
        del arg1068_1
        del arg1069_1
        del arg1070_1
        del arg1071_1
        del buf623
        # Topologically Sorted Source Nodes: [x_1525, x_1526], Original ATen: [aten.relu, aten.convolution]
        buf626 = extern_kernels.convolution(buf625, arg1072_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf626, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1072_1
        del buf625
        # Topologically Sorted Source Nodes: [x_1527], Original ATen: [aten.convolution]
        buf627 = extern_kernels.convolution(buf626, arg1073_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf627, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1073_1
        del buf626
        buf628 = buf627; del buf627  # reuse
        # Topologically Sorted Source Nodes: [x_1528, x_1529], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf628, arg1074_1, arg1075_1, arg1076_1, arg1077_1, 836352, grid=grid(836352), stream=stream0)
        del arg1074_1
        del arg1075_1
        del arg1076_1
        del arg1077_1
        # Topologically Sorted Source Nodes: [x_1528, x_1529, x_1530], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf629 = extern_kernels.convolution(buf628, arg1078_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf629, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1078_1
        del buf628
        # Topologically Sorted Source Nodes: [x_1531], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, arg1079_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1079_1
        del buf629
        # Topologically Sorted Source Nodes: [x_1533, x_1534], Original ATen: [aten.relu, aten.convolution]
        buf632 = extern_kernels.convolution(buf631, arg1084_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf632, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1084_1
        del buf631
        # Topologically Sorted Source Nodes: [x_1535], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, arg1085_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1085_1
        del buf632
        buf634 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [x_1536, x_1537], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf634, arg1086_1, arg1087_1, arg1088_1, arg1089_1, 836352, grid=grid(836352), stream=stream0)
        del arg1086_1
        del arg1087_1
        del arg1088_1
        del arg1089_1
        # Topologically Sorted Source Nodes: [x_1536, x_1537, x_1538], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf635 = extern_kernels.convolution(buf634, arg1090_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf635, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1090_1
        del buf634
        # Topologically Sorted Source Nodes: [x_1539], Original ATen: [aten.convolution]
        buf636 = extern_kernels.convolution(buf635, arg1091_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf636, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1091_1
        del buf635
        buf638 = buf596; del buf596  # reuse
        # Topologically Sorted Source Nodes: [x_out_26, x_1544], Original ATen: [aten.cat, aten.relu]
        triton_poi_fused_cat_relu_83.run(buf605, arg1032_1, arg1033_1, arg1034_1, arg1035_1, buf552, buf611, arg1044_1, arg1045_1, arg1046_1, arg1047_1, buf594, buf624, buf630, arg1080_1, arg1081_1, arg1082_1, arg1083_1, buf595, buf636, arg1092_1, arg1093_1, arg1094_1, arg1095_1, buf593, buf638, 34560, 121, grid=grid(34560, 121), stream=stream0)
        del arg1032_1
        del arg1033_1
        del arg1034_1
        del arg1035_1
        del arg1044_1
        del arg1045_1
        del arg1046_1
        del arg1047_1
        del arg1080_1
        del arg1081_1
        del arg1082_1
        del arg1083_1
        del arg1092_1
        del arg1093_1
        del arg1094_1
        del arg1095_1
        del buf552
        del buf593
        del buf594
        del buf595
        # Topologically Sorted Source Nodes: [x_1544, x_1545], Original ATen: [aten.relu, aten.convolution]
        buf639 = extern_kernels.convolution(buf638, arg1101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf639, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1101_1
        del buf638
        buf640 = buf639; del buf639  # reuse
        # Topologically Sorted Source Nodes: [x_1546], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_78.run(buf640, arg1102_1, arg1103_1, arg1104_1, arg1105_1, 836352, grid=grid(836352), stream=stream0)
        del arg1102_1
        del arg1103_1
        del arg1104_1
        del arg1105_1
        buf641 = buf636; del buf636  # reuse
        buf642 = buf630; del buf630  # reuse
        buf649 = buf624; del buf624  # reuse
        buf655 = buf611; del buf611  # reuse
        buf661 = buf605; del buf605  # reuse
        # Topologically Sorted Source Nodes: [x_comb_iter_1_right_27, x_comb_iter_3_right_27, x_1555, x_1563, x_1571], Original ATen: [aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_max_pool2d_with_indices_relu_79.run(buf640, buf641, buf642, buf649, buf655, buf661, 836352, grid=grid(836352), stream=stream0)
        # Topologically Sorted Source Nodes: [x_1547, x_1548], Original ATen: [aten.relu, aten.convolution]
        buf644 = extern_kernels.convolution(buf643, arg1106_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf644, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1106_1
        del buf643
        # Topologically Sorted Source Nodes: [x_1549], Original ATen: [aten.convolution]
        buf645 = extern_kernels.convolution(buf644, arg1107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1107_1
        del buf644
        buf646 = buf645; del buf645  # reuse
        # Topologically Sorted Source Nodes: [x_1550, x_1551], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf646, arg1108_1, arg1109_1, arg1110_1, arg1111_1, 836352, grid=grid(836352), stream=stream0)
        del arg1108_1
        del arg1109_1
        del arg1110_1
        del arg1111_1
        # Topologically Sorted Source Nodes: [x_1550, x_1551, x_1552], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf647 = extern_kernels.convolution(buf646, arg1112_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf647, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1112_1
        del buf646
        # Topologically Sorted Source Nodes: [x_1553], Original ATen: [aten.convolution]
        buf648 = extern_kernels.convolution(buf647, arg1113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf648, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1113_1
        del buf647
        # Topologically Sorted Source Nodes: [x_1555, x_1556], Original ATen: [aten.relu, aten.convolution]
        buf650 = extern_kernels.convolution(buf649, arg1118_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf650, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1118_1
        del buf649
        # Topologically Sorted Source Nodes: [x_1557], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf650, arg1119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf651, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1119_1
        del buf650
        buf652 = buf651; del buf651  # reuse
        # Topologically Sorted Source Nodes: [x_1558, x_1559], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf652, arg1120_1, arg1121_1, arg1122_1, arg1123_1, 836352, grid=grid(836352), stream=stream0)
        del arg1120_1
        del arg1121_1
        del arg1122_1
        del arg1123_1
        # Topologically Sorted Source Nodes: [x_1558, x_1559, x_1560], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf653 = extern_kernels.convolution(buf652, arg1124_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf653, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1124_1
        del buf652
        # Topologically Sorted Source Nodes: [x_1561], Original ATen: [aten.convolution]
        buf654 = extern_kernels.convolution(buf653, arg1125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1125_1
        del buf653
        # Topologically Sorted Source Nodes: [x_1563, x_1564], Original ATen: [aten.relu, aten.convolution]
        buf656 = extern_kernels.convolution(buf655, arg1130_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf656, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1130_1
        del buf655
        # Topologically Sorted Source Nodes: [x_1565], Original ATen: [aten.convolution]
        buf657 = extern_kernels.convolution(buf656, arg1131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1131_1
        del buf656
        buf658 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [x_1566, x_1567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf658, arg1132_1, arg1133_1, arg1134_1, arg1135_1, 836352, grid=grid(836352), stream=stream0)
        del arg1132_1
        del arg1133_1
        del arg1134_1
        del arg1135_1
        # Topologically Sorted Source Nodes: [x_1566, x_1567, x_1568], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf659 = extern_kernels.convolution(buf658, arg1136_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf659, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1136_1
        del buf658
        # Topologically Sorted Source Nodes: [x_1569], Original ATen: [aten.convolution]
        buf660 = extern_kernels.convolution(buf659, arg1137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf660, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1137_1
        del buf659
        # Topologically Sorted Source Nodes: [x_1571, x_1572], Original ATen: [aten.relu, aten.convolution]
        buf662 = extern_kernels.convolution(buf661, arg1142_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf662, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1142_1
        del buf661
        # Topologically Sorted Source Nodes: [x_1573], Original ATen: [aten.convolution]
        buf663 = extern_kernels.convolution(buf662, arg1143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf663, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1143_1
        del buf662
        buf664 = buf663; del buf663  # reuse
        # Topologically Sorted Source Nodes: [x_1574, x_1575], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf664, arg1144_1, arg1145_1, arg1146_1, arg1147_1, 836352, grid=grid(836352), stream=stream0)
        del arg1144_1
        del arg1145_1
        del arg1146_1
        del arg1147_1
        # Topologically Sorted Source Nodes: [x_1574, x_1575, x_1576], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf665 = extern_kernels.convolution(buf664, arg1148_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf665, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1148_1
        del buf664
        # Topologically Sorted Source Nodes: [x_1577], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, arg1149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1149_1
        buf667 = buf660; del buf660  # reuse
        buf668 = buf665; del buf665  # reuse
        # Topologically Sorted Source Nodes: [x_1570, x_1578, x_comb_iter_137, x_1579], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_75.run(buf667, arg1138_1, arg1139_1, arg1140_1, arg1141_1, buf666, arg1150_1, arg1151_1, arg1152_1, arg1153_1, buf668, 836352, grid=grid(836352), stream=stream0)
        del arg1138_1
        del arg1139_1
        del arg1140_1
        del arg1141_1
        del arg1150_1
        del arg1151_1
        del arg1152_1
        del arg1153_1
        del buf666
        # Topologically Sorted Source Nodes: [x_1579, x_1580], Original ATen: [aten.relu, aten.convolution]
        buf669 = extern_kernels.convolution(buf668, arg1154_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf669, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1154_1
        del buf668
        # Topologically Sorted Source Nodes: [x_1581], Original ATen: [aten.convolution]
        buf670 = extern_kernels.convolution(buf669, arg1155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf670, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1155_1
        del buf669
        buf671 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [x_1582, x_1583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf671, arg1156_1, arg1157_1, arg1158_1, arg1159_1, 836352, grid=grid(836352), stream=stream0)
        del arg1156_1
        del arg1157_1
        del arg1158_1
        del arg1159_1
        # Topologically Sorted Source Nodes: [x_1582, x_1583, x_1584], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf672 = extern_kernels.convolution(buf671, arg1160_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf672, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1160_1
        del buf671
        # Topologically Sorted Source Nodes: [x_1585], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, arg1161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1161_1
        del buf672
        # Topologically Sorted Source Nodes: [x_1587, x_1588], Original ATen: [aten.relu, aten.convolution]
        buf675 = extern_kernels.convolution(buf674, arg1166_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf675, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1166_1
        del buf674
        # Topologically Sorted Source Nodes: [x_1589], Original ATen: [aten.convolution]
        buf676 = extern_kernels.convolution(buf675, arg1167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf676, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1167_1
        del buf675
        buf677 = buf676; del buf676  # reuse
        # Topologically Sorted Source Nodes: [x_1590, x_1591], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf677, arg1168_1, arg1169_1, arg1170_1, arg1171_1, 836352, grid=grid(836352), stream=stream0)
        del arg1168_1
        del arg1169_1
        del arg1170_1
        del arg1171_1
        # Topologically Sorted Source Nodes: [x_1590, x_1591, x_1592], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf678 = extern_kernels.convolution(buf677, arg1172_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=864, bias=None)
        assert_size_stride(buf678, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1172_1
        del buf677
        # Topologically Sorted Source Nodes: [x_1593], Original ATen: [aten.convolution]
        buf679 = extern_kernels.convolution(buf678, arg1173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf679, (8, 864, 11, 11), (104544, 1, 9504, 864))
        del arg1173_1
        del buf678
        buf682 = empty_strided_cuda((8, 4320, 1, 1), (4320, 1, 34560, 34560), torch.float32)
        # Topologically Sorted Source Nodes: [x_out_27, x_1595, x_1596], Original ATen: [aten.cat, aten.relu, aten.mean]
        triton_red_fused_cat_mean_relu_84.run(buf648, arg1114_1, arg1115_1, arg1116_1, arg1117_1, buf599, buf654, arg1126_1, arg1127_1, arg1128_1, arg1129_1, buf641, buf667, buf673, arg1162_1, arg1163_1, arg1164_1, arg1165_1, buf642, buf679, arg1174_1, arg1175_1, arg1176_1, arg1177_1, buf640, buf682, 34560, 121, grid=grid(34560), stream=stream0)
        del arg1114_1
        del arg1115_1
        del arg1116_1
        del arg1117_1
        del arg1126_1
        del arg1127_1
        del arg1128_1
        del arg1129_1
        del arg1162_1
        del arg1163_1
        del arg1164_1
        del arg1165_1
        del arg1174_1
        del arg1175_1
        del arg1176_1
        del arg1177_1
        del buf599
        del buf640
        del buf641
        del buf642
        del buf648
        del buf654
        del buf667
        del buf673
        del buf679
        buf683 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1599], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg1179_1, reinterpret_tensor(buf682, (8, 4320), (4320, 1), 0), reinterpret_tensor(arg1178_1, (4320, 1000), (1, 4320), 0), alpha=1, beta=1, out=buf683)
        del arg1178_1
        del arg1179_1
        del buf682
    return (buf683, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((96, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 331, 331), (328683, 109561, 331, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((96, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((54, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((54, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((54, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((54, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((54, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((54, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((54, 54, 1, 1), (54, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((54, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((54, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((108, 270, 1, 1), (270, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((108, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((108, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((108, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((108, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((108, 108, 1, 1), (108, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((108, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((108, 270, 1, 1), (270, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((108, 270, 1, 1), (270, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((216, 540, 1, 1), (540, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((216, 540, 1, 1), (540, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((216, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((216, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((216, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((216, 216, 1, 1), (216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((432, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((432, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((216, 1080, 1, 1), (1080, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((432, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((432, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg806_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg809_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg812_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg815_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg818_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg821_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg824_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg827_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg830_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg833_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg836_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((432, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg839_1 = rand_strided((432, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg842_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((864, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg845_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg848_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((864, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg851_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg853_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg854_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg855_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg856_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg857_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg858_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg859_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg860_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg861_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg862_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg863_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg864_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg865_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg866_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg867_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg868_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg869_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg870_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg871_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg872_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg873_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg874_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg875_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg876_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg877_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg878_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg879_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg880_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg881_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg882_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg883_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg884_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg885_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg886_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg887_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg888_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg889_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg890_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg891_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg892_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg893_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg894_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg895_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg896_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg897_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg898_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg899_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg900_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg901_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg902_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg903_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg904_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg905_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg906_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg907_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg908_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg909_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg910_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg911_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg912_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg913_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg914_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg915_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg916_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg917_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg918_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg919_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg920_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg921_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg922_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg923_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg924_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg925_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg926_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg927_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg928_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg929_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg930_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg931_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg932_1 = rand_strided((432, 2160, 1, 1), (2160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg933_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg934_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg935_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg936_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg937_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg938_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg939_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg940_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg941_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg942_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg943_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg944_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg945_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg946_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg947_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg948_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg949_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg950_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg951_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg952_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg953_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg954_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg955_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg956_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg957_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg958_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg959_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg960_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg961_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg962_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg963_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg964_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg965_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg966_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg967_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg968_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg969_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg970_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg971_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg972_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg973_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg974_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg975_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg976_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg977_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg978_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg979_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg980_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg981_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg982_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg983_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg984_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg985_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg986_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg987_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg988_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg989_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg990_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg991_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg992_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg993_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg994_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg995_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg996_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg997_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg998_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg999_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1000_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1001_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1002_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1003_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1004_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1005_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1006_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1007_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1008_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1009_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1010_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1011_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1012_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1013_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1014_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1015_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1016_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1017_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1018_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1019_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1020_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1021_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1022_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1023_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1024_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1025_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1026_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1027_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1028_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1029_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1030_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1031_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1032_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1033_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1034_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1035_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1036_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1037_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1038_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1039_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1040_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1041_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1042_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1043_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1044_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1045_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1046_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1047_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1048_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1049_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1050_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1051_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1052_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1053_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1054_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1055_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1056_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1057_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1058_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1059_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1060_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1061_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1062_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1063_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1064_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1065_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1066_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1067_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1068_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1069_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1070_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1071_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1072_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1073_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1074_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1075_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1076_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1077_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1078_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1079_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1080_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1081_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1082_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1083_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1084_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1085_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1086_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1087_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1088_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1089_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1090_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1091_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1092_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1093_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1094_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1095_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1096_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1097_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1098_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1099_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1100_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1101_1 = rand_strided((864, 4320, 1, 1), (4320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1102_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1103_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1104_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1105_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1106_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1107_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1108_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1109_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1110_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1111_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1112_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1113_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1114_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1115_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1116_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1117_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1118_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1119_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1120_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1121_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1122_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1123_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1124_1 = rand_strided((864, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1125_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1126_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1127_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1128_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1129_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1130_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1131_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1132_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1133_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1134_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1135_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1136_1 = rand_strided((864, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg1137_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1138_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1139_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1140_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1141_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1142_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1143_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1144_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1145_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1146_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1147_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1148_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1149_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1150_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1151_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1152_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1153_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1154_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1155_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1156_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1157_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1158_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1159_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1160_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1161_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1162_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1163_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1164_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1165_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1166_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1167_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1168_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1169_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1170_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1171_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1172_1 = rand_strided((864, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1173_1 = rand_strided((864, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1174_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1175_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1176_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1177_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1178_1 = rand_strided((1000, 4320), (4320, 1), device='cuda:0', dtype=torch.float32)
    arg1179_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pnasnet5large', benchmark_compiled_module)
