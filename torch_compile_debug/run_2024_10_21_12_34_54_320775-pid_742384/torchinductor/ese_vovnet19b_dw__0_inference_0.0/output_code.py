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


# kernel path: /tmp/torchinductor_sahanp/3f/c3fncxgde2thpvzyp2u4w2psqtj2hypvbt33orumjzg67j6yuyoc.py
# Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_98 => convolution_41
# Graph fragment:
#   %convolution_41 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 82944
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
    tmp0 = tl.load(in_ptr0 + (x2 + (82944*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (248832*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g4/cg44vkswcejqn6yhzia3bvq33k4wssu3n2hwq3ducaq2xqudsjyz.py
# Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_98 => convolution_41
# Graph fragment:
#   %convolution_41 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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


# kernel path: /tmp/torchinductor_sahanp/te/cte5fohwjqx7snhuh6rn6e2cbt6jtohypxonym4bouq42t245kn6.py
# Topologically Sorted Source Nodes: [x_99, x_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_100 => relu_23
#   x_99 => add_51, mul_74, mul_75, sub_23
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_185), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_74, %unsqueeze_189), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_75, %unsqueeze_191), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_51,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ga/cgavyxvyi3ufdyapeirtedt2ftpsavmbnzlkfpirpzmun47wv3yz.py
# Topologically Sorted Source Nodes: [x_107, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_107 => add_55, mul_80, mul_81, sub_25
#   x_108 => relu_25
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_201), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_80, %unsqueeze_205), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %unsqueeze_207), kwargs = {})
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_55,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3t/c3tsukdtg5tk3ydrqpynbusc7fqykh6diedr6zlkmytbo7iguos2.py
# Topologically Sorted Source Nodes: [x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_110 => add_57, mul_83, mul_84, sub_26
#   x_111 => relu_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_209), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %unsqueeze_213), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zv/czvasgjnfbkz5yqghjdyx4n5tkadmislcase5pdadio53oh7uty4.py
# Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_124 => cat_4
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_25, %relu_27, %relu_28, %relu_29], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18579456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 448
    x1 = (xindex // 448)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((64*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((128*x1) + ((-64) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 320, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((128*x1) + ((-192) + x0)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 448, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + ((128*x1) + ((-320) + x0)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-320) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr5 + ((-320) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-320) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr7 + ((-320) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp16, tmp36, tmp37)
    tmp39 = tl.where(tmp14, tmp15, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cy/ccylatgv7kpcoarv5uvqpojg6jd42im7rrmlmdjputq6uljwpjsw.py
# Topologically Sorted Source Nodes: [x_126, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_126 => add_65, mul_95, mul_96, sub_30
#   x_127 => relu_30
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_241), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %unsqueeze_245), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %unsqueeze_247), kwargs = {})
#   %relu_30 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kr/ckr56dqjkwsczoacxityekq357ftpjvnli3ki3345uhpwf4xoi7u.py
# Topologically Sorted Source Nodes: [x_se_8], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_8 => mean_5
# Graph fragment:
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_30, [2, 3], True), kwargs = {})
triton_red_fused_mean_7 = async_compile.triton('triton_red_fused_mean_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_7(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 83968
    rnumel = 127
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256) % 41
    x0 = xindex % 256
    x2 = (xindex // 10496)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (127*x1)
        tmp1 = tl.full([1, 1], 5184, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*((r3 + (127*x1)) % 5184)) + (1327104*x2)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uq/cuq5cbdo7ze7mkaawa3nqfhh5xxtag522rbbvmsb67hywcbdgwlz.py
# Topologically Sorted Source Nodes: [x_se_8], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_8 => mean_5
# Graph fragment:
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_30, [2, 3], True), kwargs = {})
triton_per_fused_mean_8 = async_compile.triton('triton_per_fused_mean_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_8(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 41
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (10496*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 5184.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yr/cyreqwpf2flbni4bicqsuhkkszwv3sd6j4zhpwhiigik2wmho6nv.py
# Topologically Sorted Source Nodes: [x_se_8, x_se_9, hardsigmoid_4, x_128], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_4 => add_66, clamp_max_4, clamp_min_4, div_4
#   x_128 => mul_97
#   x_se_8 => mean_5
#   x_se_9 => convolution_54
# Graph fragment:
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_30, [2, 3], True), kwargs = {})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_5, %arg46_1, %arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_54, 3), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_66, 0), kwargs = {})
#   %clamp_max_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_4, 6), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_30, %div_4), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_9 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_9(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 1327104)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/be/cbe7jfr2czgs4me6u26axz2bku7pcxfp4gkzdp6jruszbrc2psek.py
# Topologically Sorted Source Nodes: [x_se_8, x_se_9, hardsigmoid_4, x_128, x_129], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hardsigmoid_4 => add_66, clamp_max_4, clamp_min_4, div_4
#   x_128 => mul_97
#   x_129 => _low_memory_max_pool2d_with_offsets_3
#   x_se_8 => mean_5
#   x_se_9 => convolution_54
# Graph fragment:
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_30, [2, 3], True), kwargs = {})
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_5, %arg46_1, %arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_54, 3), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_66, 0), kwargs = {})
#   %clamp_max_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_4, 6), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_30, %div_4), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%mul_97, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_10 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 9216) % 36
    x1 = (xindex // 256) % 36
    x0 = xindex % 256
    x5 = (xindex // 9216)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 72, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (512*x1) + (36864*x5)), tmp10, other=float("-inf"))
    tmp12 = 1 + (2*x1)
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (36864*x5)), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 2 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (512 + x0 + (512*x1) + (36864*x5)), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 1 + (2*x2)
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (18432 + x0 + (512*x1) + (36864*x5)), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (18688 + x0 + (512*x1) + (36864*x5)), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (18944 + x0 + (512*x1) + (36864*x5)), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 2 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (36864 + x0 + (512*x1) + (36864*x5)), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (37120 + x0 + (512*x1) + (36864*x5)), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (37376 + x0 + (512*x1) + (36864*x5)), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/j2/cj22rih5gch7ysluykltse5bz7syo5kvcrjsctt4rwr4atolxmyy.py
# Topologically Sorted Source Nodes: [x_131, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_131 => add_68, mul_100, mul_99, sub_31
#   x_132 => relu_31
# Graph fragment:
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_249), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_99, %unsqueeze_253), kwargs = {})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_100, %unsqueeze_255), kwargs = {})
#   %relu_31 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_68,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1658880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 160
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5z/c5znamcwi7ty72bmxziaohwnnu4idj3zgztjh7ukwnbgov4sgyml.py
# Topologically Sorted Source Nodes: [x_145], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_145 => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_6, %relu_32, %relu_33, %relu_34], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7630848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 736
    x1 = (xindex // 736)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((256*x1) + x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 416, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((160*x1) + ((-256) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 576, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((160*x1) + ((-416) + x0)), tmp14, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 736, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + ((160*x1) + ((-576) + x0)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-576) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr5 + ((-576) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-576) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr7 + ((-576) + x0), tmp16, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp16, tmp36, tmp37)
    tmp39 = tl.where(tmp14, tmp15, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vw/cvwvn3hv6dahtoo5ege6ptbr3q45unyk7i54ke25olpc76wjljxj.py
# Topologically Sorted Source Nodes: [x_147, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_147 => add_76, mul_111, mul_112, sub_35
#   x_148 => relu_35
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_281), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_111, %unsqueeze_285), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %unsqueeze_287), kwargs = {})
#   %relu_35 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_76,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ob/cobh2nyauzjtsodphvvzyilasvlk6abzx6e4ghzcnnxthx2effvw.py
# Topologically Sorted Source Nodes: [x_se_10], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_10 => mean_6
# Graph fragment:
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_35, [2, 3], True), kwargs = {})
triton_red_fused_mean_14 = async_compile.triton('triton_red_fused_mean_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_14(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 45056
    rnumel = 118
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 512) % 11
    x0 = xindex % 512
    x2 = (xindex // 5632)
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = r3 + (118*x1)
        tmp1 = tl.full([1, 1], 1296, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (512*((r3 + (118*x1)) % 1296)) + (663552*x2)), rmask & tmp2, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xw/cxwvqzocoi62speolmabob2boxhbjvpykub3f5t67p4aezpldqbq.py
# Topologically Sorted Source Nodes: [x_se_10], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_10 => mean_6
# Graph fragment:
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_35, [2, 3], True), kwargs = {})
triton_per_fused_mean_15 = async_compile.triton('triton_per_fused_mean_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_15(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 11
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (5632*x1)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 1296.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4m/c4m3zlh23tzzfzeagz677gpdonymd5qx5l6oqkzy7324pf4uczmp.py
# Topologically Sorted Source Nodes: [x_se_10, x_se_11, hardsigmoid_5, x_149], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_5 => add_77, clamp_max_5, clamp_min_5, div_5
#   x_149 => mul_113
#   x_se_10 => mean_6
#   x_se_11 => convolution_63
# Graph fragment:
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_35, [2, 3], True), kwargs = {})
#   %convolution_63 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_6, %arg76_1, %arg77_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_63, 3), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_77, 0), kwargs = {})
#   %clamp_max_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 6), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_5, 6), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_35, %div_5), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_16 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_16(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 663552)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/et/cetyhvtcvzgug4ikz3wk2x3qakqx36ep4ingmr3dcdu6lf5xfig4.py
# Topologically Sorted Source Nodes: [x_se_10, x_se_11, hardsigmoid_5, x_149, x_150], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hardsigmoid_5 => add_77, clamp_max_5, clamp_min_5, div_5
#   x_149 => mul_113
#   x_150 => _low_memory_max_pool2d_with_offsets_4
#   x_se_10 => mean_6
#   x_se_11 => convolution_63
# Graph fragment:
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_35, [2, 3], True), kwargs = {})
#   %convolution_63 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_6, %arg76_1, %arg77_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_77 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_63, 3), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_77, 0), kwargs = {})
#   %clamp_max_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 6), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_5, 6), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_35, %div_5), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_4 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%mul_113, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 9216) % 18
    x1 = (xindex // 512) % 18
    x0 = xindex % 512
    x5 = (xindex // 9216)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 36, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (1024*x1) + (36864*x5)), tmp10, other=float("-inf"))
    tmp12 = 1 + (2*x1)
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (512 + x0 + (1024*x1) + (36864*x5)), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 2 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (1024 + x0 + (1024*x1) + (36864*x5)), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 1 + (2*x2)
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (18432 + x0 + (1024*x1) + (36864*x5)), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (18944 + x0 + (1024*x1) + (36864*x5)), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (19456 + x0 + (1024*x1) + (36864*x5)), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 2 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (36864 + x0 + (1024*x1) + (36864*x5)), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (37376 + x0 + (1024*x1) + (36864*x5)), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (37888 + x0 + (1024*x1) + (36864*x5)), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s7/cs7naw6tbgdfktcbxkmb5vdrzzr3vmbzaatmx54urokigq5yz5p2.py
# Topologically Sorted Source Nodes: [x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_152 => add_79, mul_115, mul_116, sub_36
#   x_153 => relu_36
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_64, %unsqueeze_289), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_293), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_295), kwargs = {})
#   %relu_36 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_79,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 497664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 192
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/di/cdi46a4644nzboja5sukf4nmbp3oi53tfbjh2vewqzecval4q3of.py
# Topologically Sorted Source Nodes: [x_166], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_166 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_8, %relu_37, %relu_38, %relu_39], 1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2820096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1088
    x1 = (xindex // 1088)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((512*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 704, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((192*x1) + ((-512) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 896, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((192*x1) + ((-704) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 1088, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + ((192*x1) + ((-896) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-896) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr5 + ((-896) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-896) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr7 + ((-896) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp16, tmp36, tmp37)
    tmp39 = tl.where(tmp14, tmp15, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jc/cjcbqyw7oopip6hwiu44twisuex2ga4jx4b4evjxoqxswkaso3ue.py
# Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_168 => add_87, mul_127, mul_128, sub_40
#   x_169 => relu_40
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_321), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_127, %unsqueeze_325), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_128, %unsqueeze_327), kwargs = {})
#   %relu_40 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_87,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1990656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/up/cupcah6g5ph4h7doqr7iivudkodracedjyrbgpr3jyltuh66aj7k.py
# Topologically Sorted Source Nodes: [x_se_12], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_12 => mean_7
# Graph fragment:
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_40, [2, 3], True), kwargs = {})
triton_red_fused_mean_21 = async_compile.triton('triton_red_fused_mean_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_21(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 108
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (82944*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bx/cbx6xtbnnglim3vpbhht32paderxpbcjvc3lsu5qoz6ehfwotrg2.py
# Topologically Sorted Source Nodes: [x_se_12], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_12 => mean_7
# Graph fragment:
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_40, [2, 3], True), kwargs = {})
triton_per_fused_mean_22 = async_compile.triton('triton_per_fused_mean_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[8192, 4],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_22(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 3
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 768
    x1 = (xindex // 768)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (2304*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 324.0
    tmp6 = tmp4 / tmp5
    tl.store(out_ptr1 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vi/cvium5c5rip6hs6zvotm5z7eaz22oqq56t2ecjvmfly6lpcklwt7.py
# Topologically Sorted Source Nodes: [x_se_12, x_se_13, hardsigmoid_6, x_170], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_6 => add_88, clamp_max_6, clamp_min_6, div_6
#   x_170 => mul_129
#   x_se_12 => mean_7
#   x_se_13 => convolution_72
# Graph fragment:
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_40, [2, 3], True), kwargs = {})
#   %convolution_72 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_7, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_72, 3), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_88, 0), kwargs = {})
#   %clamp_max_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_6, 6), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_40, %div_6), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_mean_mul_23 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_mean_mul_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_mean_mul_23(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1990656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = xindex % 768
    x2 = (xindex // 248832)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (768*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = 0.16666666666666666
    tmp11 = tmp9 * tmp10
    tmp12 = tmp0 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/s4/cs4ewlpcaetixwcb277avs3zxifpi77dtdh3z4egywy7mwbjsmis.py
# Topologically Sorted Source Nodes: [x_se_12, x_se_13, hardsigmoid_6, x_170, x_171], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   hardsigmoid_6 => add_88, clamp_max_6, clamp_min_6, div_6
#   x_170 => mul_129
#   x_171 => _low_memory_max_pool2d_with_offsets_5
#   x_se_12 => mean_7
#   x_se_13 => convolution_72
# Graph fragment:
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_40, [2, 3], True), kwargs = {})
#   %convolution_72 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_7, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_88 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_72, 3), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_88, 0), kwargs = {})
#   %clamp_max_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_6, 6), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_40, %div_6), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%mul_129, [3, 3], [2, 2], [0, 0], [1, 1], True), kwargs = {})
triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_24 = async_compile.triton('triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 497664
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6912) % 9
    x1 = (xindex // 768) % 9
    x0 = xindex % 768
    x5 = (xindex // 6912)
    x6 = xindex
    tmp0 = 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 18, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (x0 + (1536*x1) + (27648*x5)), tmp10 & xmask, other=float("-inf"))
    tmp12 = 1 + (2*x1)
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + (768 + x0 + (1536*x1) + (27648*x5)), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 2 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + (1536 + x0 + (1536*x1) + (27648*x5)), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 1 + (2*x2)
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + (13824 + x0 + (1536*x1) + (27648*x5)), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (14592 + x0 + (1536*x1) + (27648*x5)), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (15360 + x0 + (1536*x1) + (27648*x5)), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 2 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (27648 + x0 + (1536*x1) + (27648*x5)), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (28416 + x0 + (1536*x1) + (27648*x5)), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (29184 + x0 + (1536*x1) + (27648*x5)), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6), tmp51, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2r/c2rhtofr2tff2mnnjvfigzdt2axoem7ggru35bdx63nfzkrugz76.py
# Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_173 => add_90, mul_131, mul_132, sub_41
#   x_174 => relu_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_73, %unsqueeze_329), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_131, %unsqueeze_333), kwargs = {})
#   %add_90 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_132, %unsqueeze_335), kwargs = {})
#   %relu_41 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_90,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 145152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 224
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/po/cpoddx6xso35h2mkqmbq3mmsr6mrmgk6b2pclqz5fow5arrnjacd.py
# Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_187 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem_10, %relu_42, %relu_43, %relu_44], 1), kwargs = {})
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 933120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1440
    x1 = (xindex // 1440)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((768*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 992, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + ((224*x1) + ((-768) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 1216, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + ((224*x1) + ((-992) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 1440, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + ((224*x1) + ((-1216) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr4 + ((-1216) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr5 + ((-1216) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-1216) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.load(in_ptr7 + ((-1216) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp16, tmp36, tmp37)
    tmp39 = tl.where(tmp14, tmp15, tmp38)
    tmp40 = tl.where(tmp9, tmp10, tmp39)
    tmp41 = tl.where(tmp4, tmp5, tmp40)
    tl.store(out_ptr0 + (x2), tmp41, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/w7/cw7tnujnuj4k2e7uw2i556nrbkenvnmwyt7gudehmqksw5khz55y.py
# Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_189 => add_98, mul_143, mul_144, sub_45
#   x_190 => relu_45
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_80, %unsqueeze_361), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_143, %unsqueeze_365), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, %unsqueeze_367), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_98,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6p/c6p4t3ot3lyez7h2xwmfcu6p34q6al4rqvvgytsgrk5sw5pnw7vt.py
# Topologically Sorted Source Nodes: [x_se_14], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   x_se_14 => mean_8
# Graph fragment:
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_45, [2, 3], True), kwargs = {})
triton_red_fused_mean_28 = async_compile.triton('triton_red_fused_mean_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_28(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 81
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (82944*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 81.0
    tmp5 = tmp2 / tmp4
    tl.store(out_ptr1 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kt/cktfliqdjkny5fqoc7ucuqrtqvt5z5taqgs5zpentynqxnnqp2kr.py
# Topologically Sorted Source Nodes: [x_se_14, x_se_15, hardsigmoid_7, x_191, x_192], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   hardsigmoid_7 => add_99, clamp_max_7, clamp_min_7, div_7
#   x_191 => mul_145
#   x_192 => mean_9
#   x_se_14 => mean_8
#   x_se_15 => convolution_81
# Graph fragment:
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_45, [2, 3], True), kwargs = {})
#   %convolution_81 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_8, %arg136_1, %arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_81, 3), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_99, 0), kwargs = {})
#   %clamp_max_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 6), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_7, 6), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_45, %div_7), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_145, [-1, -2], True), kwargs = {})
triton_red_fused_convolution_hardsigmoid_mean_mul_29 = async_compile.triton('triton_red_fused_convolution_hardsigmoid_mean_mul_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_hardsigmoid_mean_mul_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_hardsigmoid_mean_mul_29(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 81
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x3 = xindex
    tmp1 = tl.load(in_out_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (82944*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = 3.0
        tmp5 = tmp3 + tmp4
        tmp6 = 0.0
        tmp7 = triton_helpers.maximum(tmp5, tmp6)
        tmp8 = 6.0
        tmp9 = triton_helpers.minimum(tmp7, tmp8)
        tmp10 = 0.16666666666666666
        tmp11 = tmp9 * tmp10
        tmp12 = tmp0 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = 81.0
    tmp17 = tmp14 / tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 288, 288), (248832, 82944, 288, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg7_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg24_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg30_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg36_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (256, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, ), (1, ))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (160, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg49_1, (160, ), (1, ))
    assert_size_stride(arg50_1, (160, ), (1, ))
    assert_size_stride(arg51_1, (160, ), (1, ))
    assert_size_stride(arg52_1, (160, ), (1, ))
    assert_size_stride(arg53_1, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg54_1, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg55_1, (160, ), (1, ))
    assert_size_stride(arg56_1, (160, ), (1, ))
    assert_size_stride(arg57_1, (160, ), (1, ))
    assert_size_stride(arg58_1, (160, ), (1, ))
    assert_size_stride(arg59_1, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg60_1, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg61_1, (160, ), (1, ))
    assert_size_stride(arg62_1, (160, ), (1, ))
    assert_size_stride(arg63_1, (160, ), (1, ))
    assert_size_stride(arg64_1, (160, ), (1, ))
    assert_size_stride(arg65_1, (160, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg66_1, (160, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg67_1, (160, ), (1, ))
    assert_size_stride(arg68_1, (160, ), (1, ))
    assert_size_stride(arg69_1, (160, ), (1, ))
    assert_size_stride(arg70_1, (160, ), (1, ))
    assert_size_stride(arg71_1, (512, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (192, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg79_1, (192, ), (1, ))
    assert_size_stride(arg80_1, (192, ), (1, ))
    assert_size_stride(arg81_1, (192, ), (1, ))
    assert_size_stride(arg82_1, (192, ), (1, ))
    assert_size_stride(arg83_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg84_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg85_1, (192, ), (1, ))
    assert_size_stride(arg86_1, (192, ), (1, ))
    assert_size_stride(arg87_1, (192, ), (1, ))
    assert_size_stride(arg88_1, (192, ), (1, ))
    assert_size_stride(arg89_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg90_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg91_1, (192, ), (1, ))
    assert_size_stride(arg92_1, (192, ), (1, ))
    assert_size_stride(arg93_1, (192, ), (1, ))
    assert_size_stride(arg94_1, (192, ), (1, ))
    assert_size_stride(arg95_1, (192, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg96_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg97_1, (192, ), (1, ))
    assert_size_stride(arg98_1, (192, ), (1, ))
    assert_size_stride(arg99_1, (192, ), (1, ))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (768, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (224, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg109_1, (224, ), (1, ))
    assert_size_stride(arg110_1, (224, ), (1, ))
    assert_size_stride(arg111_1, (224, ), (1, ))
    assert_size_stride(arg112_1, (224, ), (1, ))
    assert_size_stride(arg113_1, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg114_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg115_1, (224, ), (1, ))
    assert_size_stride(arg116_1, (224, ), (1, ))
    assert_size_stride(arg117_1, (224, ), (1, ))
    assert_size_stride(arg118_1, (224, ), (1, ))
    assert_size_stride(arg119_1, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg120_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg121_1, (224, ), (1, ))
    assert_size_stride(arg122_1, (224, ), (1, ))
    assert_size_stride(arg123_1, (224, ), (1, ))
    assert_size_stride(arg124_1, (224, ), (1, ))
    assert_size_stride(arg125_1, (224, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg126_1, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg127_1, (224, ), (1, ))
    assert_size_stride(arg128_1, (224, ), (1, ))
    assert_size_stride(arg129_1, (224, ), (1, ))
    assert_size_stride(arg130_1, (224, ), (1, ))
    assert_size_stride(arg131_1, (1024, 1440, 1, 1), (1440, 1, 1, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg139_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 288, 288), (248832, 1, 864, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 82944, grid=grid(24, 82944), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 144, 144), (1327104, 1, 9216, 64))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_99, x_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [x_99, x_100, x_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf4, (8, 64, 144, 144), (1327104, 1, 9216, 64))
        del arg6_1
        del buf3
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg7_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 64, 144, 144), (1327104, 1, 9216, 64))
        del arg7_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_103, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, arg8_1, arg9_1, arg10_1, arg11_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg10_1
        del arg11_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [x_103, x_104, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf7, (8, 64, 72, 72), (331776, 1, 4608, 64))
        del arg12_1
        del buf6
        # Topologically Sorted Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg13_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 64, 72, 72), (331776, 1, 4608, 64))
        del arg13_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_107, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf9, arg14_1, arg15_1, arg16_1, arg17_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg14_1
        del arg15_1
        del arg16_1
        del arg17_1
        # Topologically Sorted Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg18_1
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_110, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf11, arg19_1, arg20_1, arg21_1, arg22_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg19_1
        del arg20_1
        del arg21_1
        del arg22_1
        # Topologically Sorted Source Nodes: [x_110, x_111, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg23_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf12, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg23_1
        del buf11
        # Topologically Sorted Source Nodes: [x_113], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg24_1
        del buf12
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf14, arg25_1, arg26_1, arg27_1, arg28_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg25_1
        del arg26_1
        del arg27_1
        del arg28_1
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg29_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf15, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg29_1
        # Topologically Sorted Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg30_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_118, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf17, arg31_1, arg32_1, arg33_1, arg34_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg35_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf18, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg35_1
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 128, 72, 72), (663552, 1, 9216, 128))
        del arg36_1
        del buf18
        buf20 = empty_strided_cuda((8, 448, 72, 72), (2322432, 1, 32256, 448), torch.float32)
        # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.cat]
        triton_poi_fused_cat_5.run(buf9, buf14, buf17, buf19, arg37_1, arg38_1, arg39_1, arg40_1, buf20, 18579456, grid=grid(18579456), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del buf14
        del buf17
        del buf19
        # Topologically Sorted Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 256, 72, 72), (1327104, 1, 18432, 256))
        del arg41_1
        del buf20
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_126, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf22, arg42_1, arg43_1, arg44_1, arg45_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf23 = empty_strided_cuda((8, 256, 1, 1, 41), (10496, 1, 83968, 83968, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_red_fused_mean_7.run(buf22, buf23, 83968, 127, grid=grid(83968), stream=stream0)
        buf25 = empty_strided_cuda((8, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_8], Original ATen: [aten.mean]
        triton_per_fused_mean_8.run(buf23, buf25, 2048, 41, grid=grid(2048), stream=stream0)
        del buf23
        # Topologically Sorted Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.mean, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg46_1
        del buf25
        buf27 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_se_8, x_se_9, hardsigmoid_4, x_128], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_9.run(buf27, buf26, arg47_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg47_1
        del buf26
        buf28 = reinterpret_tensor(buf9, (8, 256, 36, 36), (331776, 1, 9216, 256), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_se_8, x_se_9, hardsigmoid_4, x_128, x_129], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_10.run(buf27, buf28, 2654208, grid=grid(2654208), stream=stream0)
        del buf27
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 160, 36, 36), (207360, 1, 5760, 160))
        del arg48_1
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [x_131, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf30, arg49_1, arg50_1, arg51_1, arg52_1, 1658880, grid=grid(1658880), stream=stream0)
        del arg49_1
        del arg50_1
        del arg51_1
        del arg52_1
        # Topologically Sorted Source Nodes: [x_131, x_132, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg53_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf31, (8, 160, 36, 36), (207360, 1, 5760, 160))
        del arg53_1
        del buf30
        # Topologically Sorted Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 160, 36, 36), (207360, 1, 5760, 160))
        del arg54_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_135, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf33, arg55_1, arg56_1, arg57_1, arg58_1, 1658880, grid=grid(1658880), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        del arg58_1
        # Topologically Sorted Source Nodes: [x_137], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg59_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf34, (8, 160, 36, 36), (207360, 1, 5760, 160))
        del arg59_1
        # Topologically Sorted Source Nodes: [x_138], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 160, 36, 36), (207360, 1, 5760, 160))
        del arg60_1
        del buf34
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_139, x_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf36, arg61_1, arg62_1, arg63_1, arg64_1, 1658880, grid=grid(1658880), stream=stream0)
        del arg61_1
        del arg62_1
        del arg63_1
        del arg64_1
        # Topologically Sorted Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg65_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=160, bias=None)
        assert_size_stride(buf37, (8, 160, 36, 36), (207360, 1, 5760, 160))
        del arg65_1
        # Topologically Sorted Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 160, 36, 36), (207360, 1, 5760, 160))
        del arg66_1
        del buf37
        buf39 = empty_strided_cuda((8, 736, 36, 36), (953856, 1, 26496, 736), torch.float32)
        # Topologically Sorted Source Nodes: [x_145], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf28, buf33, buf36, buf38, arg67_1, arg68_1, arg69_1, arg70_1, buf39, 7630848, grid=grid(7630848), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del buf28
        del buf33
        del buf36
        del buf38
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg71_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 512, 36, 36), (663552, 1, 18432, 512))
        del arg71_1
        del buf39
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_147, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf41, arg72_1, arg73_1, arg74_1, arg75_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf42 = empty_strided_cuda((8, 512, 1, 1, 11), (5632, 1, 45056, 45056, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_10], Original ATen: [aten.mean]
        triton_red_fused_mean_14.run(buf41, buf42, 45056, 118, grid=grid(45056), stream=stream0)
        buf44 = empty_strided_cuda((8, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_10], Original ATen: [aten.mean]
        triton_per_fused_mean_15.run(buf42, buf44, 4096, 11, grid=grid(4096), stream=stream0)
        del buf42
        # Topologically Sorted Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.mean, aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg76_1
        del buf44
        buf46 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_se_10, x_se_11, hardsigmoid_5, x_149], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_16.run(buf46, buf45, arg77_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg77_1
        del buf45
        buf47 = empty_strided_cuda((8, 512, 18, 18), (165888, 1, 9216, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_10, x_se_11, hardsigmoid_5, x_149, x_150], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_17.run(buf46, buf47, 1327104, grid=grid(1327104), stream=stream0)
        del buf46
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 192, 18, 18), (62208, 1, 3456, 192))
        del arg78_1
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf49, arg79_1, arg80_1, arg81_1, arg82_1, 497664, grid=grid(497664), stream=stream0)
        del arg79_1
        del arg80_1
        del arg81_1
        del arg82_1
        # Topologically Sorted Source Nodes: [x_152, x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg83_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf50, (8, 192, 18, 18), (62208, 1, 3456, 192))
        del arg83_1
        del buf49
        # Topologically Sorted Source Nodes: [x_155], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 192, 18, 18), (62208, 1, 3456, 192))
        del arg84_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf52, arg85_1, arg86_1, arg87_1, arg88_1, 497664, grid=grid(497664), stream=stream0)
        del arg85_1
        del arg86_1
        del arg87_1
        del arg88_1
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, arg89_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf53, (8, 192, 18, 18), (62208, 1, 3456, 192))
        del arg89_1
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 192, 18, 18), (62208, 1, 3456, 192))
        del arg90_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_160, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf55, arg91_1, arg92_1, arg93_1, arg94_1, 497664, grid=grid(497664), stream=stream0)
        del arg91_1
        del arg92_1
        del arg93_1
        del arg94_1
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg95_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192, bias=None)
        assert_size_stride(buf56, (8, 192, 18, 18), (62208, 1, 3456, 192))
        del arg95_1
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 192, 18, 18), (62208, 1, 3456, 192))
        del arg96_1
        del buf56
        buf58 = empty_strided_cuda((8, 1088, 18, 18), (352512, 1, 19584, 1088), torch.float32)
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf47, buf52, buf55, buf57, arg97_1, arg98_1, arg99_1, arg100_1, buf58, 2820096, grid=grid(2820096), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf47
        del buf52
        del buf55
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 768, 18, 18), (248832, 1, 13824, 768))
        del arg101_1
        del buf58
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_168, x_169], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf60, arg102_1, arg103_1, arg104_1, arg105_1, 1990656, grid=grid(1990656), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        buf61 = empty_strided_cuda((8, 768, 1, 1, 3), (2304, 1, 18432, 18432, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_red_fused_mean_21.run(buf60, buf61, 18432, 108, grid=grid(18432), stream=stream0)
        buf63 = empty_strided_cuda((8, 768, 1, 1), (768, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_12], Original ATen: [aten.mean]
        triton_per_fused_mean_22.run(buf61, buf63, 6144, 3, grid=grid(6144), stream=stream0)
        del buf61
        # Topologically Sorted Source Nodes: [x_se_12, x_se_13], Original ATen: [aten.mean, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg106_1
        del buf63
        buf65 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_se_12, x_se_13, hardsigmoid_6, x_170], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_23.run(buf65, buf64, arg107_1, 1990656, grid=grid(1990656), stream=stream0)
        del arg107_1
        del buf64
        buf66 = reinterpret_tensor(buf57, (8, 768, 9, 9), (62208, 1, 6912, 768), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_se_12, x_se_13, hardsigmoid_6, x_170, x_171], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_hardsigmoid_max_pool2d_with_indices_mean_mul_24.run(buf65, buf66, 497664, grid=grid(497664), stream=stream0)
        del buf65
        # Topologically Sorted Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 224, 9, 9), (18144, 1, 2016, 224))
        del arg108_1
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_173, x_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf68, arg109_1, arg110_1, arg111_1, arg112_1, 145152, grid=grid(145152), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        del arg112_1
        # Topologically Sorted Source Nodes: [x_173, x_174, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg113_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf69, (8, 224, 9, 9), (18144, 1, 2016, 224))
        del arg113_1
        del buf68
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 224, 9, 9), (18144, 1, 2016, 224))
        del arg114_1
        del buf69
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_177, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf71, arg115_1, arg116_1, arg117_1, arg118_1, 145152, grid=grid(145152), stream=stream0)
        del arg115_1
        del arg116_1
        del arg117_1
        del arg118_1
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg119_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf72, (8, 224, 9, 9), (18144, 1, 2016, 224))
        del arg119_1
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 224, 9, 9), (18144, 1, 2016, 224))
        del arg120_1
        del buf72
        buf74 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_181, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf74, arg121_1, arg122_1, arg123_1, arg124_1, 145152, grid=grid(145152), stream=stream0)
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg125_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=224, bias=None)
        assert_size_stride(buf75, (8, 224, 9, 9), (18144, 1, 2016, 224))
        del arg125_1
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 224, 9, 9), (18144, 1, 2016, 224))
        del arg126_1
        del buf75
        buf77 = empty_strided_cuda((8, 1440, 9, 9), (116640, 1, 12960, 1440), torch.float32)
        # Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf66, buf71, buf74, buf76, arg127_1, arg128_1, arg129_1, arg130_1, buf77, 933120, grid=grid(933120), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        del buf66
        del buf71
        del buf74
        del buf76
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 1024, 9, 9), (82944, 1, 9216, 1024))
        del arg131_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_189, x_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf79, arg132_1, arg133_1, arg134_1, arg135_1, 663552, grid=grid(663552), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf81 = empty_strided_cuda((8, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_se_14], Original ATen: [aten.mean]
        triton_red_fused_mean_28.run(buf79, buf81, 8192, 81, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.mean, aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg136_1
        buf83 = reinterpret_tensor(buf82, (8, 1024, 1, 1), (1024, 1, 8192, 8192), 0); del buf82  # reuse
        buf84 = reinterpret_tensor(buf81, (8, 1024, 1, 1), (1024, 1, 8192, 8192), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_se_14, x_se_15, hardsigmoid_7, x_191, x_192], Original ATen: [aten.mean, aten.convolution, aten.hardsigmoid, aten.mul]
        triton_red_fused_convolution_hardsigmoid_mean_mul_29.run(buf83, buf79, arg137_1, buf84, 8192, 81, grid=grid(8192), stream=stream0)
        del arg137_1
        del buf79
        del buf83
        buf85 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg139_1, reinterpret_tensor(buf84, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg138_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf85)
        del arg138_1
        del arg139_1
        del buf84
    return (buf85, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 288, 288), (248832, 82944, 288, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((160, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((160, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((160, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((192, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((192, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((224, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((224, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 1440, 1, 1), (1440, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ese_vovnet19b_dw', benchmark_compiled_module)
