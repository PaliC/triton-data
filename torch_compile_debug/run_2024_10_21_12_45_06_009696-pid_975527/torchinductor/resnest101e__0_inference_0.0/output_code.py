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


# kernel path: /tmp/torchinductor_sahanp/bc/cbczdymg6rvzlxtsbvsq6hhitzx3d5f3kgch5nh3br2xr7dsku25.py
# Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   input_20 => add_312, mul_451, mul_452, sub_172
#   input_21 => relu_135
#   input_22 => convolution_173
# Graph fragment:
#   %sub_172 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_172, %unsqueeze_1113), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_172, %unsqueeze_1115), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, %unsqueeze_1117), kwargs = {})
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_452, %unsqueeze_1119), kwargs = {})
#   %relu_135 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_312,), kwargs = {})
#   %convolution_173 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_135, %arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 16384) % 64
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


# kernel path: /tmp/torchinductor_sahanp/an/canvfxnoruedvv3apnetehfrcuezmlkzuhm7644x76r3dsu6oka2.py
# Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_237 => add_316, mul_457, mul_458, sub_174
#   x_238 => relu_137
# Graph fragment:
#   %sub_174 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_174, %unsqueeze_1129), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_174, %unsqueeze_1131), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_457, %unsqueeze_1133), kwargs = {})
#   %add_316 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_458, %unsqueeze_1135), kwargs = {})
#   %relu_137 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_316,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 16384) % 128
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


# kernel path: /tmp/torchinductor_sahanp/xr/cxrubwxqutw7x6uwtreqldrohezqezcm5tdtbbeksiahgex7iui5.py
# Topologically Sorted Source Nodes: [x_237, x_238, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_237 => add_316, mul_457, mul_458, sub_174
#   x_238 => relu_137
#   x_239 => _low_memory_max_pool2d_with_offsets_1
# Graph fragment:
#   %sub_174 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_174, %unsqueeze_1129), kwargs = {})
#   %mul_457 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_174, %unsqueeze_1131), kwargs = {})
#   %mul_458 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_457, %unsqueeze_1133), kwargs = {})
#   %add_316 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_458, %unsqueeze_1135), kwargs = {})
#   %relu_137 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_316,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_137, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x3 = (xindex // 64)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-129) + (2*x0) + (256*x3)), tmp10, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-128) + (2*x0) + (256*x3)), tmp16, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-127) + (2*x0) + (256*x3)), tmp23, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + (2*x0) + (256*x3)), tmp30, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + ((2*x0) + (256*x3)), tmp33, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + (2*x0) + (256*x3)), tmp36, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + (2*x1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (127 + (2*x0) + (256*x3)), tmp43, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (128 + (2*x0) + (256*x3)), tmp46, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (129 + (2*x0) + (256*x3)), tmp49, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x4), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4f/c4fdxczs446qhcmxfjah4hyrdcs5gqifmjvlsrn42cdifzptml5h.py
# Topologically Sorted Source Nodes: [out_301, out_302, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_301 => add_318, mul_460, mul_461, sub_175
#   out_302 => relu_138
#   x_240 => convolution_176
# Graph fragment:
#   %sub_175 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_175, %unsqueeze_1137), kwargs = {})
#   %mul_460 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_175, %unsqueeze_1139), kwargs = {})
#   %mul_461 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_460, %unsqueeze_1141), kwargs = {})
#   %add_318 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_461, %unsqueeze_1143), kwargs = {})
#   %relu_138 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_318,), kwargs = {})
#   %convolution_176 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_138, %arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 64
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


# kernel path: /tmp/torchinductor_sahanp/wc/cwcwe2olxitw6vmjhqneffhrdjzy2tb7avzohwwhwag3f2vwgm36.py
# Topologically Sorted Source Nodes: [x_241, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_241 => add_320, mul_463, mul_464, sub_176
#   x_242 => relu_139
# Graph fragment:
#   %sub_176 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_176, %unsqueeze_1145), kwargs = {})
#   %mul_463 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_176, %unsqueeze_1147), kwargs = {})
#   %mul_464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_463, %unsqueeze_1149), kwargs = {})
#   %add_320 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_464, %unsqueeze_1151), kwargs = {})
#   %relu_139 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_320,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 128
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


# kernel path: /tmp/torchinductor_sahanp/6i/c6iagck754qm2vwakjdhho7jjo2gi75dzbwlyvubq2m3y3xjece4.py
# Topologically Sorted Source Nodes: [x_gap_165, x_gap_166, x_gap_167], Original ATen: [aten.sum, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_gap_165 => sum_100
#   x_gap_166 => mean_34
#   x_gap_167 => convolution_177
# Graph fragment:
#   %sum_100 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_200, [1]), kwargs = {})
#   %mean_34 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_100, [2, 3], True), kwargs = {})
#   %convolution_177 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_34, %arg26_1, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_red_fused_convolution_mean_sum_5 = async_compile.triton('triton_red_fused_convolution_mean_sum_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_sum_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_sum_5(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x0) + (524288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (262144 + r2 + (4096*x0) + (524288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ws/cws44ed62zajr4hi6na6clwlmgs2edebpaieiopamdwccjjskr35.py
# Topologically Sorted Source Nodes: [x_gap_165, x_gap_166, x_gap_167, x_gap_168, x_gap_169, x_attn_66], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_attn_66 => convolution_178
#   x_gap_165 => sum_100
#   x_gap_166 => mean_34
#   x_gap_167 => convolution_177
#   x_gap_168 => add_322, mul_466, mul_467, sub_177
#   x_gap_169 => relu_140
# Graph fragment:
#   %sum_100 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_200, [1]), kwargs = {})
#   %mean_34 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_100, [2, 3], True), kwargs = {})
#   %convolution_177 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_34, %arg26_1, %arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_177 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_177, %unsqueeze_1153), kwargs = {})
#   %mul_466 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_177, %unsqueeze_1155), kwargs = {})
#   %mul_467 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_466, %unsqueeze_1157), kwargs = {})
#   %add_322 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_467, %unsqueeze_1159), kwargs = {})
#   %relu_140 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_322,), kwargs = {})
#   %convolution_178 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_140, %arg32_1, %arg33_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cl/cclhagfynexyozugvpfqc3yjjmvc5zada5p5ok67n4bayutwnb4q.py
# Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_245 => amax_33, exp_33, sub_178
# Graph fragment:
#   %amax_33 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%permute_34, [1], True), kwargs = {})
#   %sub_178 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_34, %amax_33), kwargs = {})
#   %exp_33 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_178,), kwargs = {})
triton_poi_fused__softmax_7 = async_compile.triton('triton_poi_fused__softmax_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 128
    x0 = xindex % 64
    x2 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hp/chp6wgtwlw4rbdqvwb4qn2cpze436frvapr7z4e4a7ixs7oovokj.py
# Topologically Sorted Source Nodes: [mul_33, out_303, out_305], Original ATen: [aten.mul, aten.sum, aten.convolution]
# Source node to ATen node mapping:
#   mul_33 => mul_468
#   out_303 => sum_102
#   out_305 => convolution_179
# Graph fragment:
#   %mul_468 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_200, %view_204), kwargs = {})
#   %sum_102 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_468, [1]), kwargs = {})
#   %convolution_179 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%sum_102, %arg34_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_mul_sum_8 = async_compile.triton('triton_poi_fused_convolution_mul_sum_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sum_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 262144)
    x3 = xindex % 262144
    x1 = (xindex // 4096) % 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (524288*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (128*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (64 + x1 + (128*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (262144 + x3 + (524288*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a5/ca5p2uerekdjyp2tzfzehqwhpc5d64zpmf7mheexckqufne6ezme.py
# Topologically Sorted Source Nodes: [out_306, input_27, out_307, out_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_27 => add_326, mul_473, mul_474, sub_180
#   out_306 => add_324, mul_470, mul_471, sub_179
#   out_307 => add_327
#   out_308 => relu_141
# Graph fragment:
#   %sub_179 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_179, %unsqueeze_1161), kwargs = {})
#   %mul_470 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_179, %unsqueeze_1163), kwargs = {})
#   %mul_471 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_470, %unsqueeze_1165), kwargs = {})
#   %add_324 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_471, %unsqueeze_1167), kwargs = {})
#   %sub_180 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_180, %unsqueeze_1169), kwargs = {})
#   %mul_473 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_180, %unsqueeze_1171), kwargs = {})
#   %mul_474 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_473, %unsqueeze_1173), kwargs = {})
#   %add_326 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_474, %unsqueeze_1175), kwargs = {})
#   %add_327 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_324, %add_326), kwargs = {})
#   %relu_141 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_327,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
    tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6l/c6lekfgiakebpzwpe2tz5kyg33rcbumxxmn4xbqb2agayhsjzmqd.py
# Topologically Sorted Source Nodes: [out_315, out_316, out_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_315 => add_335, mul_486, mul_487, sub_185
#   out_316 => add_336
#   out_317 => relu_145
# Graph fragment:
#   %sub_185 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_185, %unsqueeze_1201), kwargs = {})
#   %mul_486 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_185, %unsqueeze_1203), kwargs = {})
#   %mul_487 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_486, %unsqueeze_1205), kwargs = {})
#   %add_335 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_487, %unsqueeze_1207), kwargs = {})
#   %add_336 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_335, %relu_141), kwargs = {})
#   %relu_145 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_336,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ra/crasz7tbhd3ocuwy4cnebz5pg7kbk5qk3npjx4gejag6t25y6ao6.py
# Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_262 => add_349, mul_505, mul_506, sub_192
#   x_263 => relu_151
# Graph fragment:
#   %sub_192 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_192, %unsqueeze_1249), kwargs = {})
#   %mul_505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_192, %unsqueeze_1251), kwargs = {})
#   %mul_506 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_505, %unsqueeze_1253), kwargs = {})
#   %add_349 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_506, %unsqueeze_1255), kwargs = {})
#   %relu_151 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_349,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 4096) % 256
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


# kernel path: /tmp/torchinductor_sahanp/iu/ciudxlh64ugfog5thdqbc62w3zruu372c76x6b4mcnkiizeosusz.py
# Topologically Sorted Source Nodes: [x_gap_180, x_gap_181, x_gap_182], Original ATen: [aten.sum, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_gap_180 => sum_109
#   x_gap_181 => mean_37
#   x_gap_182 => convolution_193
# Graph fragment:
#   %sum_109 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_218, [1]), kwargs = {})
#   %mean_37 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_109, [2, 3], True), kwargs = {})
#   %convolution_193 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_37, %arg100_1, %arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_red_fused_convolution_mean_sum_12 = async_compile.triton('triton_red_fused_convolution_mean_sum_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_sum_12(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x0) + (1048576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (524288 + r2 + (4096*x0) + (1048576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pr/cprjytwtu2cxar5somajqlbj237xzagmmwh5csj2scx3logm2kw4.py
# Topologically Sorted Source Nodes: [x_gap_180, x_gap_181, x_gap_182, x_gap_183, x_gap_184, x_attn_72], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_attn_72 => convolution_194
#   x_gap_180 => sum_109
#   x_gap_181 => mean_37
#   x_gap_182 => convolution_193
#   x_gap_183 => add_351, mul_508, mul_509, sub_193
#   x_gap_184 => relu_152
# Graph fragment:
#   %sum_109 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_218, [1]), kwargs = {})
#   %mean_37 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_109, [2, 3], True), kwargs = {})
#   %convolution_193 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_37, %arg100_1, %arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_193 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_193, %unsqueeze_1257), kwargs = {})
#   %mul_508 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_193, %unsqueeze_1259), kwargs = {})
#   %mul_509 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_508, %unsqueeze_1261), kwargs = {})
#   %add_351 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_509, %unsqueeze_1263), kwargs = {})
#   %relu_152 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_351,), kwargs = {})
#   %convolution_194 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_152, %arg106_1, %arg107_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yg/cyg3yzcfes5neky3pm7aqepg3qmljztee6f6qbwjwmyk4v36bz6o.py
# Topologically Sorted Source Nodes: [x_266], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_266 => amax_36, exp_36, sub_194
# Graph fragment:
#   %amax_36 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%permute_37, [1], True), kwargs = {})
#   %sub_194 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_37, %amax_36), kwargs = {})
#   %exp_36 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_194,), kwargs = {})
triton_poi_fused__softmax_14 = async_compile.triton('triton_poi_fused__softmax_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_14(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 256
    x0 = xindex % 128
    x2 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (256*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (128 + x0 + (256*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/q3/cq3zjnkksnkxu5r52syodwadxtdtbvzzw7wnrpxviyeujbb7zuoy.py
# Topologically Sorted Source Nodes: [mul_36, out_330], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_36 => mul_510
#   out_330 => sum_111
# Graph fragment:
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_218, %view_222), kwargs = {})
#   %sum_111 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_510, [1]), kwargs = {})
triton_poi_fused_mul_sum_15 = async_compile.triton('triton_poi_fused_mul_sum_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 524288)
    x3 = xindex % 524288
    x1 = (xindex // 4096) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (1048576*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (128 + x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (524288 + x3 + (1048576*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yh/cyhobecmtt2a6yzxx6dm3bjptwmwdhy3f2ccdtbpxqz3ebth2qml.py
# Topologically Sorted Source Nodes: [mul_36, out_330, out_332], Original ATen: [aten.mul, aten.sum, aten.avg_pool2d]
# Source node to ATen node mapping:
#   mul_36 => mul_510
#   out_330 => sum_111
#   out_332 => avg_pool2d_6
# Graph fragment:
#   %mul_510 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_218, %view_222), kwargs = {})
#   %sum_111 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_510, [1]), kwargs = {})
#   %avg_pool2d_6 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%sum_111, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_mul_sum_16 = async_compile.triton('triton_poi_fused_avg_pool2d_mul_sum_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_mul_sum_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_mul_sum_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 32) % 32
    x0 = xindex % 32
    x3 = (xindex // 32)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-65) + (2*x0) + (128*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-64) + (2*x0) + (128*x3)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*x0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-63) + (2*x0) + (128*x3)), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + (2*x0) + (128*x3)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + ((2*x0) + (128*x3)), tmp33, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + (2*x0) + (128*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*x1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (63 + (2*x0) + (128*x3)), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (64 + (2*x0) + (128*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (65 + (2*x0) + (128*x3)), tmp49, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + (((65) * ((65) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (65)))*((65) * ((65) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (65)))) + ((-2)*x0*((65) * ((65) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (65)))) + ((-2)*x1*((65) * ((65) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (65)))) + (4*x0*x1) + ((65) * ((65) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (65))) + ((65) * ((65) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (65)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6c/c6clkxfsig7csoupfvckq2wuulxh3p3gjlueavaykqncyr7nsdo4.py
# Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   input_28 => avg_pool2d_7
#   input_29 => convolution_196
# Graph fragment:
#   %avg_pool2d_7 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_149, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_196 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_7, %arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_17 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 32
    x1 = (xindex // 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pd/cpdew7sihigixfwgg7q62qasxxjs6vppd2ol6urendtdtcjizyri.py
# Topologically Sorted Source Nodes: [out_334, input_30, out_335, out_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_30 => add_355, mul_515, mul_516, sub_196
#   out_334 => add_353, mul_512, mul_513, sub_195
#   out_335 => add_356
#   out_336 => relu_153
# Graph fragment:
#   %sub_195 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_195, %unsqueeze_1265), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_195, %unsqueeze_1267), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, %unsqueeze_1269), kwargs = {})
#   %add_353 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_513, %unsqueeze_1271), kwargs = {})
#   %sub_196 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_196, %unsqueeze_1273), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_196, %unsqueeze_1275), kwargs = {})
#   %mul_516 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_515, %unsqueeze_1277), kwargs = {})
#   %add_355 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_516, %unsqueeze_1279), kwargs = {})
#   %add_356 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_353, %add_355), kwargs = {})
#   %relu_153 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_356,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
    tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ai/caimhdiqd2bxxgnfzvhcefcpqokoae6ouogdaeli7dyi3v7t6bgr.py
# Topologically Sorted Source Nodes: [out_338, out_339, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_338 => add_358, mul_518, mul_519, sub_197
#   out_339 => relu_154
#   x_268 => convolution_198
# Graph fragment:
#   %sub_197 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_197, %unsqueeze_1281), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_197, %unsqueeze_1283), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_518, %unsqueeze_1285), kwargs = {})
#   %add_358 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_519, %unsqueeze_1287), kwargs = {})
#   %relu_154 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_358,), kwargs = {})
#   %convolution_198 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_154, %arg123_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 128
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


# kernel path: /tmp/torchinductor_sahanp/h7/ch7b45q5fvmwtqubaivdonhtm7jx67wkrmfkotb4y24jgzsgx7u2.py
# Topologically Sorted Source Nodes: [x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_269 => add_360, mul_521, mul_522, sub_198
#   x_270 => relu_155
# Graph fragment:
#   %sub_198 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_198, %unsqueeze_1289), kwargs = {})
#   %mul_521 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_198, %unsqueeze_1291), kwargs = {})
#   %mul_522 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_521, %unsqueeze_1293), kwargs = {})
#   %add_360 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_522, %unsqueeze_1295), kwargs = {})
#   %relu_155 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_360,), kwargs = {})
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
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 256
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


# kernel path: /tmp/torchinductor_sahanp/7h/c7huyei5aqkecke62zengiuwueohzpxuwkff36t3fnlam22ihnq7.py
# Topologically Sorted Source Nodes: [x_gap_185, x_gap_186, x_gap_187], Original ATen: [aten.sum, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_gap_185 => sum_112
#   x_gap_186 => mean_38
#   x_gap_187 => convolution_199
# Graph fragment:
#   %sum_112 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_224, [1]), kwargs = {})
#   %mean_38 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_112, [2, 3], True), kwargs = {})
#   %convolution_199 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_38, %arg128_1, %arg129_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_convolution_mean_sum_21 = async_compile.triton('triton_per_fused_convolution_mean_sum_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_sum_21(in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0) + (262144*x1)), None)
    tmp1 = tl.load(in_ptr0 + (131072 + r2 + (1024*x0) + (262144*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 1024.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a7/ca7np7jqicb3lab7layrng3fzmbowoo7d7hc54ugxqj5xz5dd3yh.py
# Topologically Sorted Source Nodes: [mul_37, out_340, out_342], Original ATen: [aten.mul, aten.sum, aten.convolution]
# Source node to ATen node mapping:
#   mul_37 => mul_526
#   out_340 => sum_114
#   out_342 => convolution_201
# Graph fragment:
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_224, %view_228), kwargs = {})
#   %sum_114 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_526, [1]), kwargs = {})
#   %convolution_201 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%sum_114, %arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_mul_sum_22 = async_compile.triton('triton_poi_fused_convolution_mul_sum_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sum_22(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 131072)
    x3 = xindex % 131072
    x1 = (xindex // 1024) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (262144*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (128 + x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (131072 + x3 + (262144*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/uc/cuc3r3ulgqm3ofrgbspdmgyiouo2cmagytn3vgb2crwzugjr5fbx.py
# Topologically Sorted Source Nodes: [out_343, out_344, out_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_343 => add_364, mul_528, mul_529, sub_201
#   out_344 => add_365
#   out_345 => relu_157
# Graph fragment:
#   %sub_201 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_201, %unsqueeze_1305), kwargs = {})
#   %mul_528 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_201, %unsqueeze_1307), kwargs = {})
#   %mul_529 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_528, %unsqueeze_1309), kwargs = {})
#   %add_364 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_529, %unsqueeze_1311), kwargs = {})
#   %add_365 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_364, %relu_153), kwargs = {})
#   %relu_157 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_365,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7x/c7xvibrio2fhwzinfnvchpg2w35hryqsarxtogvnpwecmiw5jb7a.py
# Topologically Sorted Source Nodes: [out_361, out_362, out_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_361 => add_382, mul_554, mul_555, sub_211
#   out_362 => add_383
#   out_363 => relu_165
# Graph fragment:
#   %sub_211 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_211, %unsqueeze_1369), kwargs = {})
#   %mul_554 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_211, %unsqueeze_1371), kwargs = {})
#   %mul_555 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_554, %unsqueeze_1373), kwargs = {})
#   %add_382 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_555, %unsqueeze_1375), kwargs = {})
#   %add_383 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_382, %relu_161), kwargs = {})
#   %relu_165 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_383,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4q/c4qic5fzgnryyg7y46f2ftca4hnleqnhy23lx546q4woyuohw6yb.py
# Topologically Sorted Source Nodes: [x_290, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_290 => add_387, mul_560, mul_561, sub_213
#   x_291 => relu_167
# Graph fragment:
#   %sub_213 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_213, %unsqueeze_1385), kwargs = {})
#   %mul_560 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_213, %unsqueeze_1387), kwargs = {})
#   %mul_561 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_560, %unsqueeze_1389), kwargs = {})
#   %add_387 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_561, %unsqueeze_1391), kwargs = {})
#   %relu_167 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_387,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 1024) % 512
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


# kernel path: /tmp/torchinductor_sahanp/ve/cve5tqeo5vbjxpypajsbgt5rww23fgujchgwtb7dh37tka6h52fl.py
# Topologically Sorted Source Nodes: [x_gap_200, x_gap_201, x_gap_202], Original ATen: [aten.sum, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_gap_200 => sum_121
#   x_gap_201 => mean_41
#   x_gap_202 => convolution_214
# Graph fragment:
#   %sum_121 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_242, [1]), kwargs = {})
#   %mean_41 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_121, [2, 3], True), kwargs = {})
#   %convolution_214 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_41, %arg197_1, %arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_convolution_mean_sum_26 = async_compile.triton('triton_per_fused_convolution_mean_sum_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_sum_26(in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0) + (524288*x1)), None)
    tmp1 = tl.load(in_ptr0 + (262144 + r2 + (1024*x0) + (524288*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 1024.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zs/czs44oebw4qsuljpwzd2smudqemx27wpqmqrivw2g3yyzrkuxih2.py
# Topologically Sorted Source Nodes: [x_gap_200, x_gap_201, x_gap_202, x_gap_203, x_gap_204, x_attn_80], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_attn_80 => convolution_215
#   x_gap_200 => sum_121
#   x_gap_201 => mean_41
#   x_gap_202 => convolution_214
#   x_gap_203 => add_389, mul_563, mul_564, sub_214
#   x_gap_204 => relu_168
# Graph fragment:
#   %sum_121 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_242, [1]), kwargs = {})
#   %mean_41 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_121, [2, 3], True), kwargs = {})
#   %convolution_214 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_41, %arg197_1, %arg198_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_214 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_214, %unsqueeze_1393), kwargs = {})
#   %mul_563 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_214, %unsqueeze_1395), kwargs = {})
#   %mul_564 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_563, %unsqueeze_1397), kwargs = {})
#   %add_389 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_564, %unsqueeze_1399), kwargs = {})
#   %relu_168 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_389,), kwargs = {})
#   %convolution_215 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_168, %arg203_1, %arg204_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dx/cdxbfesauyqk5nzd7ufk2ojdlhs24z4owgdkw37qzoku2a6h6z7v.py
# Topologically Sorted Source Nodes: [x_294], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_294 => amax_40, exp_40, sub_215
# Graph fragment:
#   %amax_40 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%permute_41, [1], True), kwargs = {})
#   %sub_215 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_41, %amax_40), kwargs = {})
#   %exp_40 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_215,), kwargs = {})
triton_poi_fused__softmax_28 = async_compile.triton('triton_poi_fused__softmax_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex % 512
    x0 = xindex % 256
    x2 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (256 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dm/cdmqmd6nrrpahg55ndvegdar2rwuwy4msllrgvsicfko72fbsfhu.py
# Topologically Sorted Source Nodes: [mul_40, out_367], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_40 => mul_565
#   out_367 => sum_123
# Graph fragment:
#   %mul_565 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_242, %view_246), kwargs = {})
#   %sum_123 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_565, [1]), kwargs = {})
triton_poi_fused_mul_sum_29 = async_compile.triton('triton_poi_fused_mul_sum_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 262144)
    x3 = xindex % 262144
    x1 = (xindex // 1024) % 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (524288*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (256 + x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (262144 + x3 + (524288*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4q/c4qzdrngogjqopnupsux5lqye5gw3q4kxyz3mc5opcqglocgjquv.py
# Topologically Sorted Source Nodes: [mul_40, out_367, out_369], Original ATen: [aten.mul, aten.sum, aten.avg_pool2d]
# Source node to ATen node mapping:
#   mul_40 => mul_565
#   out_367 => sum_123
#   out_369 => avg_pool2d_8
# Graph fragment:
#   %mul_565 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_242, %view_246), kwargs = {})
#   %sum_123 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_565, [1]), kwargs = {})
#   %avg_pool2d_8 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%sum_123, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_mul_sum_30 = async_compile.triton('triton_poi_fused_avg_pool2d_mul_sum_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_mul_sum_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_mul_sum_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 16) % 16
    x0 = xindex % 16
    x3 = (xindex // 16)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + (2*x0) + (64*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + (2*x0) + (64*x3)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*x0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + (2*x0) + (64*x3)), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + (2*x0) + (64*x3)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + ((2*x0) + (64*x3)), tmp33, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + (2*x0) + (64*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*x1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + (2*x0) + (64*x3)), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + (2*x0) + (64*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + (2*x0) + (64*x3)), tmp49, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + (((33) * ((33) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (33)))*((33) * ((33) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (33)))) + ((-2)*x0*((33) * ((33) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (33)))) + ((-2)*x1*((33) * ((33) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (33)))) + (4*x0*x1) + ((33) * ((33) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (33))) + ((33) * ((33) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (33)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4u/c4ua4tizul2lo5pfiyy5ijmided2j6k27xmsjheqdiv6fxiyjlu3.py
# Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   input_31 => avg_pool2d_9
#   input_32 => convolution_217
# Graph fragment:
#   %avg_pool2d_9 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_165, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_217 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_9, %arg210_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_31 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_31(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/d5/cd5wn6mai6fhlcbzerft4w3w4csxp32kr77wfzryi552xf4cikn7.py
# Topologically Sorted Source Nodes: [out_371, input_33, out_372, out_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_33 => add_393, mul_570, mul_571, sub_217
#   out_371 => add_391, mul_567, mul_568, sub_216
#   out_372 => add_394
#   out_373 => relu_169
# Graph fragment:
#   %sub_216 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_216, %unsqueeze_1401), kwargs = {})
#   %mul_567 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_216, %unsqueeze_1403), kwargs = {})
#   %mul_568 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_567, %unsqueeze_1405), kwargs = {})
#   %add_391 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_568, %unsqueeze_1407), kwargs = {})
#   %sub_217 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_217, %unsqueeze_1409), kwargs = {})
#   %mul_570 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_217, %unsqueeze_1411), kwargs = {})
#   %mul_571 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_570, %unsqueeze_1413), kwargs = {})
#   %add_393 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_571, %unsqueeze_1415), kwargs = {})
#   %add_394 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_391, %add_393), kwargs = {})
#   %relu_169 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_394,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
    tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6p/c6pslzxjypg2o3t42qz2xztkrkbednee4u7vf3xkn5xmxjyivgvx.py
# Topologically Sorted Source Nodes: [out_375, out_376, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_375 => add_396, mul_573, mul_574, sub_218
#   out_376 => relu_170
#   x_296 => convolution_219
# Graph fragment:
#   %sub_218 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_218, %unsqueeze_1417), kwargs = {})
#   %mul_573 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_218, %unsqueeze_1419), kwargs = {})
#   %mul_574 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_573, %unsqueeze_1421), kwargs = {})
#   %add_396 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_574, %unsqueeze_1423), kwargs = {})
#   %relu_170 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_396,), kwargs = {})
#   %convolution_219 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_170, %arg220_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 256
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


# kernel path: /tmp/torchinductor_sahanp/32/c32eo3njqocnyktj4gok5s2vx4fncs7jbb5z5wrb4iccle746hje.py
# Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_297 => add_398, mul_576, mul_577, sub_219
#   x_298 => relu_171
# Graph fragment:
#   %sub_219 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_219, %unsqueeze_1425), kwargs = {})
#   %mul_576 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_219, %unsqueeze_1427), kwargs = {})
#   %mul_577 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_576, %unsqueeze_1429), kwargs = {})
#   %add_398 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_577, %unsqueeze_1431), kwargs = {})
#   %relu_171 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_398,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 512
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


# kernel path: /tmp/torchinductor_sahanp/53/c53qh56pmlgxq6ozuyj6auna2oyesfjlx4rp74462zwz4xokdwt5.py
# Topologically Sorted Source Nodes: [x_gap_205, x_gap_206, x_gap_207], Original ATen: [aten.sum, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_gap_205 => sum_124
#   x_gap_206 => mean_42
#   x_gap_207 => convolution_220
# Graph fragment:
#   %sum_124 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_248, [1]), kwargs = {})
#   %mean_42 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_124, [2, 3], True), kwargs = {})
#   %convolution_220 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_42, %arg225_1, %arg226_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_convolution_mean_sum_35 = async_compile.triton('triton_per_fused_convolution_mean_sum_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_sum_35(in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x0) + (131072*x1)), None)
    tmp1 = tl.load(in_ptr0 + (65536 + r2 + (256*x0) + (131072*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mx/cmxglpgxjth3rnrymm7s5i3ljiiougvf4n3to37s3fmtbhetluj7.py
# Topologically Sorted Source Nodes: [mul_41, out_377, out_379], Original ATen: [aten.mul, aten.sum, aten.convolution]
# Source node to ATen node mapping:
#   mul_41 => mul_581
#   out_377 => sum_126
#   out_379 => convolution_222
# Graph fragment:
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_248, %view_252), kwargs = {})
#   %sum_126 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_581, [1]), kwargs = {})
#   %convolution_222 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%sum_126, %arg233_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_mul_sum_36 = async_compile.triton('triton_poi_fused_convolution_mul_sum_36', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sum_36(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 65536)
    x3 = xindex % 65536
    x1 = (xindex // 256) % 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (131072*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (256 + x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (65536 + x3 + (131072*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ig/ciggwfb5edv3x57dbm2ats6dtgahs7by3se7yfrf63k3hib7dcvu.py
# Topologically Sorted Source Nodes: [out_380, out_381, out_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_380 => add_402, mul_583, mul_584, sub_222
#   out_381 => add_403
#   out_382 => relu_173
# Graph fragment:
#   %sub_222 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_222, %unsqueeze_1441), kwargs = {})
#   %mul_583 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_222, %unsqueeze_1443), kwargs = {})
#   %mul_584 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_583, %unsqueeze_1445), kwargs = {})
#   %add_402 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_584, %unsqueeze_1447), kwargs = {})
#   %add_403 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_402, %relu_169), kwargs = {})
#   %relu_173 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_403,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nl/cnlhies2kmxmdasiddffq4aun6whjrqjjiiqa3q3rbemab5ws4ax.py
# Topologically Sorted Source Nodes: [x_451, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_451 => add_596, mul_862, mul_863, sub_329
#   x_452 => relu_259
# Graph fragment:
#   %sub_329 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_329, %unsqueeze_2129), kwargs = {})
#   %mul_862 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_329, %unsqueeze_2131), kwargs = {})
#   %mul_863 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_862, %unsqueeze_2133), kwargs = {})
#   %add_596 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_863, %unsqueeze_2135), kwargs = {})
#   %relu_259 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_596,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 256) % 1024
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


# kernel path: /tmp/torchinductor_sahanp/go/cgoqoxmdu5yqehs4ncsfril6ftdfx7wsu3depfod65szcuugeq7h.py
# Topologically Sorted Source Nodes: [x_gap_315, x_gap_316, x_gap_317], Original ATen: [aten.sum, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_gap_315 => sum_190
#   x_gap_316 => mean_64
#   x_gap_317 => convolution_330
# Graph fragment:
#   %sum_190 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_380, [1]), kwargs = {})
#   %mean_64 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_190, [2, 3], True), kwargs = {})
#   %convolution_330 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_64, %arg731_1, %arg732_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_convolution_mean_sum_39 = async_compile.triton('triton_per_fused_convolution_mean_sum_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_sum_39(in_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x0) + (262144*x1)), None)
    tmp1 = tl.load(in_ptr0 + (131072 + r2 + (256*x0) + (262144*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yb/cybcx4e5mv4m25yftp5xjupkur4x4kugxsro7vvgvxwd6nadyrsn.py
# Topologically Sorted Source Nodes: [x_gap_315, x_gap_316, x_gap_317, x_gap_318, x_gap_319, x_attn_126], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_attn_126 => convolution_331
#   x_gap_315 => sum_190
#   x_gap_316 => mean_64
#   x_gap_317 => convolution_330
#   x_gap_318 => add_598, mul_865, mul_866, sub_330
#   x_gap_319 => relu_260
# Graph fragment:
#   %sum_190 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_380, [1]), kwargs = {})
#   %mean_64 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_190, [2, 3], True), kwargs = {})
#   %convolution_330 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_64, %arg731_1, %arg732_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_330 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_330, %unsqueeze_2137), kwargs = {})
#   %mul_865 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_330, %unsqueeze_2139), kwargs = {})
#   %mul_866 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_865, %unsqueeze_2141), kwargs = {})
#   %add_598 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_866, %unsqueeze_2143), kwargs = {})
#   %relu_260 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_598,), kwargs = {})
#   %convolution_331 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_260, %arg737_1, %arg738_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rv/crvwvtcprbteuxms74jfjjor32w6cy22yc7k3guimzewzjnvg63o.py
# Topologically Sorted Source Nodes: [x_455], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   x_455 => amax_63, exp_63, sub_331
# Graph fragment:
#   %amax_63 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%permute_64, [1], True), kwargs = {})
#   %sub_331 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_64, %amax_63), kwargs = {})
#   %exp_63 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_331,), kwargs = {})
triton_poi_fused__softmax_41 = async_compile.triton('triton_poi_fused__softmax_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_41(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex % 1024
    x0 = xindex % 512
    x2 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (512 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/43/c43plyf2gmncq7eb23gmcbcbqe4b6c2sebt7kgpfcbz63jfwjaql.py
# Topologically Sorted Source Nodes: [mul_63, out_575], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_63 => mul_867
#   out_575 => sum_192
# Graph fragment:
#   %mul_867 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_380, %view_384), kwargs = {})
#   %sum_192 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_867, [1]), kwargs = {})
triton_poi_fused_mul_sum_42 = async_compile.triton('triton_poi_fused_mul_sum_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_42(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 131072)
    x3 = xindex % 131072
    x1 = (xindex // 256) % 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (262144*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (131072 + x3 + (262144*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/p5/cp5vyic2c7zsjruegmaeybwczzalponiolmq52g4ml25nglsgr2g.py
# Topologically Sorted Source Nodes: [mul_63, out_575, out_577], Original ATen: [aten.mul, aten.sum, aten.avg_pool2d]
# Source node to ATen node mapping:
#   mul_63 => mul_867
#   out_575 => sum_192
#   out_577 => avg_pool2d_10
# Graph fragment:
#   %mul_867 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_380, %view_384), kwargs = {})
#   %sum_192 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_867, [1]), kwargs = {})
#   %avg_pool2d_10 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%sum_192, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_mul_sum_43 = async_compile.triton('triton_poi_fused_avg_pool2d_mul_sum_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_mul_sum_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_mul_sum_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = (xindex // 8) % 8
    x0 = xindex % 8
    x3 = (xindex // 8)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-17) + (2*x0) + (32*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-16) + (2*x0) + (32*x3)), tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*x0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-15) + (2*x0) + (32*x3)), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + (2*x0) + (32*x3)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + ((2*x0) + (32*x3)), tmp33, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*x1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (15 + (2*x0) + (32*x3)), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x3)), tmp49, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + (((17) * ((17) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (17)))*((17) * ((17) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (17)))) + ((-2)*x0*((17) * ((17) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (17)))) + ((-2)*x1*((17) * ((17) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (17)))) + (4*x0*x1) + ((17) * ((17) <= (2 + (2*x0))) + (2 + (2*x0)) * ((2 + (2*x0)) < (17))) + ((17) * ((17) <= (2 + (2*x1))) + (2 + (2*x1)) * ((2 + (2*x1)) < (17)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/27/c27q5t6nf5f3ifiygrr6xkavskciglxlgnfsgh7oui43l6blw6mm.py
# Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.avg_pool2d, aten.convolution]
# Source node to ATen node mapping:
#   input_34 => avg_pool2d_11
#   input_35 => convolution_333
# Graph fragment:
#   %avg_pool2d_11 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%relu_257, [2, 2], [2, 2], [0, 0], True, False), kwargs = {})
#   %convolution_333 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%avg_pool2d_11, %arg744_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_avg_pool2d_convolution_44 = async_compile.triton('triton_poi_fused_avg_pool2d_convolution_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_convolution_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex % 8
    x1 = (xindex // 8)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/of/cofdk2tgjep7kp7qkpongefrhk6l4hk63ph3yur4iv2focmojae5.py
# Topologically Sorted Source Nodes: [out_579, input_36, out_580, out_581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_36 => add_602, mul_872, mul_873, sub_333
#   out_579 => add_600, mul_869, mul_870, sub_332
#   out_580 => add_603
#   out_581 => relu_261
# Graph fragment:
#   %sub_332 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_332, %unsqueeze_2145), kwargs = {})
#   %mul_869 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_332, %unsqueeze_2147), kwargs = {})
#   %mul_870 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_869, %unsqueeze_2149), kwargs = {})
#   %add_600 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_870, %unsqueeze_2151), kwargs = {})
#   %sub_333 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_333, %unsqueeze_2153), kwargs = {})
#   %mul_872 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_333, %unsqueeze_2155), kwargs = {})
#   %mul_873 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_872, %unsqueeze_2157), kwargs = {})
#   %add_602 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_873, %unsqueeze_2159), kwargs = {})
#   %add_603 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_600, %add_602), kwargs = {})
#   %relu_261 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_603,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x3), None)
    tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/im/cimfmgb6yb52jwwzbx7nubi4tfsxghb746wbumdn7y2vmnmn3jzi.py
# Topologically Sorted Source Nodes: [out_583, out_584, x_457], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_583 => add_605, mul_875, mul_876, sub_334
#   out_584 => relu_262
#   x_457 => convolution_335
# Graph fragment:
#   %sub_334 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_334, %unsqueeze_2161), kwargs = {})
#   %mul_875 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_334, %unsqueeze_2163), kwargs = {})
#   %mul_876 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_875, %unsqueeze_2165), kwargs = {})
#   %add_605 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_876, %unsqueeze_2167), kwargs = {})
#   %relu_262 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_605,), kwargs = {})
#   %convolution_335 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_262, %arg754_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 512
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


# kernel path: /tmp/torchinductor_sahanp/yh/cyhxzcmm4wuc26a6upevoeluwzhmlvd3vr72jsrdcjlhdpm62qpz.py
# Topologically Sorted Source Nodes: [x_458, x_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_458 => add_607, mul_878, mul_879, sub_335
#   x_459 => relu_263
# Graph fragment:
#   %sub_335 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_335, %unsqueeze_2169), kwargs = {})
#   %mul_878 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_335, %unsqueeze_2171), kwargs = {})
#   %mul_879 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_878, %unsqueeze_2173), kwargs = {})
#   %add_607 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_879, %unsqueeze_2175), kwargs = {})
#   %relu_263 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_607,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 1024
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


# kernel path: /tmp/torchinductor_sahanp/6w/c6wuijcflhgo54pma5fkhgdw6guwtaqcgkky5e4o5apatcz4zaog.py
# Topologically Sorted Source Nodes: [x_gap_320, x_gap_321, x_gap_322], Original ATen: [aten.sum, aten.mean, aten.convolution]
# Source node to ATen node mapping:
#   x_gap_320 => sum_193
#   x_gap_321 => mean_65
#   x_gap_322 => convolution_336
# Graph fragment:
#   %sum_193 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_386, [1]), kwargs = {})
#   %mean_65 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_193, [2, 3], True), kwargs = {})
#   %convolution_336 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_65, %arg759_1, %arg760_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_per_fused_convolution_mean_sum_48 = async_compile.triton('triton_per_fused_convolution_mean_sum_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[4096, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_sum_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_mean_sum_48(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x0) + (65536*x1)), None)
    tmp1 = tl.load(in_ptr0 + (32768 + r2 + (64*x0) + (65536*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None]
    tmp6 = 64.0
    tmp7 = tmp5 / tmp6
    tl.store(out_ptr1 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/tr/ctr5fcceambrajnknfcwwf3jkfqedbeangismzkfngixcamx7ni6.py
# Topologically Sorted Source Nodes: [mul_64, out_585, out_587], Original ATen: [aten.mul, aten.sum, aten.convolution]
# Source node to ATen node mapping:
#   mul_64 => mul_883
#   out_585 => sum_195
#   out_587 => convolution_338
# Graph fragment:
#   %mul_883 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_386, %view_390), kwargs = {})
#   %sum_195 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_883, [1]), kwargs = {})
#   %convolution_338 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%sum_195, %arg767_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_mul_sum_49 = async_compile.triton('triton_poi_fused_convolution_mul_sum_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sum_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sum_49(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 32768)
    x3 = xindex % 32768
    x1 = (xindex // 64) % 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (32768 + x3 + (65536*x2)), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 / tmp3
    tmp5 = tmp0 * tmp4
    tmp7 = tmp2 / tmp3
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tl.store(out_ptr0 + (x4), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ro/croecloar3zizzmovz6bdx7bvbiki7iakm2yorelzjwqlprtmex5.py
# Topologically Sorted Source Nodes: [out_588, out_589, out_590], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_588 => add_611, mul_885, mul_886, sub_338
#   out_589 => add_612
#   out_590 => relu_265
# Graph fragment:
#   %sub_338 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_338, %unsqueeze_2185), kwargs = {})
#   %mul_885 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_338, %unsqueeze_2187), kwargs = {})
#   %mul_886 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_885, %unsqueeze_2189), kwargs = {})
#   %add_611 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_886, %unsqueeze_2191), kwargs = {})
#   %add_612 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_611, %relu_261), kwargs = {})
#   %relu_265 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_612,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/so/csot7p2g2yy23qoy7r5d3ykagtuwyrbxetikobgypa6j7gixulzb.py
# Topologically Sorted Source Nodes: [out_597, out_598, out_599, x_471], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   out_597 => add_620, mul_898, mul_899, sub_343
#   out_598 => add_621
#   out_599 => relu_269
#   x_471 => mean_67
# Graph fragment:
#   %sub_343 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_343, %unsqueeze_2217), kwargs = {})
#   %mul_898 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_343, %unsqueeze_2219), kwargs = {})
#   %mul_899 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_898, %unsqueeze_2221), kwargs = {})
#   %add_620 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_899, %unsqueeze_2223), kwargs = {})
#   %add_621 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_620, %relu_265), kwargs = {})
#   %relu_269 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_621,), kwargs = {})
#   %mean_67 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_269, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (r2 + (64*x3)), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None]
    tmp23 = 64.0
    tmp24 = tmp22 / tmp23
    tl.store(out_ptr1 + (x3), tmp24, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg22_1, (128, ), (1, ))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg27_1, (32, ), (1, ))
    assert_size_stride(arg28_1, (32, ), (1, ))
    assert_size_stride(arg29_1, (32, ), (1, ))
    assert_size_stride(arg30_1, (32, ), (1, ))
    assert_size_stride(arg31_1, (32, ), (1, ))
    assert_size_stride(arg32_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (64, ), (1, ))
    assert_size_stride(arg47_1, (64, ), (1, ))
    assert_size_stride(arg48_1, (64, ), (1, ))
    assert_size_stride(arg49_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg55_1, (32, ), (1, ))
    assert_size_stride(arg56_1, (32, ), (1, ))
    assert_size_stride(arg57_1, (32, ), (1, ))
    assert_size_stride(arg58_1, (32, ), (1, ))
    assert_size_stride(arg59_1, (32, ), (1, ))
    assert_size_stride(arg60_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (256, ), (1, ))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg68_1, (64, ), (1, ))
    assert_size_stride(arg69_1, (64, ), (1, ))
    assert_size_stride(arg70_1, (64, ), (1, ))
    assert_size_stride(arg71_1, (64, ), (1, ))
    assert_size_stride(arg72_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg78_1, (32, ), (1, ))
    assert_size_stride(arg79_1, (32, ), (1, ))
    assert_size_stride(arg80_1, (32, ), (1, ))
    assert_size_stride(arg81_1, (32, ), (1, ))
    assert_size_stride(arg82_1, (32, ), (1, ))
    assert_size_stride(arg83_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg91_1, (128, ), (1, ))
    assert_size_stride(arg92_1, (128, ), (1, ))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg96_1, (256, ), (1, ))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg101_1, (64, ), (1, ))
    assert_size_stride(arg102_1, (64, ), (1, ))
    assert_size_stride(arg103_1, (64, ), (1, ))
    assert_size_stride(arg104_1, (64, ), (1, ))
    assert_size_stride(arg105_1, (64, ), (1, ))
    assert_size_stride(arg106_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (512, ), (1, ))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (256, ), (1, ))
    assert_size_stride(arg126_1, (256, ), (1, ))
    assert_size_stride(arg127_1, (256, ), (1, ))
    assert_size_stride(arg128_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg129_1, (64, ), (1, ))
    assert_size_stride(arg130_1, (64, ), (1, ))
    assert_size_stride(arg131_1, (64, ), (1, ))
    assert_size_stride(arg132_1, (64, ), (1, ))
    assert_size_stride(arg133_1, (64, ), (1, ))
    assert_size_stride(arg134_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, ), (1, ))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (128, ), (1, ))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg147_1, (256, ), (1, ))
    assert_size_stride(arg148_1, (256, ), (1, ))
    assert_size_stride(arg149_1, (256, ), (1, ))
    assert_size_stride(arg150_1, (256, ), (1, ))
    assert_size_stride(arg151_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg152_1, (64, ), (1, ))
    assert_size_stride(arg153_1, (64, ), (1, ))
    assert_size_stride(arg154_1, (64, ), (1, ))
    assert_size_stride(arg155_1, (64, ), (1, ))
    assert_size_stride(arg156_1, (64, ), (1, ))
    assert_size_stride(arg157_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg158_1, (256, ), (1, ))
    assert_size_stride(arg159_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg165_1, (128, ), (1, ))
    assert_size_stride(arg166_1, (128, ), (1, ))
    assert_size_stride(arg167_1, (128, ), (1, ))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (256, ), (1, ))
    assert_size_stride(arg172_1, (256, ), (1, ))
    assert_size_stride(arg173_1, (256, ), (1, ))
    assert_size_stride(arg174_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg175_1, (64, ), (1, ))
    assert_size_stride(arg176_1, (64, ), (1, ))
    assert_size_stride(arg177_1, (64, ), (1, ))
    assert_size_stride(arg178_1, (64, ), (1, ))
    assert_size_stride(arg179_1, (64, ), (1, ))
    assert_size_stride(arg180_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg181_1, (256, ), (1, ))
    assert_size_stride(arg182_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (256, ), (1, ))
    assert_size_stride(arg190_1, (256, ), (1, ))
    assert_size_stride(arg191_1, (256, ), (1, ))
    assert_size_stride(arg192_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, ), (1, ))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg204_1, (512, ), (1, ))
    assert_size_stride(arg205_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg206_1, (1024, ), (1, ))
    assert_size_stride(arg207_1, (1024, ), (1, ))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg211_1, (1024, ), (1, ))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg216_1, (256, ), (1, ))
    assert_size_stride(arg217_1, (256, ), (1, ))
    assert_size_stride(arg218_1, (256, ), (1, ))
    assert_size_stride(arg219_1, (256, ), (1, ))
    assert_size_stride(arg220_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg226_1, (128, ), (1, ))
    assert_size_stride(arg227_1, (128, ), (1, ))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (128, ), (1, ))
    assert_size_stride(arg231_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg234_1, (1024, ), (1, ))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (1024, ), (1, ))
    assert_size_stride(arg238_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg239_1, (256, ), (1, ))
    assert_size_stride(arg240_1, (256, ), (1, ))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (256, ), (1, ))
    assert_size_stride(arg243_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (128, ), (1, ))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg262_1, (256, ), (1, ))
    assert_size_stride(arg263_1, (256, ), (1, ))
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (256, ), (1, ))
    assert_size_stride(arg266_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (512, ), (1, ))
    assert_size_stride(arg271_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg272_1, (128, ), (1, ))
    assert_size_stride(arg273_1, (128, ), (1, ))
    assert_size_stride(arg274_1, (128, ), (1, ))
    assert_size_stride(arg275_1, (128, ), (1, ))
    assert_size_stride(arg276_1, (128, ), (1, ))
    assert_size_stride(arg277_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (1024, ), (1, ))
    assert_size_stride(arg283_1, (1024, ), (1, ))
    assert_size_stride(arg284_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg285_1, (256, ), (1, ))
    assert_size_stride(arg286_1, (256, ), (1, ))
    assert_size_stride(arg287_1, (256, ), (1, ))
    assert_size_stride(arg288_1, (256, ), (1, ))
    assert_size_stride(arg289_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg295_1, (128, ), (1, ))
    assert_size_stride(arg296_1, (128, ), (1, ))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (128, ), (1, ))
    assert_size_stride(arg299_1, (128, ), (1, ))
    assert_size_stride(arg300_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg301_1, (512, ), (1, ))
    assert_size_stride(arg302_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, ), (1, ))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg308_1, (256, ), (1, ))
    assert_size_stride(arg309_1, (256, ), (1, ))
    assert_size_stride(arg310_1, (256, ), (1, ))
    assert_size_stride(arg311_1, (256, ), (1, ))
    assert_size_stride(arg312_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg313_1, (512, ), (1, ))
    assert_size_stride(arg314_1, (512, ), (1, ))
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg318_1, (128, ), (1, ))
    assert_size_stride(arg319_1, (128, ), (1, ))
    assert_size_stride(arg320_1, (128, ), (1, ))
    assert_size_stride(arg321_1, (128, ), (1, ))
    assert_size_stride(arg322_1, (128, ), (1, ))
    assert_size_stride(arg323_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1024, ), (1, ))
    assert_size_stride(arg328_1, (1024, ), (1, ))
    assert_size_stride(arg329_1, (1024, ), (1, ))
    assert_size_stride(arg330_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg331_1, (256, ), (1, ))
    assert_size_stride(arg332_1, (256, ), (1, ))
    assert_size_stride(arg333_1, (256, ), (1, ))
    assert_size_stride(arg334_1, (256, ), (1, ))
    assert_size_stride(arg335_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg336_1, (512, ), (1, ))
    assert_size_stride(arg337_1, (512, ), (1, ))
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg341_1, (128, ), (1, ))
    assert_size_stride(arg342_1, (128, ), (1, ))
    assert_size_stride(arg343_1, (128, ), (1, ))
    assert_size_stride(arg344_1, (128, ), (1, ))
    assert_size_stride(arg345_1, (128, ), (1, ))
    assert_size_stride(arg346_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg347_1, (512, ), (1, ))
    assert_size_stride(arg348_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (1024, ), (1, ))
    assert_size_stride(arg353_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg354_1, (256, ), (1, ))
    assert_size_stride(arg355_1, (256, ), (1, ))
    assert_size_stride(arg356_1, (256, ), (1, ))
    assert_size_stride(arg357_1, (256, ), (1, ))
    assert_size_stride(arg358_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg359_1, (512, ), (1, ))
    assert_size_stride(arg360_1, (512, ), (1, ))
    assert_size_stride(arg361_1, (512, ), (1, ))
    assert_size_stride(arg362_1, (512, ), (1, ))
    assert_size_stride(arg363_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg364_1, (128, ), (1, ))
    assert_size_stride(arg365_1, (128, ), (1, ))
    assert_size_stride(arg366_1, (128, ), (1, ))
    assert_size_stride(arg367_1, (128, ), (1, ))
    assert_size_stride(arg368_1, (128, ), (1, ))
    assert_size_stride(arg369_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg370_1, (512, ), (1, ))
    assert_size_stride(arg371_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, ), (1, ))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, ), (1, ))
    assert_size_stride(arg376_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg377_1, (256, ), (1, ))
    assert_size_stride(arg378_1, (256, ), (1, ))
    assert_size_stride(arg379_1, (256, ), (1, ))
    assert_size_stride(arg380_1, (256, ), (1, ))
    assert_size_stride(arg381_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg382_1, (512, ), (1, ))
    assert_size_stride(arg383_1, (512, ), (1, ))
    assert_size_stride(arg384_1, (512, ), (1, ))
    assert_size_stride(arg385_1, (512, ), (1, ))
    assert_size_stride(arg386_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg387_1, (128, ), (1, ))
    assert_size_stride(arg388_1, (128, ), (1, ))
    assert_size_stride(arg389_1, (128, ), (1, ))
    assert_size_stride(arg390_1, (128, ), (1, ))
    assert_size_stride(arg391_1, (128, ), (1, ))
    assert_size_stride(arg392_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg393_1, (512, ), (1, ))
    assert_size_stride(arg394_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, ), (1, ))
    assert_size_stride(arg397_1, (1024, ), (1, ))
    assert_size_stride(arg398_1, (1024, ), (1, ))
    assert_size_stride(arg399_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg400_1, (256, ), (1, ))
    assert_size_stride(arg401_1, (256, ), (1, ))
    assert_size_stride(arg402_1, (256, ), (1, ))
    assert_size_stride(arg403_1, (256, ), (1, ))
    assert_size_stride(arg404_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg405_1, (512, ), (1, ))
    assert_size_stride(arg406_1, (512, ), (1, ))
    assert_size_stride(arg407_1, (512, ), (1, ))
    assert_size_stride(arg408_1, (512, ), (1, ))
    assert_size_stride(arg409_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg410_1, (128, ), (1, ))
    assert_size_stride(arg411_1, (128, ), (1, ))
    assert_size_stride(arg412_1, (128, ), (1, ))
    assert_size_stride(arg413_1, (128, ), (1, ))
    assert_size_stride(arg414_1, (128, ), (1, ))
    assert_size_stride(arg415_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg416_1, (512, ), (1, ))
    assert_size_stride(arg417_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg418_1, (1024, ), (1, ))
    assert_size_stride(arg419_1, (1024, ), (1, ))
    assert_size_stride(arg420_1, (1024, ), (1, ))
    assert_size_stride(arg421_1, (1024, ), (1, ))
    assert_size_stride(arg422_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg423_1, (256, ), (1, ))
    assert_size_stride(arg424_1, (256, ), (1, ))
    assert_size_stride(arg425_1, (256, ), (1, ))
    assert_size_stride(arg426_1, (256, ), (1, ))
    assert_size_stride(arg427_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg428_1, (512, ), (1, ))
    assert_size_stride(arg429_1, (512, ), (1, ))
    assert_size_stride(arg430_1, (512, ), (1, ))
    assert_size_stride(arg431_1, (512, ), (1, ))
    assert_size_stride(arg432_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg433_1, (128, ), (1, ))
    assert_size_stride(arg434_1, (128, ), (1, ))
    assert_size_stride(arg435_1, (128, ), (1, ))
    assert_size_stride(arg436_1, (128, ), (1, ))
    assert_size_stride(arg437_1, (128, ), (1, ))
    assert_size_stride(arg438_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg439_1, (512, ), (1, ))
    assert_size_stride(arg440_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (1024, ), (1, ))
    assert_size_stride(arg443_1, (1024, ), (1, ))
    assert_size_stride(arg444_1, (1024, ), (1, ))
    assert_size_stride(arg445_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg446_1, (256, ), (1, ))
    assert_size_stride(arg447_1, (256, ), (1, ))
    assert_size_stride(arg448_1, (256, ), (1, ))
    assert_size_stride(arg449_1, (256, ), (1, ))
    assert_size_stride(arg450_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg451_1, (512, ), (1, ))
    assert_size_stride(arg452_1, (512, ), (1, ))
    assert_size_stride(arg453_1, (512, ), (1, ))
    assert_size_stride(arg454_1, (512, ), (1, ))
    assert_size_stride(arg455_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg456_1, (128, ), (1, ))
    assert_size_stride(arg457_1, (128, ), (1, ))
    assert_size_stride(arg458_1, (128, ), (1, ))
    assert_size_stride(arg459_1, (128, ), (1, ))
    assert_size_stride(arg460_1, (128, ), (1, ))
    assert_size_stride(arg461_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg462_1, (512, ), (1, ))
    assert_size_stride(arg463_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg464_1, (1024, ), (1, ))
    assert_size_stride(arg465_1, (1024, ), (1, ))
    assert_size_stride(arg466_1, (1024, ), (1, ))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg469_1, (256, ), (1, ))
    assert_size_stride(arg470_1, (256, ), (1, ))
    assert_size_stride(arg471_1, (256, ), (1, ))
    assert_size_stride(arg472_1, (256, ), (1, ))
    assert_size_stride(arg473_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg474_1, (512, ), (1, ))
    assert_size_stride(arg475_1, (512, ), (1, ))
    assert_size_stride(arg476_1, (512, ), (1, ))
    assert_size_stride(arg477_1, (512, ), (1, ))
    assert_size_stride(arg478_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg479_1, (128, ), (1, ))
    assert_size_stride(arg480_1, (128, ), (1, ))
    assert_size_stride(arg481_1, (128, ), (1, ))
    assert_size_stride(arg482_1, (128, ), (1, ))
    assert_size_stride(arg483_1, (128, ), (1, ))
    assert_size_stride(arg484_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg485_1, (512, ), (1, ))
    assert_size_stride(arg486_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg487_1, (1024, ), (1, ))
    assert_size_stride(arg488_1, (1024, ), (1, ))
    assert_size_stride(arg489_1, (1024, ), (1, ))
    assert_size_stride(arg490_1, (1024, ), (1, ))
    assert_size_stride(arg491_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg492_1, (256, ), (1, ))
    assert_size_stride(arg493_1, (256, ), (1, ))
    assert_size_stride(arg494_1, (256, ), (1, ))
    assert_size_stride(arg495_1, (256, ), (1, ))
    assert_size_stride(arg496_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg497_1, (512, ), (1, ))
    assert_size_stride(arg498_1, (512, ), (1, ))
    assert_size_stride(arg499_1, (512, ), (1, ))
    assert_size_stride(arg500_1, (512, ), (1, ))
    assert_size_stride(arg501_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg502_1, (128, ), (1, ))
    assert_size_stride(arg503_1, (128, ), (1, ))
    assert_size_stride(arg504_1, (128, ), (1, ))
    assert_size_stride(arg505_1, (128, ), (1, ))
    assert_size_stride(arg506_1, (128, ), (1, ))
    assert_size_stride(arg507_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg508_1, (512, ), (1, ))
    assert_size_stride(arg509_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg510_1, (1024, ), (1, ))
    assert_size_stride(arg511_1, (1024, ), (1, ))
    assert_size_stride(arg512_1, (1024, ), (1, ))
    assert_size_stride(arg513_1, (1024, ), (1, ))
    assert_size_stride(arg514_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg515_1, (256, ), (1, ))
    assert_size_stride(arg516_1, (256, ), (1, ))
    assert_size_stride(arg517_1, (256, ), (1, ))
    assert_size_stride(arg518_1, (256, ), (1, ))
    assert_size_stride(arg519_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg520_1, (512, ), (1, ))
    assert_size_stride(arg521_1, (512, ), (1, ))
    assert_size_stride(arg522_1, (512, ), (1, ))
    assert_size_stride(arg523_1, (512, ), (1, ))
    assert_size_stride(arg524_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg525_1, (128, ), (1, ))
    assert_size_stride(arg526_1, (128, ), (1, ))
    assert_size_stride(arg527_1, (128, ), (1, ))
    assert_size_stride(arg528_1, (128, ), (1, ))
    assert_size_stride(arg529_1, (128, ), (1, ))
    assert_size_stride(arg530_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg531_1, (512, ), (1, ))
    assert_size_stride(arg532_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg533_1, (1024, ), (1, ))
    assert_size_stride(arg534_1, (1024, ), (1, ))
    assert_size_stride(arg535_1, (1024, ), (1, ))
    assert_size_stride(arg536_1, (1024, ), (1, ))
    assert_size_stride(arg537_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg538_1, (256, ), (1, ))
    assert_size_stride(arg539_1, (256, ), (1, ))
    assert_size_stride(arg540_1, (256, ), (1, ))
    assert_size_stride(arg541_1, (256, ), (1, ))
    assert_size_stride(arg542_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg543_1, (512, ), (1, ))
    assert_size_stride(arg544_1, (512, ), (1, ))
    assert_size_stride(arg545_1, (512, ), (1, ))
    assert_size_stride(arg546_1, (512, ), (1, ))
    assert_size_stride(arg547_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg548_1, (128, ), (1, ))
    assert_size_stride(arg549_1, (128, ), (1, ))
    assert_size_stride(arg550_1, (128, ), (1, ))
    assert_size_stride(arg551_1, (128, ), (1, ))
    assert_size_stride(arg552_1, (128, ), (1, ))
    assert_size_stride(arg553_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg554_1, (512, ), (1, ))
    assert_size_stride(arg555_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg556_1, (1024, ), (1, ))
    assert_size_stride(arg557_1, (1024, ), (1, ))
    assert_size_stride(arg558_1, (1024, ), (1, ))
    assert_size_stride(arg559_1, (1024, ), (1, ))
    assert_size_stride(arg560_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg561_1, (256, ), (1, ))
    assert_size_stride(arg562_1, (256, ), (1, ))
    assert_size_stride(arg563_1, (256, ), (1, ))
    assert_size_stride(arg564_1, (256, ), (1, ))
    assert_size_stride(arg565_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg566_1, (512, ), (1, ))
    assert_size_stride(arg567_1, (512, ), (1, ))
    assert_size_stride(arg568_1, (512, ), (1, ))
    assert_size_stride(arg569_1, (512, ), (1, ))
    assert_size_stride(arg570_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg571_1, (128, ), (1, ))
    assert_size_stride(arg572_1, (128, ), (1, ))
    assert_size_stride(arg573_1, (128, ), (1, ))
    assert_size_stride(arg574_1, (128, ), (1, ))
    assert_size_stride(arg575_1, (128, ), (1, ))
    assert_size_stride(arg576_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg577_1, (512, ), (1, ))
    assert_size_stride(arg578_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg579_1, (1024, ), (1, ))
    assert_size_stride(arg580_1, (1024, ), (1, ))
    assert_size_stride(arg581_1, (1024, ), (1, ))
    assert_size_stride(arg582_1, (1024, ), (1, ))
    assert_size_stride(arg583_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg584_1, (256, ), (1, ))
    assert_size_stride(arg585_1, (256, ), (1, ))
    assert_size_stride(arg586_1, (256, ), (1, ))
    assert_size_stride(arg587_1, (256, ), (1, ))
    assert_size_stride(arg588_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg589_1, (512, ), (1, ))
    assert_size_stride(arg590_1, (512, ), (1, ))
    assert_size_stride(arg591_1, (512, ), (1, ))
    assert_size_stride(arg592_1, (512, ), (1, ))
    assert_size_stride(arg593_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg594_1, (128, ), (1, ))
    assert_size_stride(arg595_1, (128, ), (1, ))
    assert_size_stride(arg596_1, (128, ), (1, ))
    assert_size_stride(arg597_1, (128, ), (1, ))
    assert_size_stride(arg598_1, (128, ), (1, ))
    assert_size_stride(arg599_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg600_1, (512, ), (1, ))
    assert_size_stride(arg601_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg602_1, (1024, ), (1, ))
    assert_size_stride(arg603_1, (1024, ), (1, ))
    assert_size_stride(arg604_1, (1024, ), (1, ))
    assert_size_stride(arg605_1, (1024, ), (1, ))
    assert_size_stride(arg606_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg607_1, (256, ), (1, ))
    assert_size_stride(arg608_1, (256, ), (1, ))
    assert_size_stride(arg609_1, (256, ), (1, ))
    assert_size_stride(arg610_1, (256, ), (1, ))
    assert_size_stride(arg611_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg612_1, (512, ), (1, ))
    assert_size_stride(arg613_1, (512, ), (1, ))
    assert_size_stride(arg614_1, (512, ), (1, ))
    assert_size_stride(arg615_1, (512, ), (1, ))
    assert_size_stride(arg616_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg617_1, (128, ), (1, ))
    assert_size_stride(arg618_1, (128, ), (1, ))
    assert_size_stride(arg619_1, (128, ), (1, ))
    assert_size_stride(arg620_1, (128, ), (1, ))
    assert_size_stride(arg621_1, (128, ), (1, ))
    assert_size_stride(arg622_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg623_1, (512, ), (1, ))
    assert_size_stride(arg624_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg625_1, (1024, ), (1, ))
    assert_size_stride(arg626_1, (1024, ), (1, ))
    assert_size_stride(arg627_1, (1024, ), (1, ))
    assert_size_stride(arg628_1, (1024, ), (1, ))
    assert_size_stride(arg629_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg630_1, (256, ), (1, ))
    assert_size_stride(arg631_1, (256, ), (1, ))
    assert_size_stride(arg632_1, (256, ), (1, ))
    assert_size_stride(arg633_1, (256, ), (1, ))
    assert_size_stride(arg634_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg635_1, (512, ), (1, ))
    assert_size_stride(arg636_1, (512, ), (1, ))
    assert_size_stride(arg637_1, (512, ), (1, ))
    assert_size_stride(arg638_1, (512, ), (1, ))
    assert_size_stride(arg639_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg640_1, (128, ), (1, ))
    assert_size_stride(arg641_1, (128, ), (1, ))
    assert_size_stride(arg642_1, (128, ), (1, ))
    assert_size_stride(arg643_1, (128, ), (1, ))
    assert_size_stride(arg644_1, (128, ), (1, ))
    assert_size_stride(arg645_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg646_1, (512, ), (1, ))
    assert_size_stride(arg647_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg648_1, (1024, ), (1, ))
    assert_size_stride(arg649_1, (1024, ), (1, ))
    assert_size_stride(arg650_1, (1024, ), (1, ))
    assert_size_stride(arg651_1, (1024, ), (1, ))
    assert_size_stride(arg652_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg653_1, (256, ), (1, ))
    assert_size_stride(arg654_1, (256, ), (1, ))
    assert_size_stride(arg655_1, (256, ), (1, ))
    assert_size_stride(arg656_1, (256, ), (1, ))
    assert_size_stride(arg657_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg658_1, (512, ), (1, ))
    assert_size_stride(arg659_1, (512, ), (1, ))
    assert_size_stride(arg660_1, (512, ), (1, ))
    assert_size_stride(arg661_1, (512, ), (1, ))
    assert_size_stride(arg662_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg663_1, (128, ), (1, ))
    assert_size_stride(arg664_1, (128, ), (1, ))
    assert_size_stride(arg665_1, (128, ), (1, ))
    assert_size_stride(arg666_1, (128, ), (1, ))
    assert_size_stride(arg667_1, (128, ), (1, ))
    assert_size_stride(arg668_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg669_1, (512, ), (1, ))
    assert_size_stride(arg670_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg671_1, (1024, ), (1, ))
    assert_size_stride(arg672_1, (1024, ), (1, ))
    assert_size_stride(arg673_1, (1024, ), (1, ))
    assert_size_stride(arg674_1, (1024, ), (1, ))
    assert_size_stride(arg675_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg676_1, (256, ), (1, ))
    assert_size_stride(arg677_1, (256, ), (1, ))
    assert_size_stride(arg678_1, (256, ), (1, ))
    assert_size_stride(arg679_1, (256, ), (1, ))
    assert_size_stride(arg680_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg681_1, (512, ), (1, ))
    assert_size_stride(arg682_1, (512, ), (1, ))
    assert_size_stride(arg683_1, (512, ), (1, ))
    assert_size_stride(arg684_1, (512, ), (1, ))
    assert_size_stride(arg685_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg686_1, (128, ), (1, ))
    assert_size_stride(arg687_1, (128, ), (1, ))
    assert_size_stride(arg688_1, (128, ), (1, ))
    assert_size_stride(arg689_1, (128, ), (1, ))
    assert_size_stride(arg690_1, (128, ), (1, ))
    assert_size_stride(arg691_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg692_1, (512, ), (1, ))
    assert_size_stride(arg693_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg694_1, (1024, ), (1, ))
    assert_size_stride(arg695_1, (1024, ), (1, ))
    assert_size_stride(arg696_1, (1024, ), (1, ))
    assert_size_stride(arg697_1, (1024, ), (1, ))
    assert_size_stride(arg698_1, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg699_1, (256, ), (1, ))
    assert_size_stride(arg700_1, (256, ), (1, ))
    assert_size_stride(arg701_1, (256, ), (1, ))
    assert_size_stride(arg702_1, (256, ), (1, ))
    assert_size_stride(arg703_1, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg704_1, (512, ), (1, ))
    assert_size_stride(arg705_1, (512, ), (1, ))
    assert_size_stride(arg706_1, (512, ), (1, ))
    assert_size_stride(arg707_1, (512, ), (1, ))
    assert_size_stride(arg708_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg709_1, (128, ), (1, ))
    assert_size_stride(arg710_1, (128, ), (1, ))
    assert_size_stride(arg711_1, (128, ), (1, ))
    assert_size_stride(arg712_1, (128, ), (1, ))
    assert_size_stride(arg713_1, (128, ), (1, ))
    assert_size_stride(arg714_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg715_1, (512, ), (1, ))
    assert_size_stride(arg716_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg717_1, (1024, ), (1, ))
    assert_size_stride(arg718_1, (1024, ), (1, ))
    assert_size_stride(arg719_1, (1024, ), (1, ))
    assert_size_stride(arg720_1, (1024, ), (1, ))
    assert_size_stride(arg721_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg722_1, (512, ), (1, ))
    assert_size_stride(arg723_1, (512, ), (1, ))
    assert_size_stride(arg724_1, (512, ), (1, ))
    assert_size_stride(arg725_1, (512, ), (1, ))
    assert_size_stride(arg726_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg727_1, (1024, ), (1, ))
    assert_size_stride(arg728_1, (1024, ), (1, ))
    assert_size_stride(arg729_1, (1024, ), (1, ))
    assert_size_stride(arg730_1, (1024, ), (1, ))
    assert_size_stride(arg731_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg732_1, (256, ), (1, ))
    assert_size_stride(arg733_1, (256, ), (1, ))
    assert_size_stride(arg734_1, (256, ), (1, ))
    assert_size_stride(arg735_1, (256, ), (1, ))
    assert_size_stride(arg736_1, (256, ), (1, ))
    assert_size_stride(arg737_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg738_1, (1024, ), (1, ))
    assert_size_stride(arg739_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg740_1, (2048, ), (1, ))
    assert_size_stride(arg741_1, (2048, ), (1, ))
    assert_size_stride(arg742_1, (2048, ), (1, ))
    assert_size_stride(arg743_1, (2048, ), (1, ))
    assert_size_stride(arg744_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg745_1, (2048, ), (1, ))
    assert_size_stride(arg746_1, (2048, ), (1, ))
    assert_size_stride(arg747_1, (2048, ), (1, ))
    assert_size_stride(arg748_1, (2048, ), (1, ))
    assert_size_stride(arg749_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg750_1, (512, ), (1, ))
    assert_size_stride(arg751_1, (512, ), (1, ))
    assert_size_stride(arg752_1, (512, ), (1, ))
    assert_size_stride(arg753_1, (512, ), (1, ))
    assert_size_stride(arg754_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg755_1, (1024, ), (1, ))
    assert_size_stride(arg756_1, (1024, ), (1, ))
    assert_size_stride(arg757_1, (1024, ), (1, ))
    assert_size_stride(arg758_1, (1024, ), (1, ))
    assert_size_stride(arg759_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg760_1, (256, ), (1, ))
    assert_size_stride(arg761_1, (256, ), (1, ))
    assert_size_stride(arg762_1, (256, ), (1, ))
    assert_size_stride(arg763_1, (256, ), (1, ))
    assert_size_stride(arg764_1, (256, ), (1, ))
    assert_size_stride(arg765_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg766_1, (1024, ), (1, ))
    assert_size_stride(arg767_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg768_1, (2048, ), (1, ))
    assert_size_stride(arg769_1, (2048, ), (1, ))
    assert_size_stride(arg770_1, (2048, ), (1, ))
    assert_size_stride(arg771_1, (2048, ), (1, ))
    assert_size_stride(arg772_1, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg773_1, (512, ), (1, ))
    assert_size_stride(arg774_1, (512, ), (1, ))
    assert_size_stride(arg775_1, (512, ), (1, ))
    assert_size_stride(arg776_1, (512, ), (1, ))
    assert_size_stride(arg777_1, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg778_1, (1024, ), (1, ))
    assert_size_stride(arg779_1, (1024, ), (1, ))
    assert_size_stride(arg780_1, (1024, ), (1, ))
    assert_size_stride(arg781_1, (1024, ), (1, ))
    assert_size_stride(arg782_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg783_1, (256, ), (1, ))
    assert_size_stride(arg784_1, (256, ), (1, ))
    assert_size_stride(arg785_1, (256, ), (1, ))
    assert_size_stride(arg786_1, (256, ), (1, ))
    assert_size_stride(arg787_1, (256, ), (1, ))
    assert_size_stride(arg788_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg789_1, (1024, ), (1, ))
    assert_size_stride(arg790_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg791_1, (2048, ), (1, ))
    assert_size_stride(arg792_1, (2048, ), (1, ))
    assert_size_stride(arg793_1, (2048, ), (1, ))
    assert_size_stride(arg794_1, (2048, ), (1, ))
    assert_size_stride(arg795_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg796_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg1_1, arg0_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg0_1
        del arg1_1
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg2_1, arg3_1, arg4_1, arg5_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        # Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf2 = extern_kernels.convolution(buf1, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del arg6_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf3, arg7_1, arg8_1, arg9_1, arg10_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        # Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg11_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 128, 128, 128), (2097152, 16384, 128, 1))
        del arg11_1
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf5, arg12_1, arg13_1, arg14_1, arg15_1, 16777216, grid=grid(16777216), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        buf6 = empty_strided_cuda((8, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_237, x_238, x_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_2.run(buf5, buf6, 4194304, grid=grid(4194304), stream=stream0)
        del buf5
        # Topologically Sorted Source Nodes: [out_300], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg16_1
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [out_301, out_302, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf8, arg17_1, arg18_1, arg19_1, arg20_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        # Topologically Sorted Source Nodes: [out_301, out_302, x_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf9, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg21_1
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [x_241, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf10, arg22_1, arg23_1, arg24_1, arg25_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        buf12 = empty_strided_cuda((8, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_gap_165, x_gap_166, x_gap_167], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_red_fused_convolution_mean_sum_5.run(buf10, buf12, 512, 4096, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_165, x_gap_166, x_gap_167], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg26_1
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_gap_165, x_gap_166, x_gap_167, x_gap_168, x_gap_169, x_attn_66], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6.run(buf14, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, 256, grid=grid(256), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del arg31_1
        # Topologically Sorted Source Nodes: [x_gap_165, x_gap_166, x_gap_167, x_gap_168, x_gap_169, x_attn_66], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf15 = extern_kernels.convolution(buf14, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg32_1
        del buf14
        buf16 = empty_strided_cuda((8, 2, 1, 64), (128, 64, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_245], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_7.run(buf15, arg33_1, buf16, 1024, grid=grid(1024), stream=stream0)
        del arg33_1
        del buf15
        buf17 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [mul_33, out_303, out_305], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_8.run(buf10, buf16, buf17, 2097152, grid=grid(2097152), stream=stream0)
        del buf10
        # Topologically Sorted Source Nodes: [mul_33, out_303, out_305], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg34_1
        del buf17
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf6, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg39_1
        del buf6
        buf20 = buf18; del buf18  # reuse
        buf21 = reinterpret_tensor(buf3, (8, 256, 64, 64), (1048576, 4096, 64, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_306, input_27, out_307, out_308], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf20, arg35_1, arg36_1, arg37_1, arg38_1, buf19, arg40_1, arg41_1, arg42_1, arg43_1, buf21, 8388608, grid=grid(8388608), stream=stream0)
        del arg35_1
        del arg36_1
        del arg37_1
        del arg38_1
        del arg40_1
        del arg41_1
        del arg42_1
        del arg43_1
        del buf19
        del buf20
        # Topologically Sorted Source Nodes: [out_309], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg44_1
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [out_310, out_311, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf23, arg45_1, arg46_1, arg47_1, arg48_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        del arg48_1
        # Topologically Sorted Source Nodes: [out_310, out_311, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf24, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg49_1
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf25, arg50_1, arg51_1, arg52_1, arg53_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg50_1
        del arg51_1
        del arg52_1
        del arg53_1
        buf27 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_gap_170, x_gap_171, x_gap_172], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_red_fused_convolution_mean_sum_5.run(buf25, buf27, 512, 4096, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_170, x_gap_171, x_gap_172], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg54_1
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_gap_170, x_gap_171, x_gap_172, x_gap_173, x_gap_174, x_attn_68], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6.run(buf29, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, 256, grid=grid(256), stream=stream0)
        del arg55_1
        del arg56_1
        del arg57_1
        del arg58_1
        del arg59_1
        # Topologically Sorted Source Nodes: [x_gap_170, x_gap_171, x_gap_172, x_gap_173, x_gap_174, x_attn_68], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg60_1
        del buf29
        buf31 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_252], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_7.run(buf30, arg61_1, buf31, 1024, grid=grid(1024), stream=stream0)
        del arg61_1
        del buf30
        buf32 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [mul_34, out_312, out_314], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_8.run(buf25, buf31, buf32, 2097152, grid=grid(2097152), stream=stream0)
        del buf25
        # Topologically Sorted Source Nodes: [mul_34, out_312, out_314], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg62_1
        del buf32
        buf34 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [out_315, out_316, out_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf34, buf33, arg63_1, arg64_1, arg65_1, arg66_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg63_1
        del arg64_1
        del arg65_1
        del arg66_1
        del buf33
        # Topologically Sorted Source Nodes: [out_318], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg67_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 64, 64, 64), (262144, 4096, 64, 1))
        del arg67_1
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [out_319, out_320, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf36, arg68_1, arg69_1, arg70_1, arg71_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg68_1
        del arg69_1
        del arg70_1
        del arg71_1
        # Topologically Sorted Source Nodes: [out_319, out_320, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf37, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg72_1
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_255, x_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf38, arg73_1, arg74_1, arg75_1, arg76_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg73_1
        del arg74_1
        del arg75_1
        del arg76_1
        buf40 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_gap_175, x_gap_176, x_gap_177], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_red_fused_convolution_mean_sum_5.run(buf38, buf40, 512, 4096, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_175, x_gap_176, x_gap_177], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg77_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_gap_175, x_gap_176, x_gap_177, x_gap_178, x_gap_179, x_attn_70], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_6.run(buf42, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, 256, grid=grid(256), stream=stream0)
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        del arg82_1
        # Topologically Sorted Source Nodes: [x_gap_175, x_gap_176, x_gap_177, x_gap_178, x_gap_179, x_attn_70], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg83_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg83_1
        del buf42
        buf44 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_259], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_7.run(buf43, arg84_1, buf44, 1024, grid=grid(1024), stream=stream0)
        del arg84_1
        del buf43
        buf45 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [mul_35, out_321, out_323], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_8.run(buf38, buf44, buf45, 2097152, grid=grid(2097152), stream=stream0)
        del buf38
        # Topologically Sorted Source Nodes: [mul_35, out_321, out_323], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg85_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg85_1
        buf47 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [out_324, out_325, out_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf47, buf46, arg86_1, arg87_1, arg88_1, arg89_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg86_1
        del arg87_1
        del arg88_1
        del arg89_1
        del buf46
        # Topologically Sorted Source Nodes: [out_327], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del arg90_1
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [out_328, out_329, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf49, arg91_1, arg92_1, arg93_1, arg94_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg91_1
        del arg92_1
        del arg93_1
        del arg94_1
        # Topologically Sorted Source Nodes: [out_328, out_329, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg95_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf50, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del arg95_1
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_262, x_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf51, arg96_1, arg97_1, arg98_1, arg99_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg96_1
        del arg97_1
        del arg98_1
        del arg99_1
        buf53 = reinterpret_tensor(buf44, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_gap_180, x_gap_181, x_gap_182], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_red_fused_convolution_mean_sum_12.run(buf51, buf53, 1024, 4096, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_180, x_gap_181, x_gap_182], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg100_1
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_gap_180, x_gap_181, x_gap_182, x_gap_183, x_gap_184, x_attn_72], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf55, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, 512, grid=grid(512), stream=stream0)
        del arg101_1
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        # Topologically Sorted Source Nodes: [x_gap_180, x_gap_181, x_gap_182, x_gap_183, x_gap_184, x_attn_72], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg106_1
        del buf55
        buf57 = empty_strided_cuda((8, 2, 1, 128), (256, 128, 2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_266], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf56, arg107_1, buf57, 2048, grid=grid(2048), stream=stream0)
        del arg107_1
        del buf56
        buf58 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [mul_36, out_330], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_15.run(buf51, buf57, buf58, 4194304, grid=grid(4194304), stream=stream0)
        del buf51
        buf59 = empty_strided_cuda((8, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_36, out_330, out_332], Original ATen: [aten.mul, aten.sum, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_mul_sum_16.run(buf58, buf59, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [out_333], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg108_1
        del buf59
        buf61 = reinterpret_tensor(buf45, (8, 256, 32, 32), (262144, 1024, 32, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_17.run(buf47, buf61, 2097152, grid=grid(2097152), stream=stream0)
        del buf47
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg113_1
        del buf61
        buf63 = buf60; del buf60  # reuse
        buf64 = reinterpret_tensor(buf58, (8, 512, 32, 32), (524288, 1024, 32, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [out_334, input_30, out_335, out_336], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf63, arg109_1, arg110_1, arg111_1, arg112_1, buf62, arg114_1, arg115_1, arg116_1, arg117_1, buf64, 4194304, grid=grid(4194304), stream=stream0)
        del arg109_1
        del arg110_1
        del arg111_1
        del arg112_1
        del arg114_1
        del arg115_1
        del arg116_1
        del arg117_1
        del buf62
        del buf63
        # Topologically Sorted Source Nodes: [out_337], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg118_1
        buf66 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [out_338, out_339, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf66, arg119_1, arg120_1, arg121_1, arg122_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg119_1
        del arg120_1
        del arg121_1
        del arg122_1
        # Topologically Sorted Source Nodes: [out_338, out_339, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf67, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg123_1
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_269, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf68, arg124_1, arg125_1, arg126_1, arg127_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg124_1
        del arg125_1
        del arg126_1
        del arg127_1
        buf70 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_gap_185, x_gap_186, x_gap_187], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_21.run(buf68, buf70, 1024, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_185, x_gap_186, x_gap_187], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf71 = extern_kernels.convolution(buf70, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg128_1
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_gap_185, x_gap_186, x_gap_187, x_gap_188, x_gap_189, x_attn_74], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf72, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, 512, grid=grid(512), stream=stream0)
        del arg129_1
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        # Topologically Sorted Source Nodes: [x_gap_185, x_gap_186, x_gap_187, x_gap_188, x_gap_189, x_attn_74], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf73 = extern_kernels.convolution(buf72, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg134_1
        del buf72
        buf74 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [x_273], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf73, arg135_1, buf74, 2048, grid=grid(2048), stream=stream0)
        del arg135_1
        del buf73
        buf75 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [mul_37, out_340, out_342], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_22.run(buf68, buf74, buf75, 1048576, grid=grid(1048576), stream=stream0)
        del buf68
        # Topologically Sorted Source Nodes: [mul_37, out_340, out_342], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg136_1
        del buf75
        buf77 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [out_343, out_344, out_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf77, buf76, arg137_1, arg138_1, arg139_1, arg140_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf76
        # Topologically Sorted Source Nodes: [out_346], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg141_1
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_347, out_348, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf79, arg142_1, arg143_1, arg144_1, arg145_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        # Topologically Sorted Source Nodes: [out_347, out_348, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg146_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf80, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg146_1
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_276, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf81, arg147_1, arg148_1, arg149_1, arg150_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf83 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_gap_190, x_gap_191, x_gap_192], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_21.run(buf81, buf83, 1024, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_190, x_gap_191, x_gap_192], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg151_1
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_gap_190, x_gap_191, x_gap_192, x_gap_193, x_gap_194, x_attn_76], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf85, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, 512, grid=grid(512), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del arg156_1
        # Topologically Sorted Source Nodes: [x_gap_190, x_gap_191, x_gap_192, x_gap_193, x_gap_194, x_attn_76], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg157_1
        del buf85
        buf87 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf86, arg158_1, buf87, 2048, grid=grid(2048), stream=stream0)
        del arg158_1
        del buf86
        buf88 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [mul_38, out_349, out_351], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_22.run(buf81, buf87, buf88, 1048576, grid=grid(1048576), stream=stream0)
        del buf81
        # Topologically Sorted Source Nodes: [mul_38, out_349, out_351], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg159_1
        del buf88
        buf90 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [out_352, out_353, out_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf90, buf89, arg160_1, arg161_1, arg162_1, arg163_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg160_1
        del arg161_1
        del arg162_1
        del arg163_1
        del buf89
        # Topologically Sorted Source Nodes: [out_355], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 128, 32, 32), (131072, 1024, 32, 1))
        del arg164_1
        buf92 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [out_356, out_357, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(buf92, arg165_1, arg166_1, arg167_1, arg168_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg165_1
        del arg166_1
        del arg167_1
        del arg168_1
        # Topologically Sorted Source Nodes: [out_356, out_357, x_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg169_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf93, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg169_1
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf94, arg170_1, arg171_1, arg172_1, arg173_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg170_1
        del arg171_1
        del arg172_1
        del arg173_1
        buf96 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_gap_195, x_gap_196, x_gap_197], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_21.run(buf94, buf96, 1024, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_195, x_gap_196, x_gap_197], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg174_1
        del buf96
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_gap_195, x_gap_196, x_gap_197, x_gap_198, x_gap_199, x_attn_78], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_13.run(buf98, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, 512, grid=grid(512), stream=stream0)
        del arg175_1
        del arg176_1
        del arg177_1
        del arg178_1
        del arg179_1
        # Topologically Sorted Source Nodes: [x_gap_195, x_gap_196, x_gap_197, x_gap_198, x_gap_199, x_attn_78], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf99 = extern_kernels.convolution(buf98, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg180_1
        del buf98
        buf100 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_287], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_14.run(buf99, arg181_1, buf100, 2048, grid=grid(2048), stream=stream0)
        del arg181_1
        del buf99
        buf101 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [mul_39, out_358, out_360], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_22.run(buf94, buf100, buf101, 1048576, grid=grid(1048576), stream=stream0)
        del buf94
        # Topologically Sorted Source Nodes: [mul_39, out_358, out_360], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg182_1
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [out_361, out_362, out_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf103, arg183_1, arg184_1, arg185_1, arg186_1, buf90, 4194304, grid=grid(4194304), stream=stream0)
        del arg183_1
        del arg184_1
        del arg185_1
        del arg186_1
        del buf90
        # Topologically Sorted Source Nodes: [out_364], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del arg187_1
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [out_365, out_366, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf105, arg188_1, arg189_1, arg190_1, arg191_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg188_1
        del arg189_1
        del arg190_1
        del arg191_1
        # Topologically Sorted Source Nodes: [out_365, out_366, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg192_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf106, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del arg192_1
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [x_290, x_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf107, arg193_1, arg194_1, arg195_1, arg196_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg193_1
        del arg194_1
        del arg195_1
        del arg196_1
        buf109 = reinterpret_tensor(buf100, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_gap_200, x_gap_201, x_gap_202], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_26.run(buf107, buf109, 2048, 1024, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_200, x_gap_201, x_gap_202], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg197_1
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_gap_200, x_gap_201, x_gap_202, x_gap_203, x_gap_204, x_attn_80], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf111, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, 1024, grid=grid(1024), stream=stream0)
        del arg198_1
        del arg199_1
        del arg200_1
        del arg201_1
        del arg202_1
        # Topologically Sorted Source Nodes: [x_gap_200, x_gap_201, x_gap_202, x_gap_203, x_gap_204, x_attn_80], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf112 = extern_kernels.convolution(buf111, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg203_1
        del buf111
        buf113 = empty_strided_cuda((8, 2, 1, 256), (512, 256, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf112, arg204_1, buf113, 4096, grid=grid(4096), stream=stream0)
        del arg204_1
        del buf112
        buf114 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [mul_40, out_367], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_29.run(buf107, buf113, buf114, 2097152, grid=grid(2097152), stream=stream0)
        del buf107
        buf115 = empty_strided_cuda((8, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_40, out_367, out_369], Original ATen: [aten.mul, aten.sum, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_mul_sum_30.run(buf114, buf115, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [out_370], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg205_1
        del buf115
        buf117 = reinterpret_tensor(buf101, (8, 512, 16, 16), (131072, 256, 16, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_31.run(buf103, buf117, 1048576, grid=grid(1048576), stream=stream0)
        del buf103
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg210_1
        del buf117
        buf119 = buf116; del buf116  # reuse
        buf120 = reinterpret_tensor(buf114, (8, 1024, 16, 16), (262144, 256, 16, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [out_371, input_33, out_372, out_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_32.run(buf119, arg206_1, arg207_1, arg208_1, arg209_1, buf118, arg211_1, arg212_1, arg213_1, arg214_1, buf120, 2097152, grid=grid(2097152), stream=stream0)
        del arg206_1
        del arg207_1
        del arg208_1
        del arg209_1
        del arg211_1
        del arg212_1
        del arg213_1
        del arg214_1
        del buf118
        del buf119
        # Topologically Sorted Source Nodes: [out_374], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg215_1
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [out_375, out_376, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf122, arg216_1, arg217_1, arg218_1, arg219_1, 524288, grid=grid(524288), stream=stream0)
        del arg216_1
        del arg217_1
        del arg218_1
        del arg219_1
        # Topologically Sorted Source Nodes: [out_375, out_376, x_296], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg220_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf123, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg220_1
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_297, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf124, arg221_1, arg222_1, arg223_1, arg224_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg221_1
        del arg222_1
        del arg223_1
        del arg224_1
        buf126 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_gap_205, x_gap_206, x_gap_207], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf124, buf126, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_205, x_gap_206, x_gap_207], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf127 = extern_kernels.convolution(buf126, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg225_1
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [x_gap_205, x_gap_206, x_gap_207, x_gap_208, x_gap_209, x_attn_82], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf128, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, 1024, grid=grid(1024), stream=stream0)
        del arg226_1
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        # Topologically Sorted Source Nodes: [x_gap_205, x_gap_206, x_gap_207, x_gap_208, x_gap_209, x_attn_82], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf129 = extern_kernels.convolution(buf128, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg231_1
        del buf128
        buf130 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_301], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf129, arg232_1, buf130, 4096, grid=grid(4096), stream=stream0)
        del arg232_1
        del buf129
        buf131 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [mul_41, out_377, out_379], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf124, buf130, buf131, 524288, grid=grid(524288), stream=stream0)
        del buf124
        # Topologically Sorted Source Nodes: [mul_41, out_377, out_379], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg233_1
        del buf131
        buf133 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [out_380, out_381, out_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf133, buf132, arg234_1, arg235_1, arg236_1, arg237_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg234_1
        del arg235_1
        del arg236_1
        del arg237_1
        del buf132
        # Topologically Sorted Source Nodes: [out_383], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg238_1
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [out_384, out_385, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf135, arg239_1, arg240_1, arg241_1, arg242_1, 524288, grid=grid(524288), stream=stream0)
        del arg239_1
        del arg240_1
        del arg241_1
        del arg242_1
        # Topologically Sorted Source Nodes: [out_384, out_385, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg243_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf136, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg243_1
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_304, x_305], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf137, arg244_1, arg245_1, arg246_1, arg247_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg244_1
        del arg245_1
        del arg246_1
        del arg247_1
        buf139 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_gap_210, x_gap_211, x_gap_212], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf137, buf139, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_210, x_gap_211, x_gap_212], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf140 = extern_kernels.convolution(buf139, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg248_1
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_gap_210, x_gap_211, x_gap_212, x_gap_213, x_gap_214, x_attn_84], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf141, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, 1024, grid=grid(1024), stream=stream0)
        del arg249_1
        del arg250_1
        del arg251_1
        del arg252_1
        del arg253_1
        # Topologically Sorted Source Nodes: [x_gap_210, x_gap_211, x_gap_212, x_gap_213, x_gap_214, x_attn_84], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf142 = extern_kernels.convolution(buf141, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg254_1
        del buf141
        buf143 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf142, arg255_1, buf143, 4096, grid=grid(4096), stream=stream0)
        del arg255_1
        del buf142
        buf144 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [mul_42, out_386, out_388], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf137, buf143, buf144, 524288, grid=grid(524288), stream=stream0)
        del buf137
        # Topologically Sorted Source Nodes: [mul_42, out_386, out_388], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg256_1
        del buf144
        buf146 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [out_389, out_390, out_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf146, buf145, arg257_1, arg258_1, arg259_1, arg260_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        del buf145
        # Topologically Sorted Source Nodes: [out_392], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg261_1
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [out_393, out_394, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf148, arg262_1, arg263_1, arg264_1, arg265_1, 524288, grid=grid(524288), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        # Topologically Sorted Source Nodes: [out_393, out_394, x_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg266_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf149, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg266_1
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_311, x_312], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf150, arg267_1, arg268_1, arg269_1, arg270_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        buf152 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_gap_215, x_gap_216, x_gap_217], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf150, buf152, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_215, x_gap_216, x_gap_217], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf153 = extern_kernels.convolution(buf152, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg271_1
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_gap_215, x_gap_216, x_gap_217, x_gap_218, x_gap_219, x_attn_86], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf154, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, 1024, grid=grid(1024), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del arg276_1
        # Topologically Sorted Source Nodes: [x_gap_215, x_gap_216, x_gap_217, x_gap_218, x_gap_219, x_attn_86], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf155 = extern_kernels.convolution(buf154, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg277_1
        del buf154
        buf156 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_315], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf155, arg278_1, buf156, 4096, grid=grid(4096), stream=stream0)
        del arg278_1
        del buf155
        buf157 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [mul_43, out_395, out_397], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf150, buf156, buf157, 524288, grid=grid(524288), stream=stream0)
        del buf150
        # Topologically Sorted Source Nodes: [mul_43, out_395, out_397], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg279_1
        del buf157
        buf159 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [out_398, out_399, out_400], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf159, buf158, arg280_1, arg281_1, arg282_1, arg283_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg280_1
        del arg281_1
        del arg282_1
        del arg283_1
        del buf158
        # Topologically Sorted Source Nodes: [out_401], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg284_1
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [out_402, out_403, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf161, arg285_1, arg286_1, arg287_1, arg288_1, 524288, grid=grid(524288), stream=stream0)
        del arg285_1
        del arg286_1
        del arg287_1
        del arg288_1
        # Topologically Sorted Source Nodes: [out_402, out_403, x_317], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf162 = extern_kernels.convolution(buf161, arg289_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf162, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg289_1
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [x_318, x_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf163, arg290_1, arg291_1, arg292_1, arg293_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg290_1
        del arg291_1
        del arg292_1
        del arg293_1
        buf165 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_gap_220, x_gap_221, x_gap_222], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf163, buf165, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_220, x_gap_221, x_gap_222], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf166 = extern_kernels.convolution(buf165, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg294_1
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [x_gap_220, x_gap_221, x_gap_222, x_gap_223, x_gap_224, x_attn_88], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf167, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, 1024, grid=grid(1024), stream=stream0)
        del arg295_1
        del arg296_1
        del arg297_1
        del arg298_1
        del arg299_1
        # Topologically Sorted Source Nodes: [x_gap_220, x_gap_221, x_gap_222, x_gap_223, x_gap_224, x_attn_88], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf168 = extern_kernels.convolution(buf167, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg300_1
        del buf167
        buf169 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [x_322], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf168, arg301_1, buf169, 4096, grid=grid(4096), stream=stream0)
        del arg301_1
        del buf168
        buf170 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [mul_44, out_404, out_406], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf163, buf169, buf170, 524288, grid=grid(524288), stream=stream0)
        del buf163
        # Topologically Sorted Source Nodes: [mul_44, out_404, out_406], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg302_1
        del buf170
        buf172 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [out_407, out_408, out_409], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf172, buf171, arg303_1, arg304_1, arg305_1, arg306_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg303_1
        del arg304_1
        del arg305_1
        del arg306_1
        del buf171
        # Topologically Sorted Source Nodes: [out_410], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, arg307_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg307_1
        buf174 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [out_411, out_412, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf174, arg308_1, arg309_1, arg310_1, arg311_1, 524288, grid=grid(524288), stream=stream0)
        del arg308_1
        del arg309_1
        del arg310_1
        del arg311_1
        # Topologically Sorted Source Nodes: [out_411, out_412, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf175 = extern_kernels.convolution(buf174, arg312_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf175, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg312_1
        buf176 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_325, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf176, arg313_1, arg314_1, arg315_1, arg316_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg313_1
        del arg314_1
        del arg315_1
        del arg316_1
        buf178 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_gap_225, x_gap_226, x_gap_227], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf176, buf178, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_225, x_gap_226, x_gap_227], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg317_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg317_1
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_gap_225, x_gap_226, x_gap_227, x_gap_228, x_gap_229, x_attn_90], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf180, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, 1024, grid=grid(1024), stream=stream0)
        del arg318_1
        del arg319_1
        del arg320_1
        del arg321_1
        del arg322_1
        # Topologically Sorted Source Nodes: [x_gap_225, x_gap_226, x_gap_227, x_gap_228, x_gap_229, x_attn_90], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf181 = extern_kernels.convolution(buf180, arg323_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg323_1
        del buf180
        buf182 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_329], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf181, arg324_1, buf182, 4096, grid=grid(4096), stream=stream0)
        del arg324_1
        del buf181
        buf183 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [mul_45, out_413, out_415], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf176, buf182, buf183, 524288, grid=grid(524288), stream=stream0)
        del buf176
        # Topologically Sorted Source Nodes: [mul_45, out_413, out_415], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg325_1
        del buf183
        buf185 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [out_416, out_417, out_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf185, buf184, arg326_1, arg327_1, arg328_1, arg329_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg326_1
        del arg327_1
        del arg328_1
        del arg329_1
        del buf184
        # Topologically Sorted Source Nodes: [out_419], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg330_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg330_1
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [out_420, out_421, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf187, arg331_1, arg332_1, arg333_1, arg334_1, 524288, grid=grid(524288), stream=stream0)
        del arg331_1
        del arg332_1
        del arg333_1
        del arg334_1
        # Topologically Sorted Source Nodes: [out_420, out_421, x_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg335_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf188, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg335_1
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_332, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf189, arg336_1, arg337_1, arg338_1, arg339_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg336_1
        del arg337_1
        del arg338_1
        del arg339_1
        buf191 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_gap_230, x_gap_231, x_gap_232], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf189, buf191, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_230, x_gap_231, x_gap_232], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf192 = extern_kernels.convolution(buf191, arg340_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg340_1
        buf193 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_gap_230, x_gap_231, x_gap_232, x_gap_233, x_gap_234, x_attn_92], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf193, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, 1024, grid=grid(1024), stream=stream0)
        del arg341_1
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        # Topologically Sorted Source Nodes: [x_gap_230, x_gap_231, x_gap_232, x_gap_233, x_gap_234, x_attn_92], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf194 = extern_kernels.convolution(buf193, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg346_1
        del buf193
        buf195 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_336], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf194, arg347_1, buf195, 4096, grid=grid(4096), stream=stream0)
        del arg347_1
        del buf194
        buf196 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [mul_46, out_422, out_424], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf189, buf195, buf196, 524288, grid=grid(524288), stream=stream0)
        del buf189
        # Topologically Sorted Source Nodes: [mul_46, out_422, out_424], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg348_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg348_1
        del buf196
        buf198 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [out_425, out_426, out_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf198, buf197, arg349_1, arg350_1, arg351_1, arg352_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg349_1
        del arg350_1
        del arg351_1
        del arg352_1
        del buf197
        # Topologically Sorted Source Nodes: [out_428], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg353_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg353_1
        buf200 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [out_429, out_430, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf200, arg354_1, arg355_1, arg356_1, arg357_1, 524288, grid=grid(524288), stream=stream0)
        del arg354_1
        del arg355_1
        del arg356_1
        del arg357_1
        # Topologically Sorted Source Nodes: [out_429, out_430, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg358_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf201, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg358_1
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_339, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf202, arg359_1, arg360_1, arg361_1, arg362_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg359_1
        del arg360_1
        del arg361_1
        del arg362_1
        buf204 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_gap_235, x_gap_236, x_gap_237], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf202, buf204, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_235, x_gap_236, x_gap_237], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf205 = extern_kernels.convolution(buf204, arg363_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg363_1
        buf206 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [x_gap_235, x_gap_236, x_gap_237, x_gap_238, x_gap_239, x_attn_94], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf206, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, 1024, grid=grid(1024), stream=stream0)
        del arg364_1
        del arg365_1
        del arg366_1
        del arg367_1
        del arg368_1
        # Topologically Sorted Source Nodes: [x_gap_235, x_gap_236, x_gap_237, x_gap_238, x_gap_239, x_attn_94], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf207 = extern_kernels.convolution(buf206, arg369_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg369_1
        del buf206
        buf208 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_343], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf207, arg370_1, buf208, 4096, grid=grid(4096), stream=stream0)
        del arg370_1
        del buf207
        buf209 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [mul_47, out_431, out_433], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf202, buf208, buf209, 524288, grid=grid(524288), stream=stream0)
        del buf202
        # Topologically Sorted Source Nodes: [mul_47, out_431, out_433], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf210 = extern_kernels.convolution(buf209, arg371_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg371_1
        del buf209
        buf211 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [out_434, out_435, out_436], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf211, buf210, arg372_1, arg373_1, arg374_1, arg375_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        del buf210
        # Topologically Sorted Source Nodes: [out_437], Original ATen: [aten.convolution]
        buf212 = extern_kernels.convolution(buf211, arg376_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg376_1
        buf213 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [out_438, out_439, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf213, arg377_1, arg378_1, arg379_1, arg380_1, 524288, grid=grid(524288), stream=stream0)
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        # Topologically Sorted Source Nodes: [out_438, out_439, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf214 = extern_kernels.convolution(buf213, arg381_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf214, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg381_1
        buf215 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_346, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf215, arg382_1, arg383_1, arg384_1, arg385_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        buf217 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_gap_240, x_gap_241, x_gap_242], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf215, buf217, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_240, x_gap_241, x_gap_242], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf218 = extern_kernels.convolution(buf217, arg386_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg386_1
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [x_gap_240, x_gap_241, x_gap_242, x_gap_243, x_gap_244, x_attn_96], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf219, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, 1024, grid=grid(1024), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        del arg391_1
        # Topologically Sorted Source Nodes: [x_gap_240, x_gap_241, x_gap_242, x_gap_243, x_gap_244, x_attn_96], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf220 = extern_kernels.convolution(buf219, arg392_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg392_1
        del buf219
        buf221 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_350], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf220, arg393_1, buf221, 4096, grid=grid(4096), stream=stream0)
        del arg393_1
        del buf220
        buf222 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [mul_48, out_440, out_442], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf215, buf221, buf222, 524288, grid=grid(524288), stream=stream0)
        del buf215
        # Topologically Sorted Source Nodes: [mul_48, out_440, out_442], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf223 = extern_kernels.convolution(buf222, arg394_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg394_1
        del buf222
        buf224 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [out_443, out_444, out_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf224, buf223, arg395_1, arg396_1, arg397_1, arg398_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg395_1
        del arg396_1
        del arg397_1
        del arg398_1
        del buf223
        # Topologically Sorted Source Nodes: [out_446], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, arg399_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg399_1
        buf226 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [out_447, out_448, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf226, arg400_1, arg401_1, arg402_1, arg403_1, 524288, grid=grid(524288), stream=stream0)
        del arg400_1
        del arg401_1
        del arg402_1
        del arg403_1
        # Topologically Sorted Source Nodes: [out_447, out_448, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf227 = extern_kernels.convolution(buf226, arg404_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf227, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg404_1
        buf228 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_353, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf228, arg405_1, arg406_1, arg407_1, arg408_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg405_1
        del arg406_1
        del arg407_1
        del arg408_1
        buf230 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_gap_245, x_gap_246, x_gap_247], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf228, buf230, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_245, x_gap_246, x_gap_247], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf231 = extern_kernels.convolution(buf230, arg409_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg409_1
        buf232 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [x_gap_245, x_gap_246, x_gap_247, x_gap_248, x_gap_249, x_attn_98], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf232, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, 1024, grid=grid(1024), stream=stream0)
        del arg410_1
        del arg411_1
        del arg412_1
        del arg413_1
        del arg414_1
        # Topologically Sorted Source Nodes: [x_gap_245, x_gap_246, x_gap_247, x_gap_248, x_gap_249, x_attn_98], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf233 = extern_kernels.convolution(buf232, arg415_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg415_1
        del buf232
        buf234 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_357], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf233, arg416_1, buf234, 4096, grid=grid(4096), stream=stream0)
        del arg416_1
        del buf233
        buf235 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [mul_49, out_449, out_451], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf228, buf234, buf235, 524288, grid=grid(524288), stream=stream0)
        del buf228
        # Topologically Sorted Source Nodes: [mul_49, out_449, out_451], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf236 = extern_kernels.convolution(buf235, arg417_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg417_1
        del buf235
        buf237 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [out_452, out_453, out_454], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf237, buf236, arg418_1, arg419_1, arg420_1, arg421_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg418_1
        del arg419_1
        del arg420_1
        del arg421_1
        del buf236
        # Topologically Sorted Source Nodes: [out_455], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, arg422_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg422_1
        buf239 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [out_456, out_457, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf239, arg423_1, arg424_1, arg425_1, arg426_1, 524288, grid=grid(524288), stream=stream0)
        del arg423_1
        del arg424_1
        del arg425_1
        del arg426_1
        # Topologically Sorted Source Nodes: [out_456, out_457, x_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf240 = extern_kernels.convolution(buf239, arg427_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf240, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg427_1
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_360, x_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf241, arg428_1, arg429_1, arg430_1, arg431_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg428_1
        del arg429_1
        del arg430_1
        del arg431_1
        buf243 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_gap_250, x_gap_251, x_gap_252], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf241, buf243, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_250, x_gap_251, x_gap_252], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf244 = extern_kernels.convolution(buf243, arg432_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg432_1
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [x_gap_250, x_gap_251, x_gap_252, x_gap_253, x_gap_254, x_attn_100], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf245, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, 1024, grid=grid(1024), stream=stream0)
        del arg433_1
        del arg434_1
        del arg435_1
        del arg436_1
        del arg437_1
        # Topologically Sorted Source Nodes: [x_gap_250, x_gap_251, x_gap_252, x_gap_253, x_gap_254, x_attn_100], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf246 = extern_kernels.convolution(buf245, arg438_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg438_1
        del buf245
        buf247 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [x_364], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf246, arg439_1, buf247, 4096, grid=grid(4096), stream=stream0)
        del arg439_1
        del buf246
        buf248 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [mul_50, out_458, out_460], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf241, buf247, buf248, 524288, grid=grid(524288), stream=stream0)
        del buf241
        # Topologically Sorted Source Nodes: [mul_50, out_458, out_460], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf249 = extern_kernels.convolution(buf248, arg440_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg440_1
        del buf248
        buf250 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [out_461, out_462, out_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf250, buf249, arg441_1, arg442_1, arg443_1, arg444_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg441_1
        del arg442_1
        del arg443_1
        del arg444_1
        del buf249
        # Topologically Sorted Source Nodes: [out_464], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, arg445_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg445_1
        buf252 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [out_465, out_466, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf252, arg446_1, arg447_1, arg448_1, arg449_1, 524288, grid=grid(524288), stream=stream0)
        del arg446_1
        del arg447_1
        del arg448_1
        del arg449_1
        # Topologically Sorted Source Nodes: [out_465, out_466, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf253 = extern_kernels.convolution(buf252, arg450_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf253, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg450_1
        buf254 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_367, x_368], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf254, arg451_1, arg452_1, arg453_1, arg454_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg451_1
        del arg452_1
        del arg453_1
        del arg454_1
        buf256 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_gap_255, x_gap_256, x_gap_257], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf254, buf256, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_255, x_gap_256, x_gap_257], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf257 = extern_kernels.convolution(buf256, arg455_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg455_1
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [x_gap_255, x_gap_256, x_gap_257, x_gap_258, x_gap_259, x_attn_102], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf258, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, 1024, grid=grid(1024), stream=stream0)
        del arg456_1
        del arg457_1
        del arg458_1
        del arg459_1
        del arg460_1
        # Topologically Sorted Source Nodes: [x_gap_255, x_gap_256, x_gap_257, x_gap_258, x_gap_259, x_attn_102], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf259 = extern_kernels.convolution(buf258, arg461_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg461_1
        del buf258
        buf260 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [x_371], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf259, arg462_1, buf260, 4096, grid=grid(4096), stream=stream0)
        del arg462_1
        del buf259
        buf261 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [mul_51, out_467, out_469], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf254, buf260, buf261, 524288, grid=grid(524288), stream=stream0)
        del buf254
        # Topologically Sorted Source Nodes: [mul_51, out_467, out_469], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf262 = extern_kernels.convolution(buf261, arg463_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg463_1
        del buf261
        buf263 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [out_470, out_471, out_472], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf263, buf262, arg464_1, arg465_1, arg466_1, arg467_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg464_1
        del arg465_1
        del arg466_1
        del arg467_1
        del buf262
        # Topologically Sorted Source Nodes: [out_473], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, arg468_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg468_1
        buf265 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [out_474, out_475, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf265, arg469_1, arg470_1, arg471_1, arg472_1, 524288, grid=grid(524288), stream=stream0)
        del arg469_1
        del arg470_1
        del arg471_1
        del arg472_1
        # Topologically Sorted Source Nodes: [out_474, out_475, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf266 = extern_kernels.convolution(buf265, arg473_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf266, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg473_1
        buf267 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [x_374, x_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf267, arg474_1, arg475_1, arg476_1, arg477_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg474_1
        del arg475_1
        del arg476_1
        del arg477_1
        buf269 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [x_gap_260, x_gap_261, x_gap_262], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf267, buf269, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_260, x_gap_261, x_gap_262], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf270 = extern_kernels.convolution(buf269, arg478_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg478_1
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [x_gap_260, x_gap_261, x_gap_262, x_gap_263, x_gap_264, x_attn_104], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf271, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, 1024, grid=grid(1024), stream=stream0)
        del arg479_1
        del arg480_1
        del arg481_1
        del arg482_1
        del arg483_1
        # Topologically Sorted Source Nodes: [x_gap_260, x_gap_261, x_gap_262, x_gap_263, x_gap_264, x_attn_104], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf272 = extern_kernels.convolution(buf271, arg484_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg484_1
        del buf271
        buf273 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [x_378], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf272, arg485_1, buf273, 4096, grid=grid(4096), stream=stream0)
        del arg485_1
        del buf272
        buf274 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [mul_52, out_476, out_478], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf267, buf273, buf274, 524288, grid=grid(524288), stream=stream0)
        del buf267
        # Topologically Sorted Source Nodes: [mul_52, out_476, out_478], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf275 = extern_kernels.convolution(buf274, arg486_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg486_1
        del buf274
        buf276 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [out_479, out_480, out_481], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf276, buf275, arg487_1, arg488_1, arg489_1, arg490_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg487_1
        del arg488_1
        del arg489_1
        del arg490_1
        del buf275
        # Topologically Sorted Source Nodes: [out_482], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, arg491_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg491_1
        buf278 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [out_483, out_484, x_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf278, arg492_1, arg493_1, arg494_1, arg495_1, 524288, grid=grid(524288), stream=stream0)
        del arg492_1
        del arg493_1
        del arg494_1
        del arg495_1
        # Topologically Sorted Source Nodes: [out_483, out_484, x_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf279 = extern_kernels.convolution(buf278, arg496_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf279, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg496_1
        buf280 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf280, arg497_1, arg498_1, arg499_1, arg500_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg497_1
        del arg498_1
        del arg499_1
        del arg500_1
        buf282 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_gap_265, x_gap_266, x_gap_267], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf280, buf282, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_265, x_gap_266, x_gap_267], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf283 = extern_kernels.convolution(buf282, arg501_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg501_1
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [x_gap_265, x_gap_266, x_gap_267, x_gap_268, x_gap_269, x_attn_106], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf284, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, 1024, grid=grid(1024), stream=stream0)
        del arg502_1
        del arg503_1
        del arg504_1
        del arg505_1
        del arg506_1
        # Topologically Sorted Source Nodes: [x_gap_265, x_gap_266, x_gap_267, x_gap_268, x_gap_269, x_attn_106], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf285 = extern_kernels.convolution(buf284, arg507_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg507_1
        del buf284
        buf286 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [x_385], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf285, arg508_1, buf286, 4096, grid=grid(4096), stream=stream0)
        del arg508_1
        del buf285
        buf287 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [mul_53, out_485, out_487], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf280, buf286, buf287, 524288, grid=grid(524288), stream=stream0)
        del buf280
        # Topologically Sorted Source Nodes: [mul_53, out_485, out_487], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf288 = extern_kernels.convolution(buf287, arg509_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg509_1
        del buf287
        buf289 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [out_488, out_489, out_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf289, buf288, arg510_1, arg511_1, arg512_1, arg513_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg510_1
        del arg511_1
        del arg512_1
        del arg513_1
        del buf288
        # Topologically Sorted Source Nodes: [out_491], Original ATen: [aten.convolution]
        buf290 = extern_kernels.convolution(buf289, arg514_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf290, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg514_1
        buf291 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [out_492, out_493, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf291, arg515_1, arg516_1, arg517_1, arg518_1, 524288, grid=grid(524288), stream=stream0)
        del arg515_1
        del arg516_1
        del arg517_1
        del arg518_1
        # Topologically Sorted Source Nodes: [out_492, out_493, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf292 = extern_kernels.convolution(buf291, arg519_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf292, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg519_1
        buf293 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [x_388, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf293, arg520_1, arg521_1, arg522_1, arg523_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg520_1
        del arg521_1
        del arg522_1
        del arg523_1
        buf295 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [x_gap_270, x_gap_271, x_gap_272], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf293, buf295, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_270, x_gap_271, x_gap_272], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf296 = extern_kernels.convolution(buf295, arg524_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg524_1
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [x_gap_270, x_gap_271, x_gap_272, x_gap_273, x_gap_274, x_attn_108], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf297, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, 1024, grid=grid(1024), stream=stream0)
        del arg525_1
        del arg526_1
        del arg527_1
        del arg528_1
        del arg529_1
        # Topologically Sorted Source Nodes: [x_gap_270, x_gap_271, x_gap_272, x_gap_273, x_gap_274, x_attn_108], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf298 = extern_kernels.convolution(buf297, arg530_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg530_1
        del buf297
        buf299 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [x_392], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf298, arg531_1, buf299, 4096, grid=grid(4096), stream=stream0)
        del arg531_1
        del buf298
        buf300 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [mul_54, out_494, out_496], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf293, buf299, buf300, 524288, grid=grid(524288), stream=stream0)
        del buf293
        # Topologically Sorted Source Nodes: [mul_54, out_494, out_496], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf301 = extern_kernels.convolution(buf300, arg532_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg532_1
        del buf300
        buf302 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [out_497, out_498, out_499], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf302, buf301, arg533_1, arg534_1, arg535_1, arg536_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg533_1
        del arg534_1
        del arg535_1
        del arg536_1
        del buf301
        # Topologically Sorted Source Nodes: [out_500], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf302, arg537_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg537_1
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [out_501, out_502, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf304, arg538_1, arg539_1, arg540_1, arg541_1, 524288, grid=grid(524288), stream=stream0)
        del arg538_1
        del arg539_1
        del arg540_1
        del arg541_1
        # Topologically Sorted Source Nodes: [out_501, out_502, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf305 = extern_kernels.convolution(buf304, arg542_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf305, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg542_1
        buf306 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [x_395, x_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf306, arg543_1, arg544_1, arg545_1, arg546_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg543_1
        del arg544_1
        del arg545_1
        del arg546_1
        buf308 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [x_gap_275, x_gap_276, x_gap_277], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf306, buf308, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_275, x_gap_276, x_gap_277], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf309 = extern_kernels.convolution(buf308, arg547_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg547_1
        buf310 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [x_gap_275, x_gap_276, x_gap_277, x_gap_278, x_gap_279, x_attn_110], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf310, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, 1024, grid=grid(1024), stream=stream0)
        del arg548_1
        del arg549_1
        del arg550_1
        del arg551_1
        del arg552_1
        # Topologically Sorted Source Nodes: [x_gap_275, x_gap_276, x_gap_277, x_gap_278, x_gap_279, x_attn_110], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf311 = extern_kernels.convolution(buf310, arg553_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg553_1
        del buf310
        buf312 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [x_399], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf311, arg554_1, buf312, 4096, grid=grid(4096), stream=stream0)
        del arg554_1
        del buf311
        buf313 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [mul_55, out_503, out_505], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf306, buf312, buf313, 524288, grid=grid(524288), stream=stream0)
        del buf306
        # Topologically Sorted Source Nodes: [mul_55, out_503, out_505], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf314 = extern_kernels.convolution(buf313, arg555_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg555_1
        del buf313
        buf315 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [out_506, out_507, out_508], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf315, buf314, arg556_1, arg557_1, arg558_1, arg559_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg556_1
        del arg557_1
        del arg558_1
        del arg559_1
        del buf314
        # Topologically Sorted Source Nodes: [out_509], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, arg560_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg560_1
        buf317 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [out_510, out_511, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf317, arg561_1, arg562_1, arg563_1, arg564_1, 524288, grid=grid(524288), stream=stream0)
        del arg561_1
        del arg562_1
        del arg563_1
        del arg564_1
        # Topologically Sorted Source Nodes: [out_510, out_511, x_401], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf318 = extern_kernels.convolution(buf317, arg565_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf318, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg565_1
        buf319 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf319, arg566_1, arg567_1, arg568_1, arg569_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg566_1
        del arg567_1
        del arg568_1
        del arg569_1
        buf321 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [x_gap_280, x_gap_281, x_gap_282], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf319, buf321, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_280, x_gap_281, x_gap_282], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf322 = extern_kernels.convolution(buf321, arg570_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg570_1
        buf323 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [x_gap_280, x_gap_281, x_gap_282, x_gap_283, x_gap_284, x_attn_112], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf323, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, 1024, grid=grid(1024), stream=stream0)
        del arg571_1
        del arg572_1
        del arg573_1
        del arg574_1
        del arg575_1
        # Topologically Sorted Source Nodes: [x_gap_280, x_gap_281, x_gap_282, x_gap_283, x_gap_284, x_attn_112], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf324 = extern_kernels.convolution(buf323, arg576_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg576_1
        del buf323
        buf325 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [x_406], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf324, arg577_1, buf325, 4096, grid=grid(4096), stream=stream0)
        del arg577_1
        del buf324
        buf326 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [mul_56, out_512, out_514], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf319, buf325, buf326, 524288, grid=grid(524288), stream=stream0)
        del buf319
        # Topologically Sorted Source Nodes: [mul_56, out_512, out_514], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf327 = extern_kernels.convolution(buf326, arg578_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg578_1
        del buf326
        buf328 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [out_515, out_516, out_517], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf328, buf327, arg579_1, arg580_1, arg581_1, arg582_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg579_1
        del arg580_1
        del arg581_1
        del arg582_1
        del buf327
        # Topologically Sorted Source Nodes: [out_518], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, arg583_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg583_1
        buf330 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [out_519, out_520, x_408], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf330, arg584_1, arg585_1, arg586_1, arg587_1, 524288, grid=grid(524288), stream=stream0)
        del arg584_1
        del arg585_1
        del arg586_1
        del arg587_1
        # Topologically Sorted Source Nodes: [out_519, out_520, x_408], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf331 = extern_kernels.convolution(buf330, arg588_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf331, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg588_1
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [x_409, x_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf332, arg589_1, arg590_1, arg591_1, arg592_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg589_1
        del arg590_1
        del arg591_1
        del arg592_1
        buf334 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [x_gap_285, x_gap_286, x_gap_287], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf332, buf334, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_285, x_gap_286, x_gap_287], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf335 = extern_kernels.convolution(buf334, arg593_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg593_1
        buf336 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [x_gap_285, x_gap_286, x_gap_287, x_gap_288, x_gap_289, x_attn_114], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf336, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, 1024, grid=grid(1024), stream=stream0)
        del arg594_1
        del arg595_1
        del arg596_1
        del arg597_1
        del arg598_1
        # Topologically Sorted Source Nodes: [x_gap_285, x_gap_286, x_gap_287, x_gap_288, x_gap_289, x_attn_114], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf337 = extern_kernels.convolution(buf336, arg599_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg599_1
        del buf336
        buf338 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_413], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf337, arg600_1, buf338, 4096, grid=grid(4096), stream=stream0)
        del arg600_1
        del buf337
        buf339 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [mul_57, out_521, out_523], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf332, buf338, buf339, 524288, grid=grid(524288), stream=stream0)
        del buf332
        # Topologically Sorted Source Nodes: [mul_57, out_521, out_523], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf340 = extern_kernels.convolution(buf339, arg601_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg601_1
        del buf339
        buf341 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [out_524, out_525, out_526], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf341, buf340, arg602_1, arg603_1, arg604_1, arg605_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg602_1
        del arg603_1
        del arg604_1
        del arg605_1
        del buf340
        # Topologically Sorted Source Nodes: [out_527], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, arg606_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg606_1
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [out_528, out_529, x_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf343, arg607_1, arg608_1, arg609_1, arg610_1, 524288, grid=grid(524288), stream=stream0)
        del arg607_1
        del arg608_1
        del arg609_1
        del arg610_1
        # Topologically Sorted Source Nodes: [out_528, out_529, x_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf344 = extern_kernels.convolution(buf343, arg611_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf344, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg611_1
        buf345 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [x_416, x_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf345, arg612_1, arg613_1, arg614_1, arg615_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg612_1
        del arg613_1
        del arg614_1
        del arg615_1
        buf347 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [x_gap_290, x_gap_291, x_gap_292], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf345, buf347, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_290, x_gap_291, x_gap_292], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf348 = extern_kernels.convolution(buf347, arg616_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg616_1
        buf349 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [x_gap_290, x_gap_291, x_gap_292, x_gap_293, x_gap_294, x_attn_116], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf349, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, 1024, grid=grid(1024), stream=stream0)
        del arg617_1
        del arg618_1
        del arg619_1
        del arg620_1
        del arg621_1
        # Topologically Sorted Source Nodes: [x_gap_290, x_gap_291, x_gap_292, x_gap_293, x_gap_294, x_attn_116], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf350 = extern_kernels.convolution(buf349, arg622_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg622_1
        del buf349
        buf351 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [x_420], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf350, arg623_1, buf351, 4096, grid=grid(4096), stream=stream0)
        del arg623_1
        del buf350
        buf352 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [mul_58, out_530, out_532], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf345, buf351, buf352, 524288, grid=grid(524288), stream=stream0)
        del buf345
        # Topologically Sorted Source Nodes: [mul_58, out_530, out_532], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf353 = extern_kernels.convolution(buf352, arg624_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg624_1
        del buf352
        buf354 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [out_533, out_534, out_535], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf354, buf353, arg625_1, arg626_1, arg627_1, arg628_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg625_1
        del arg626_1
        del arg627_1
        del arg628_1
        del buf353
        # Topologically Sorted Source Nodes: [out_536], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, arg629_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg629_1
        buf356 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [out_537, out_538, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf356, arg630_1, arg631_1, arg632_1, arg633_1, 524288, grid=grid(524288), stream=stream0)
        del arg630_1
        del arg631_1
        del arg632_1
        del arg633_1
        # Topologically Sorted Source Nodes: [out_537, out_538, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf357 = extern_kernels.convolution(buf356, arg634_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf357, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg634_1
        buf358 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [x_423, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf358, arg635_1, arg636_1, arg637_1, arg638_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg635_1
        del arg636_1
        del arg637_1
        del arg638_1
        buf360 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [x_gap_295, x_gap_296, x_gap_297], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf358, buf360, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_295, x_gap_296, x_gap_297], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf361 = extern_kernels.convolution(buf360, arg639_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg639_1
        buf362 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [x_gap_295, x_gap_296, x_gap_297, x_gap_298, x_gap_299, x_attn_118], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf362, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, 1024, grid=grid(1024), stream=stream0)
        del arg640_1
        del arg641_1
        del arg642_1
        del arg643_1
        del arg644_1
        # Topologically Sorted Source Nodes: [x_gap_295, x_gap_296, x_gap_297, x_gap_298, x_gap_299, x_attn_118], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf363 = extern_kernels.convolution(buf362, arg645_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg645_1
        del buf362
        buf364 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [x_427], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf363, arg646_1, buf364, 4096, grid=grid(4096), stream=stream0)
        del arg646_1
        del buf363
        buf365 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [mul_59, out_539, out_541], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf358, buf364, buf365, 524288, grid=grid(524288), stream=stream0)
        del buf358
        # Topologically Sorted Source Nodes: [mul_59, out_539, out_541], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf366 = extern_kernels.convolution(buf365, arg647_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg647_1
        del buf365
        buf367 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [out_542, out_543, out_544], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf367, buf366, arg648_1, arg649_1, arg650_1, arg651_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg648_1
        del arg649_1
        del arg650_1
        del arg651_1
        del buf366
        # Topologically Sorted Source Nodes: [out_545], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, arg652_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg652_1
        buf369 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [out_546, out_547, x_429], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf369, arg653_1, arg654_1, arg655_1, arg656_1, 524288, grid=grid(524288), stream=stream0)
        del arg653_1
        del arg654_1
        del arg655_1
        del arg656_1
        # Topologically Sorted Source Nodes: [out_546, out_547, x_429], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf370 = extern_kernels.convolution(buf369, arg657_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf370, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg657_1
        buf371 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [x_430, x_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf371, arg658_1, arg659_1, arg660_1, arg661_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg658_1
        del arg659_1
        del arg660_1
        del arg661_1
        buf373 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [x_gap_300, x_gap_301, x_gap_302], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf371, buf373, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_300, x_gap_301, x_gap_302], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf374 = extern_kernels.convolution(buf373, arg662_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg662_1
        buf375 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [x_gap_300, x_gap_301, x_gap_302, x_gap_303, x_gap_304, x_attn_120], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf375, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, 1024, grid=grid(1024), stream=stream0)
        del arg663_1
        del arg664_1
        del arg665_1
        del arg666_1
        del arg667_1
        # Topologically Sorted Source Nodes: [x_gap_300, x_gap_301, x_gap_302, x_gap_303, x_gap_304, x_attn_120], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf376 = extern_kernels.convolution(buf375, arg668_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg668_1
        del buf375
        buf377 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [x_434], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf376, arg669_1, buf377, 4096, grid=grid(4096), stream=stream0)
        del arg669_1
        del buf376
        buf378 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [mul_60, out_548, out_550], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf371, buf377, buf378, 524288, grid=grid(524288), stream=stream0)
        del buf371
        # Topologically Sorted Source Nodes: [mul_60, out_548, out_550], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf379 = extern_kernels.convolution(buf378, arg670_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg670_1
        del buf378
        buf380 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [out_551, out_552, out_553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf380, buf379, arg671_1, arg672_1, arg673_1, arg674_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg671_1
        del arg672_1
        del arg673_1
        del arg674_1
        del buf379
        # Topologically Sorted Source Nodes: [out_554], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, arg675_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg675_1
        buf382 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [out_555, out_556, x_436], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf382, arg676_1, arg677_1, arg678_1, arg679_1, 524288, grid=grid(524288), stream=stream0)
        del arg676_1
        del arg677_1
        del arg678_1
        del arg679_1
        # Topologically Sorted Source Nodes: [out_555, out_556, x_436], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf383 = extern_kernels.convolution(buf382, arg680_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf383, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg680_1
        buf384 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [x_437, x_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf384, arg681_1, arg682_1, arg683_1, arg684_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg681_1
        del arg682_1
        del arg683_1
        del arg684_1
        buf386 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [x_gap_305, x_gap_306, x_gap_307], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf384, buf386, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_305, x_gap_306, x_gap_307], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf387 = extern_kernels.convolution(buf386, arg685_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg685_1
        buf388 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_gap_305, x_gap_306, x_gap_307, x_gap_308, x_gap_309, x_attn_122], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf388, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, 1024, grid=grid(1024), stream=stream0)
        del arg686_1
        del arg687_1
        del arg688_1
        del arg689_1
        del arg690_1
        # Topologically Sorted Source Nodes: [x_gap_305, x_gap_306, x_gap_307, x_gap_308, x_gap_309, x_attn_122], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf389 = extern_kernels.convolution(buf388, arg691_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg691_1
        del buf388
        buf390 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [x_441], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf389, arg692_1, buf390, 4096, grid=grid(4096), stream=stream0)
        del arg692_1
        del buf389
        buf391 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [mul_61, out_557, out_559], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf384, buf390, buf391, 524288, grid=grid(524288), stream=stream0)
        del buf384
        # Topologically Sorted Source Nodes: [mul_61, out_557, out_559], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf392 = extern_kernels.convolution(buf391, arg693_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg693_1
        del buf391
        buf393 = buf380; del buf380  # reuse
        # Topologically Sorted Source Nodes: [out_560, out_561, out_562], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf393, buf392, arg694_1, arg695_1, arg696_1, arg697_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg694_1
        del arg695_1
        del arg696_1
        del arg697_1
        del buf392
        # Topologically Sorted Source Nodes: [out_563], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, arg698_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 256, 16, 16), (65536, 256, 16, 1))
        del arg698_1
        buf395 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [out_564, out_565, x_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(buf395, arg699_1, arg700_1, arg701_1, arg702_1, 524288, grid=grid(524288), stream=stream0)
        del arg699_1
        del arg700_1
        del arg701_1
        del arg702_1
        # Topologically Sorted Source Nodes: [out_564, out_565, x_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf396 = extern_kernels.convolution(buf395, arg703_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf396, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg703_1
        buf397 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [x_444, x_445], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf397, arg704_1, arg705_1, arg706_1, arg707_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg704_1
        del arg705_1
        del arg706_1
        del arg707_1
        buf399 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [x_gap_310, x_gap_311, x_gap_312], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_35.run(buf397, buf399, 2048, 256, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_310, x_gap_311, x_gap_312], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf400 = extern_kernels.convolution(buf399, arg708_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg708_1
        del buf399
        buf401 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [x_gap_310, x_gap_311, x_gap_312, x_gap_313, x_gap_314, x_attn_124], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_27.run(buf401, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, 1024, grid=grid(1024), stream=stream0)
        del arg709_1
        del arg710_1
        del arg711_1
        del arg712_1
        del arg713_1
        # Topologically Sorted Source Nodes: [x_gap_310, x_gap_311, x_gap_312, x_gap_313, x_gap_314, x_attn_124], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf402 = extern_kernels.convolution(buf401, arg714_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg714_1
        del buf401
        buf403 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [x_448], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf402, arg715_1, buf403, 4096, grid=grid(4096), stream=stream0)
        del arg715_1
        del buf402
        buf404 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [mul_62, out_566, out_568], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_36.run(buf397, buf403, buf404, 524288, grid=grid(524288), stream=stream0)
        del buf397
        # Topologically Sorted Source Nodes: [mul_62, out_566, out_568], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf405 = extern_kernels.convolution(buf404, arg716_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg716_1
        buf406 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [out_569, out_570, out_571], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_37.run(buf406, buf405, arg717_1, arg718_1, arg719_1, arg720_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg717_1
        del arg718_1
        del arg719_1
        del arg720_1
        del buf405
        # Topologically Sorted Source Nodes: [out_572], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, arg721_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 512, 16, 16), (131072, 256, 16, 1))
        del arg721_1
        buf408 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [out_573, out_574, x_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf408, arg722_1, arg723_1, arg724_1, arg725_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg722_1
        del arg723_1
        del arg724_1
        del arg725_1
        # Topologically Sorted Source Nodes: [out_573, out_574, x_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf409 = extern_kernels.convolution(buf408, arg726_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf409, (8, 1024, 16, 16), (262144, 256, 16, 1))
        del arg726_1
        buf410 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [x_451, x_452], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_38.run(buf410, arg727_1, arg728_1, arg729_1, arg730_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg727_1
        del arg728_1
        del arg729_1
        del arg730_1
        buf412 = reinterpret_tensor(buf403, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf403  # reuse
        # Topologically Sorted Source Nodes: [x_gap_315, x_gap_316, x_gap_317], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_39.run(buf410, buf412, 4096, 256, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_315, x_gap_316, x_gap_317], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf413 = extern_kernels.convolution(buf412, arg731_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf413, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg731_1
        buf414 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [x_gap_315, x_gap_316, x_gap_317, x_gap_318, x_gap_319, x_attn_126], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40.run(buf414, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, 2048, grid=grid(2048), stream=stream0)
        del arg732_1
        del arg733_1
        del arg734_1
        del arg735_1
        del arg736_1
        # Topologically Sorted Source Nodes: [x_gap_315, x_gap_316, x_gap_317, x_gap_318, x_gap_319, x_attn_126], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf415 = extern_kernels.convolution(buf414, arg737_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg737_1
        del buf414
        buf416 = empty_strided_cuda((8, 2, 1, 512), (1024, 512, 8192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_455], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_41.run(buf415, arg738_1, buf416, 8192, grid=grid(8192), stream=stream0)
        del arg738_1
        del buf415
        buf417 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [mul_63, out_575], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_42.run(buf410, buf416, buf417, 1048576, grid=grid(1048576), stream=stream0)
        del buf410
        buf418 = empty_strided_cuda((8, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_63, out_575, out_577], Original ATen: [aten.mul, aten.sum, aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_mul_sum_43.run(buf417, buf418, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [out_578], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, arg739_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg739_1
        del buf418
        buf420 = reinterpret_tensor(buf404, (8, 1024, 8, 8), (65536, 64, 8, 1), 0); del buf404  # reuse
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_44.run(buf406, buf420, 524288, grid=grid(524288), stream=stream0)
        del buf406
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf421 = extern_kernels.convolution(buf420, arg744_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg744_1
        del buf420
        buf422 = buf419; del buf419  # reuse
        buf423 = reinterpret_tensor(buf417, (8, 2048, 8, 8), (131072, 64, 8, 1), 0); del buf417  # reuse
        # Topologically Sorted Source Nodes: [out_579, input_36, out_580, out_581], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_45.run(buf422, arg740_1, arg741_1, arg742_1, arg743_1, buf421, arg745_1, arg746_1, arg747_1, arg748_1, buf423, 1048576, grid=grid(1048576), stream=stream0)
        del arg740_1
        del arg741_1
        del arg742_1
        del arg743_1
        del arg745_1
        del arg746_1
        del arg747_1
        del arg748_1
        del buf421
        del buf422
        # Topologically Sorted Source Nodes: [out_582], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, arg749_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg749_1
        buf425 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [out_583, out_584, x_457], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46.run(buf425, arg750_1, arg751_1, arg752_1, arg753_1, 262144, grid=grid(262144), stream=stream0)
        del arg750_1
        del arg751_1
        del arg752_1
        del arg753_1
        # Topologically Sorted Source Nodes: [out_583, out_584, x_457], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf426 = extern_kernels.convolution(buf425, arg754_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf426, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del arg754_1
        buf427 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [x_458, x_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf427, arg755_1, arg756_1, arg757_1, arg758_1, 524288, grid=grid(524288), stream=stream0)
        del arg755_1
        del arg756_1
        del arg757_1
        del arg758_1
        buf429 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [x_gap_320, x_gap_321, x_gap_322], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_48.run(buf427, buf429, 4096, 64, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_320, x_gap_321, x_gap_322], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf430 = extern_kernels.convolution(buf429, arg759_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg759_1
        buf431 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [x_gap_320, x_gap_321, x_gap_322, x_gap_323, x_gap_324, x_attn_128], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40.run(buf431, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, 2048, grid=grid(2048), stream=stream0)
        del arg760_1
        del arg761_1
        del arg762_1
        del arg763_1
        del arg764_1
        # Topologically Sorted Source Nodes: [x_gap_320, x_gap_321, x_gap_322, x_gap_323, x_gap_324, x_attn_128], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf432 = extern_kernels.convolution(buf431, arg765_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg765_1
        del buf431
        buf433 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [x_462], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_41.run(buf432, arg766_1, buf433, 8192, grid=grid(8192), stream=stream0)
        del arg766_1
        del buf432
        buf434 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [mul_64, out_585, out_587], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_49.run(buf427, buf433, buf434, 262144, grid=grid(262144), stream=stream0)
        del buf427
        # Topologically Sorted Source Nodes: [mul_64, out_585, out_587], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf435 = extern_kernels.convolution(buf434, arg767_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg767_1
        del buf434
        buf436 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [out_588, out_589, out_590], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_50.run(buf436, buf435, arg768_1, arg769_1, arg770_1, arg771_1, 1048576, grid=grid(1048576), stream=stream0)
        del arg768_1
        del arg769_1
        del arg770_1
        del arg771_1
        del buf435
        # Topologically Sorted Source Nodes: [out_591], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, arg772_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 512, 8, 8), (32768, 64, 8, 1))
        del arg772_1
        buf438 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [out_592, out_593, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46.run(buf438, arg773_1, arg774_1, arg775_1, arg776_1, 262144, grid=grid(262144), stream=stream0)
        del arg773_1
        del arg774_1
        del arg775_1
        del arg776_1
        # Topologically Sorted Source Nodes: [out_592, out_593, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf439 = extern_kernels.convolution(buf438, arg777_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf439, (8, 1024, 8, 8), (65536, 64, 8, 1))
        del arg777_1
        buf440 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [x_465, x_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf440, arg778_1, arg779_1, arg780_1, arg781_1, 524288, grid=grid(524288), stream=stream0)
        del arg778_1
        del arg779_1
        del arg780_1
        del arg781_1
        buf442 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [x_gap_325, x_gap_326, x_gap_327], Original ATen: [aten.sum, aten.mean, aten.convolution]
        triton_per_fused_convolution_mean_sum_48.run(buf440, buf442, 4096, 64, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [x_gap_325, x_gap_326, x_gap_327], Original ATen: [aten.sum, aten.mean, aten.convolution]
        buf443 = extern_kernels.convolution(buf442, arg782_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg782_1
        del buf442
        buf444 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [x_gap_325, x_gap_326, x_gap_327, x_gap_328, x_gap_329, x_attn_130], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sum_40.run(buf444, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, 2048, grid=grid(2048), stream=stream0)
        del arg783_1
        del arg784_1
        del arg785_1
        del arg786_1
        del arg787_1
        # Topologically Sorted Source Nodes: [x_gap_325, x_gap_326, x_gap_327, x_gap_328, x_gap_329, x_attn_130], Original ATen: [aten.sum, aten.mean, aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        buf445 = extern_kernels.convolution(buf444, arg788_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (8, 1024, 1, 1), (1024, 1, 1, 1))
        del arg788_1
        del buf444
        buf446 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [x_469], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_41.run(buf445, arg789_1, buf446, 8192, grid=grid(8192), stream=stream0)
        del arg789_1
        del buf445
        buf447 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [mul_65, out_594, out_596], Original ATen: [aten.mul, aten.sum, aten.convolution]
        triton_poi_fused_convolution_mul_sum_49.run(buf440, buf446, buf447, 262144, grid=grid(262144), stream=stream0)
        del buf440
        del buf446
        # Topologically Sorted Source Nodes: [mul_65, out_594, out_596], Original ATen: [aten.mul, aten.sum, aten.convolution]
        buf448 = extern_kernels.convolution(buf447, arg790_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 2048, 8, 8), (131072, 64, 8, 1))
        del arg790_1
        del buf447
        buf450 = empty_strided_cuda((8, 2048, 1, 1), (2048, 1, 16384, 16384), torch.float32)
        # Topologically Sorted Source Nodes: [out_597, out_598, out_599, x_471], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_51.run(buf448, arg791_1, arg792_1, arg793_1, arg794_1, buf436, buf450, 16384, 64, grid=grid(16384), stream=stream0)
        del arg791_1
        del arg792_1
        del arg793_1
        del arg794_1
        del buf436
        del buf448
        buf451 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_473], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg796_1, reinterpret_tensor(buf450, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg795_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf451)
        del arg795_1
        del arg796_1
        del buf450
    return (buf451, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnest101e', benchmark_compiled_module)
