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


# kernel path: /tmp/torchinductor_sahanp/2q/c2q7p7npaq253tnmyivd4uti6mpvtrweyqscra455u2rxcnywgoh.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_7 => convolution_85
# Graph fragment:
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qf/cqflub2y6iz2z7ewuf3uappzueej37bkssst5ahu4p6ldjwqdoto.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_7 => convolution_85
# Graph fragment:
#   %convolution_85 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (147*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cz/cczi5gg3b7juhbzml2qsxaep3irwgnmmqqno5qffufpvejua2ji4.py
# Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_8 => add_211, mul_256, mul_257, sub_85
#   x_9 => relu_81
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %relu_81 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_211,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_sahanp/dv/cdvyuwrkhcqphr2tx6d5kx5er7w4nien6b7zbzwvxezj6tacdkj3.py
# Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_10 => _low_memory_max_pool2d_with_offsets_1
#   x_8 => add_211, mul_256, mul_257, sub_85
#   x_9 => relu_81
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_681), kwargs = {})
#   %mul_256 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_257 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_256, %unsqueeze_685), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_257, %unsqueeze_687), kwargs = {})
#   %relu_81 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_211,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_81, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = (xindex // 3584) % 56
    x1 = (xindex // 64) % 56
    x0 = xindex % 64
    x5 = (xindex // 3584)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7232) + x0 + (128*x1) + (14336*x5)), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7168) + x0 + (128*x1) + (14336*x5)), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + (2*x1)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-7104) + x0 + (128*x1) + (14336*x5)), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (14336*x5)), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + (128*x1) + (14336*x5)), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (14336*x5)), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + (2*x2)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (14336*x5)), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (14336*x5)), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7232 + x0 + (128*x1) + (14336*x5)), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tl.store(out_ptr0 + (x6), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ml/cmlay32ewmjmnodcz5bzm5yzwk7243aa64arbfskduzzn4akwmmq.py
# Topologically Sorted Source Nodes: [out_129, out_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_129 => add_213, mul_259, mul_260, sub_86
#   out_130 => relu_82
# Graph fragment:
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_689), kwargs = {})
#   %mul_259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_260 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_259, %unsqueeze_693), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_260, %unsqueeze_695), kwargs = {})
#   %relu_82 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_213,), kwargs = {})
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
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_sahanp/dz/cdzywzznczfqxltniglk3modnjrvqvwpj3p2cdph6wfautriyhtu.py
# Topologically Sorted Source Nodes: [sp_193], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_193 => convolution_87
# Graph fragment:
#   %convolution_87 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_328, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 4
    y1 = (yindex // 4)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (4*x2) + (36*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vd/cvdgblqpfjwx73sjxkyuehlsi6f5artfjmc5cciiytefljlxrwuj.py
# Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_343, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_6 = async_compile.triton('triton_poi_fused_avg_pool2d_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 56) % 56
    y0 = yindex % 56
    x3 = xindex
    y5 = yindex
    y2 = (yindex // 3136)
    y6 = yindex % 3136
    tmp0 = (-1) + y1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + y0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7200) + x3 + (128*y5)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7072) + x3 + (128*y5)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-6944) + x3 + (128*y5)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-32) + x3 + (128*y5)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (96 + x3 + (128*y5)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (224 + x3 + (128*y5)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (7136 + x3 + (128*y5)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7264 + x3 + (128*y5)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7392 + x3 + (128*y5)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*y0) + ((-1)*y1) + (y0*y1) + (((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57)))*((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))) + ((-1)*y0*((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))) + ((-1)*y1*((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57)))) + ((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57))) + ((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y6 + (3136*x3) + (401408*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4g/c4gsewyrqnp37k22uqult2zydzkk5difni4rrumbtsjk7fqtcsh3.py
# Topologically Sorted Source Nodes: [sp_194, sp_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_194 => add_215, mul_262, mul_263, sub_87
#   sp_195 => relu_83
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_697), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_262, %unsqueeze_701), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_263, %unsqueeze_703), kwargs = {})
#   %relu_83 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_215,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + (3136*y0) + (401408*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ht/chtq2rx57abzagba4pgzsvqwn7qmyb3aqiicoue5cii5r4nb2nxg.py
# Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_132 => convolution_90
# Graph fragment:
#   %convolution_90 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_16, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_8 = async_compile.triton('triton_poi_fused_convolution_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (401408*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xx/cxx4x6yndyw5nnibgriubhzn46v6fabjfvtib4f32ouv5srfthxx.py
# Topologically Sorted Source Nodes: [out_133, input_10, out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_223, mul_274, mul_275, sub_91
#   out_133 => add_221, mul_271, mul_272, sub_90
#   out_134 => add_224
#   out_135 => relu_86
# Graph fragment:
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_721), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_725), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_727), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_729), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %unsqueeze_733), kwargs = {})
#   %add_223 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %unsqueeze_735), kwargs = {})
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_221, %add_223), kwargs = {})
#   %relu_86 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_224,), kwargs = {})
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
    xnumel = 6422528
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
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7q/c7qxi4h75y6hcunvw2lkzimq5wwapn7ipmxjtqvjh255sriqsg7j.py
# Topologically Sorted Source Nodes: [sp_208], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_208 => add_229
# Graph fragment:
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_88, %getitem_353), kwargs = {})
triton_poi_fused_add_10 = async_compile.triton('triton_poi_fused_add_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_10(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (32 + x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mg/cmgcicla6nd6emvqakwno7cqtcy4bmr4jt2fhly4pvibxfhl2td2.py
# Topologically Sorted Source Nodes: [sp_212], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_212 => add_232
# Graph fragment:
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_89, %getitem_358), kwargs = {})
triton_poi_fused_add_11 = async_compile.triton('triton_poi_fused_add_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_11(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (64 + x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (32*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yz/cyzf2gtiexuoh7pooxdiqevioeeyhb3ph3hlrk7ocshxcgsioxm6.py
# Topologically Sorted Source Nodes: [out_139], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_139 => cat_17
# Graph fragment:
#   %cat_17 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_88, %relu_89, %relu_90, %getitem_363], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (96 + y0 + (128*x2) + (401408*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y0) + (401408*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2c/c2czh36ngygmmtpxk25kxdedpmd63eier2lruzeaoobw5th4jt3v.py
# Topologically Sorted Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_141 => add_236, mul_289, mul_290, sub_96
#   out_142 => add_237
#   out_143 => relu_91
# Graph fragment:
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_769), kwargs = {})
#   %mul_289 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %unsqueeze_773), kwargs = {})
#   %add_236 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %unsqueeze_775), kwargs = {})
#   %add_237 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_236, %relu_86), kwargs = {})
#   %relu_91 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_237,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/24/c24lgjrfjb5q2mtr2ekpxfwfyc7a7l7nq3zlfp22gugbs5bxm2yf.py
# Topologically Sorted Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_153 => add_252, mul_307, mul_308, sub_102
#   out_154 => relu_97
# Graph fragment:
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %mul_307 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_308 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_307, %unsqueeze_821), kwargs = {})
#   %add_252 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_308, %unsqueeze_823), kwargs = {})
#   %relu_97 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_252,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_sahanp/wn/cwn2qivn5vcd546b6xccvwl3jtj5hpzx2b72zvrtbfvdbryb7n5y.py
# Topologically Sorted Source Nodes: [sp_229], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_229 => convolution_103
# Graph fragment:
#   %convolution_103 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_388, %arg91_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 8
    y1 = (yindex // 8)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (8*x2) + (72*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ix/cix665jsxvrtif6hv7q2auils2mxxvxw3wbb5z5hdlcvoqo6snp4.py
# Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_403, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_16 = async_compile.triton('triton_poi_fused_avg_pool2d_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 28) % 28
    y0 = yindex % 28
    x3 = xindex
    y4 = (yindex // 28)
    y2 = (yindex // 784)
    y5 = yindex % 784
    tmp0 = (-1) + (2*y1)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*y0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-14400) + x3 + (512*y0) + (28672*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-14144) + x3 + (512*y0) + (28672*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-13888) + x3 + (512*y0) + (28672*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x3 + (512*y0) + (28672*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (192 + x3 + (512*y0) + (28672*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (448 + x3 + (512*y0) + (28672*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (14272 + x3 + (512*y0) + (28672*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (14528 + x3 + (512*y0) + (28672*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (14784 + x3 + (512*y0) + (28672*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57)))*((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))) + ((-2)*y0*((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))) + ((-2)*y1*((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57)))) + (4*y0*y1) + ((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57))) + ((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (784*x3) + (200704*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/a3/ca3ajbk4e7wenvxm3dwzjz75ro3vtamaghmqdndnwb6ukjbd4ttl.py
# Topologically Sorted Source Nodes: [sp_230, sp_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_230 => add_254, mul_310, mul_311, sub_103
#   sp_231 => relu_98
# Graph fragment:
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %mul_310 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_310, %unsqueeze_829), kwargs = {})
#   %add_254 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_311, %unsqueeze_831), kwargs = {})
#   %relu_98 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_254,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (y0 + (64*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + (784*y0) + (200704*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pb/cpb7ftmk7tzcc2oxkun5yahg7isk3mzryikgz3zkcr6tsx6ttxbl.py
# Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_156 => convolution_106
# Graph fragment:
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_19, %arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_18 = async_compile.triton('triton_poi_fused_convolution_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (200704*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ay/cay3mgtetgjqws32liuom3vqgzjtbk7qljzmu5q3nryziq2637wh.py
# Topologically Sorted Source Nodes: [out_157, input_12, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_12 => add_262, mul_322, mul_323, sub_107
#   out_157 => add_260, mul_319, mul_320, sub_106
#   out_158 => add_263
#   out_159 => relu_101
# Graph fragment:
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_849), kwargs = {})
#   %mul_319 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_319, %unsqueeze_853), kwargs = {})
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_320, %unsqueeze_855), kwargs = {})
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_857), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_861), kwargs = {})
#   %add_262 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_863), kwargs = {})
#   %add_263 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_260, %add_262), kwargs = {})
#   %relu_101 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_263,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2d/c2ddkpeqqtblsqgenu7wqvm3maie5r2axcemaymsuezo6u5exlxs.py
# Topologically Sorted Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_161 => add_265, mul_325, mul_326, sub_108
#   out_162 => relu_102
# Graph fragment:
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_865), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_867), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %unsqueeze_869), kwargs = {})
#   %add_265 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %unsqueeze_871), kwargs = {})
#   %relu_102 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_265,), kwargs = {})
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
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_sahanp/rs/crsxeavmgaimaymhejfvvyol545szla3t6bs4bwkvngkgx3w67a7.py
# Topologically Sorted Source Nodes: [sp_244], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_244 => add_268
# Graph fragment:
#   %add_268 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_103, %getitem_413), kwargs = {})
triton_poi_fused_add_21 = async_compile.triton('triton_poi_fused_add_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_21(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (64 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mc/cmcqmuy4owsidmb47epvy2wk5acrxxiuj3ttlpy3p52tt3rv744s.py
# Topologically Sorted Source Nodes: [sp_248], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_248 => add_271
# Graph fragment:
#   %add_271 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_104, %getitem_418), kwargs = {})
triton_poi_fused_add_22 = async_compile.triton('triton_poi_fused_add_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_22(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 64
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + x2 + (256*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (64*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/7v/c7vtyxm3br4a7ywulbmw47d3poixniuazglyxnutmtiw3qryhkhf.py
# Topologically Sorted Source Nodes: [out_163], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_163 => cat_20
# Graph fragment:
#   %cat_20 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_103, %relu_104, %relu_105, %getitem_423], 1), kwargs = {})
triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (192 + y0 + (256*x2) + (200704*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y0) + (200704*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cy/ccy65qcswuvqrezjv2dgq7yuxznvtzfh6awuj6gdc76etne3bkpz.py
# Topologically Sorted Source Nodes: [out_165, out_166, out_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_165 => add_275, mul_337, mul_338, sub_112
#   out_166 => add_276
#   out_167 => relu_106
# Graph fragment:
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_897), kwargs = {})
#   %mul_337 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_899), kwargs = {})
#   %mul_338 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_337, %unsqueeze_901), kwargs = {})
#   %add_275 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_338, %unsqueeze_903), kwargs = {})
#   %add_276 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_275, %relu_101), kwargs = {})
#   %relu_106 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_276,), kwargs = {})
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
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ru/crugltb5srnk3xqg53wv67rgmf4pbcti5vtufibpkoxkwqupl6uk.py
# Topologically Sorted Source Nodes: [out_173, out_174, out_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_173 => add_288, mul_352, mul_353, sub_117
#   out_174 => add_289
#   out_175 => relu_111
# Graph fragment:
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_288 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %add_289 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_288, %relu_106), kwargs = {})
#   %relu_111 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_289,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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
    tmp16 = tl.load(in_ptr4 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ay/cayyei4mg322p4os4mgytq6f3inbqzb64msic2bw7hufdiu7sjhs.py
# Topologically Sorted Source Nodes: [out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_185 => add_304, mul_370, mul_371, sub_123
#   out_186 => relu_117
# Graph fragment:
#   %sub_123 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_123, %unsqueeze_985), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_123, %unsqueeze_987), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_370, %unsqueeze_989), kwargs = {})
#   %add_304 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %unsqueeze_991), kwargs = {})
#   %relu_117 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_304,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_sahanp/ph/cphfzxzoaae5x36ws7yejutyhk5iqg7ia6xl6uc54svfvksypder.py
# Topologically Sorted Source Nodes: [sp_277], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_277 => convolution_124
# Graph fragment:
#   %convolution_124 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_468, %arg196_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vf/cvfbtjto3ba4c6yohbcflyehyquwpciv25pdhr4njojtv4wwj43s.py
# Topologically Sorted Source Nodes: [avg_pool2d_6], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_6 => avg_pool2d_6
# Graph fragment:
#   %avg_pool2d_6 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_483, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_28 = async_compile.triton('triton_poi_fused_avg_pool2d_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 14) % 14
    y0 = yindex % 14
    x3 = xindex
    y4 = (yindex // 14)
    y2 = (yindex // 196)
    y5 = yindex % 196
    tmp0 = (-1) + (2*y1)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*y0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-14464) + x3 + (1024*y0) + (28672*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-13952) + x3 + (1024*y0) + (28672*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-13440) + x3 + (1024*y0) + (28672*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-128) + x3 + (1024*y0) + (28672*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (384 + x3 + (1024*y0) + (28672*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (896 + x3 + (1024*y0) + (28672*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (14208 + x3 + (1024*y0) + (28672*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (14720 + x3 + (1024*y0) + (28672*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (15232 + x3 + (1024*y0) + (28672*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29)))*((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))) + ((-2)*y0*((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))) + ((-2)*y1*((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29)))) + (4*y0*y1) + ((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29))) + ((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (196*x3) + (100352*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/yl/cylfh3z5w24khyleb2xebmnhq46mjmq6vsyiyqtix6ddtkibdv6j.py
# Topologically Sorted Source Nodes: [sp_278, sp_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_278 => add_306, mul_373, mul_374, sub_124
#   sp_279 => relu_118
# Graph fragment:
#   %sub_124 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_993), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_124, %unsqueeze_995), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_373, %unsqueeze_997), kwargs = {})
#   %add_306 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %unsqueeze_999), kwargs = {})
#   %relu_118 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_306,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (y0 + (128*x2) + (25088*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + (196*y0) + (100352*y1)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i6/ci6o7pw34j6j6ug34eakyjeev57c7fl6h6hckn6bj5ihaslhsnx6.py
# Topologically Sorted Source Nodes: [out_188], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_188 => convolution_127
# Graph fragment:
#   %convolution_127 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_23, %arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_30 = async_compile.triton('triton_poi_fused_convolution_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (100352*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xg/cxgvwgahmzqc62tuyt7rdok6y7slncvczxpejsiotugnitp4prii.py
# Topologically Sorted Source Nodes: [out_189, input_14, out_190, out_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_14 => add_314, mul_385, mul_386, sub_128
#   out_189 => add_312, mul_382, mul_383, sub_127
#   out_190 => add_315
#   out_191 => relu_121
# Graph fragment:
#   %sub_127 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1017), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_127, %unsqueeze_1019), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_1021), kwargs = {})
#   %add_312 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_1023), kwargs = {})
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_1025), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %unsqueeze_1029), kwargs = {})
#   %add_314 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %unsqueeze_1031), kwargs = {})
#   %add_315 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_312, %add_314), kwargs = {})
#   %relu_121 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_315,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/id/cidicdib4nkjluzwuerjryfigcft6kzuurredh5zjiufrbuuggoc.py
# Topologically Sorted Source Nodes: [out_193, out_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_193 => add_317, mul_388, mul_389, sub_129
#   out_194 => relu_122
# Graph fragment:
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_1033), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1037), kwargs = {})
#   %add_317 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1039), kwargs = {})
#   %relu_122 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_317,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_sahanp/jc/cjc5czgeeh4i6m2qoyu6unyxdbnz6zyjwgctxnuld2eapp6rnoue.py
# Topologically Sorted Source Nodes: [sp_292], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_292 => add_320
# Graph fragment:
#   %add_320 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_123, %getitem_493), kwargs = {})
triton_poi_fused_add_33 = async_compile.triton('triton_poi_fused_add_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_33(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (128 + x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wn/cwnuutzjtkkmbqvhlpn7lgeyddnyfmpogryxxxj6kaveen3g6xy5.py
# Topologically Sorted Source Nodes: [sp_296], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_296 => add_323
# Graph fragment:
#   %add_323 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_124, %getitem_498), kwargs = {})
triton_poi_fused_add_34 = async_compile.triton('triton_poi_fused_add_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_34(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (100352*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + x2 + (512*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bp/cbpqaywi5jimwex6kw4izkdizqiema4oon2snd3ky3s6y4ib5ss2.py
# Topologically Sorted Source Nodes: [out_195], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_195 => cat_24
# Graph fragment:
#   %cat_24 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_123, %relu_124, %relu_125, %getitem_503], 1), kwargs = {})
triton_poi_fused_cat_35 = async_compile.triton('triton_poi_fused_cat_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (384 + y0 + (512*x2) + (100352*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y0) + (100352*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/so/csofprbecbpcjkp5h2u5b3lsj24pq2zqswwwxv34vxzpexzcrc74.py
# Topologically Sorted Source Nodes: [out_197, out_198, out_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_197 => add_327, mul_400, mul_401, sub_133
#   out_198 => add_328
#   out_199 => relu_126
# Graph fragment:
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_133, %unsqueeze_1065), kwargs = {})
#   %mul_400 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %unsqueeze_1067), kwargs = {})
#   %mul_401 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_400, %unsqueeze_1069), kwargs = {})
#   %add_327 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_401, %unsqueeze_1071), kwargs = {})
#   %add_328 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_327, %relu_121), kwargs = {})
#   %relu_126 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_328,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ii/ciiagd7xwu3abqwfxuzma2r2vomngd7xknmwomwht52fde2looqa.py
# Topologically Sorted Source Nodes: [out_233, out_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_233 => add_382, mul_463, mul_464, sub_154
#   out_234 => relu_147
# Graph fragment:
#   %sub_154 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_154, %unsqueeze_1233), kwargs = {})
#   %mul_463 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_154, %unsqueeze_1235), kwargs = {})
#   %mul_464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_463, %unsqueeze_1237), kwargs = {})
#   %add_382 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_464, %unsqueeze_1239), kwargs = {})
#   %relu_147 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_382,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_sahanp/zu/czulnseovfzqj6plui3t6vez4aqh4pvdz3dspzoe7hrpbvzc6sok.py
# Topologically Sorted Source Nodes: [sp_349], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_349 => convolution_155
# Graph fragment:
#   %convolution_155 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_588, %arg351_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zv/czviolsquwectkgokzy5ejhli3irrfzkcf7f2muvciha64mgbiqg.py
# Topologically Sorted Source Nodes: [avg_pool2d_7], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_7 => avg_pool2d_7
# Graph fragment:
#   %avg_pool2d_7 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_603, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_39 = async_compile.triton('triton_poi_fused_avg_pool2d_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y1 = (yindex // 7) % 7
    y0 = yindex % 7
    x3 = xindex
    y4 = (yindex // 7)
    y2 = (yindex // 49)
    y5 = yindex % 49
    tmp0 = (-1) + (2*y1)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*y0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-14592) + x3 + (2048*y0) + (28672*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-13568) + x3 + (2048*y0) + (28672*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-12544) + x3 + (2048*y0) + (28672*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-256) + x3 + (2048*y0) + (28672*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (768 + x3 + (2048*y0) + (28672*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1792 + x3 + (2048*y0) + (28672*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (14080 + x3 + (2048*y0) + (28672*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (15104 + x3 + (2048*y0) + (28672*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (16128 + x3 + (2048*y0) + (28672*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15)))*((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))) + ((-2)*y0*((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))) + ((-2)*y1*((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15)))) + (4*y0*y1) + ((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15))) + ((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (49*x3) + (50176*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pd/cpdsiyz27eimww5s4uadcfogy7doh5g3hilr76t4mesff3xylyga.py
# Topologically Sorted Source Nodes: [sp_350, sp_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_350 => add_384, mul_466, mul_467, sub_155
#   sp_351 => relu_148
# Graph fragment:
#   %sub_155 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_155, %unsqueeze_1241), kwargs = {})
#   %mul_466 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_155, %unsqueeze_1243), kwargs = {})
#   %mul_467 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_466, %unsqueeze_1245), kwargs = {})
#   %add_384 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_467, %unsqueeze_1247), kwargs = {})
#   %relu_148 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_384,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (y0 + (256*x2) + (12544*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + (49*y0) + (50176*y1)), tmp17, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/5u/c5uqh7mez7y33s5o23ww5dt4hwiwkc2ololrhg4gw3vzh5fihjkh.py
# Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_236 => convolution_158
# Graph fragment:
#   %convolution_158 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg366_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_41 = async_compile.triton('triton_poi_fused_convolution_41', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_41(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (1024*x2) + (50176*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3y/c3ybzhn7z6qmw2ms2lblx6pzp4gnetrh7hjxrr7passm3ianpeit.py
# Topologically Sorted Source Nodes: [out_237, input_16, out_238, out_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_392, mul_478, mul_479, sub_159
#   out_237 => add_390, mul_475, mul_476, sub_158
#   out_238 => add_393
#   out_239 => relu_151
# Graph fragment:
#   %sub_158 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_158, %unsqueeze_1265), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_158, %unsqueeze_1267), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, %unsqueeze_1269), kwargs = {})
#   %add_390 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_476, %unsqueeze_1271), kwargs = {})
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_159, %unsqueeze_1273), kwargs = {})
#   %mul_478 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_478, %unsqueeze_1277), kwargs = {})
#   %add_392 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_479, %unsqueeze_1279), kwargs = {})
#   %add_393 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_390, %add_392), kwargs = {})
#   %relu_151 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_393,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), None)
    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xo/cxomzhfkgw2hiydi3dzteixphuwnix2tjc7bq6lpty5l4uouwsdg.py
# Topologically Sorted Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_241 => add_395, mul_481, mul_482, sub_160
#   out_242 => relu_152
# Graph fragment:
#   %sub_160 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_160, %unsqueeze_1281), kwargs = {})
#   %mul_481 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_160, %unsqueeze_1283), kwargs = {})
#   %mul_482 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_481, %unsqueeze_1285), kwargs = {})
#   %add_395 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_482, %unsqueeze_1287), kwargs = {})
#   %relu_152 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_395,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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


# kernel path: /tmp/torchinductor_sahanp/ws/cwsqfrfhobwmbwyrjsx7hyy3j77527q4wjfe6l4nvpqnjjcxiatw.py
# Topologically Sorted Source Nodes: [sp_364], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_364 => add_398
# Graph fragment:
#   %add_398 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_153, %getitem_613), kwargs = {})
triton_poi_fused_add_44 = async_compile.triton('triton_poi_fused_add_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_44(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (256 + x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ey/cey7j2kj7olxdl4k3ljxtvityotvthhuzoeyfn34qwu7cwfy5y25.py
# Topologically Sorted Source Nodes: [sp_368], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_368 => add_401
# Graph fragment:
#   %add_401 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_154, %getitem_618), kwargs = {})
triton_poi_fused_add_45 = async_compile.triton('triton_poi_fused_add_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_45(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (50176*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (512 + x2 + (1024*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (256*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2c/c2cpgd2jebx3cyj6gp2v4rpayoigqo4uhot7zfo6t5yqi5d5ktl2.py
# Topologically Sorted Source Nodes: [out_243], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_243 => cat_30
# Graph fragment:
#   %cat_30 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_153, %relu_154, %relu_155, %getitem_623], 1), kwargs = {})
triton_poi_fused_cat_46 = async_compile.triton('triton_poi_fused_cat_46', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_46(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (768 + y0 + (1024*x2) + (50176*y1)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y0) + (50176*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g5/cg5g2lkzaaxjlnp4jgir6if7wg7zsakagfqg33hec2ceuicohlkg.py
# Topologically Sorted Source Nodes: [out_245, out_246, out_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_245 => add_405, mul_493, mul_494, sub_164
#   out_246 => add_406
#   out_247 => relu_156
# Graph fragment:
#   %sub_164 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_164, %unsqueeze_1313), kwargs = {})
#   %mul_493 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_164, %unsqueeze_1315), kwargs = {})
#   %mul_494 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_493, %unsqueeze_1317), kwargs = {})
#   %add_405 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_494, %unsqueeze_1319), kwargs = {})
#   %add_406 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_405, %relu_151), kwargs = {})
#   %relu_156 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_406,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/nx/cnxhzjfmkdbe33eeewj226g5vr32ww6btedt4qz7grcgrqqtgrkj.py
# Topologically Sorted Source Nodes: [out_253, out_254, out_255, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   out_253 => add_418, mul_508, mul_509, sub_169
#   out_254 => add_419
#   out_255 => relu_161
#   x_11 => mean_1
# Graph fragment:
#   %sub_169 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_169, %unsqueeze_1353), kwargs = {})
#   %mul_508 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_169, %unsqueeze_1355), kwargs = {})
#   %mul_509 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_508, %unsqueeze_1357), kwargs = {})
#   %add_418 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_509, %unsqueeze_1359), kwargs = {})
#   %add_419 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_418, %relu_156), kwargs = {})
#   %relu_161 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_419,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_161, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_48 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0 + (2048*r2) + (100352*x1)), rmask, other=0.0)
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
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr1 + (x3), tmp25, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg12_1, (32, ), (1, ))
    assert_size_stride(arg13_1, (32, ), (1, ))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (32, ), (1, ))
    assert_size_stride(arg20_1, (32, ), (1, ))
    assert_size_stride(arg21_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg22_1, (32, ), (1, ))
    assert_size_stride(arg23_1, (32, ), (1, ))
    assert_size_stride(arg24_1, (32, ), (1, ))
    assert_size_stride(arg25_1, (32, ), (1, ))
    assert_size_stride(arg26_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg42_1, (32, ), (1, ))
    assert_size_stride(arg43_1, (32, ), (1, ))
    assert_size_stride(arg44_1, (32, ), (1, ))
    assert_size_stride(arg45_1, (32, ), (1, ))
    assert_size_stride(arg46_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg47_1, (32, ), (1, ))
    assert_size_stride(arg48_1, (32, ), (1, ))
    assert_size_stride(arg49_1, (32, ), (1, ))
    assert_size_stride(arg50_1, (32, ), (1, ))
    assert_size_stride(arg51_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg52_1, (32, ), (1, ))
    assert_size_stride(arg53_1, (32, ), (1, ))
    assert_size_stride(arg54_1, (32, ), (1, ))
    assert_size_stride(arg55_1, (32, ), (1, ))
    assert_size_stride(arg56_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg67_1, (32, ), (1, ))
    assert_size_stride(arg68_1, (32, ), (1, ))
    assert_size_stride(arg69_1, (32, ), (1, ))
    assert_size_stride(arg70_1, (32, ), (1, ))
    assert_size_stride(arg71_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg72_1, (32, ), (1, ))
    assert_size_stride(arg73_1, (32, ), (1, ))
    assert_size_stride(arg74_1, (32, ), (1, ))
    assert_size_stride(arg75_1, (32, ), (1, ))
    assert_size_stride(arg76_1, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg77_1, (32, ), (1, ))
    assert_size_stride(arg78_1, (32, ), (1, ))
    assert_size_stride(arg79_1, (32, ), (1, ))
    assert_size_stride(arg80_1, (32, ), (1, ))
    assert_size_stride(arg81_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg92_1, (64, ), (1, ))
    assert_size_stride(arg93_1, (64, ), (1, ))
    assert_size_stride(arg94_1, (64, ), (1, ))
    assert_size_stride(arg95_1, (64, ), (1, ))
    assert_size_stride(arg96_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg97_1, (64, ), (1, ))
    assert_size_stride(arg98_1, (64, ), (1, ))
    assert_size_stride(arg99_1, (64, ), (1, ))
    assert_size_stride(arg100_1, (64, ), (1, ))
    assert_size_stride(arg101_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg102_1, (64, ), (1, ))
    assert_size_stride(arg103_1, (64, ), (1, ))
    assert_size_stride(arg104_1, (64, ), (1, ))
    assert_size_stride(arg105_1, (64, ), (1, ))
    assert_size_stride(arg106_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (256, ), (1, ))
    assert_size_stride(arg120_1, (256, ), (1, ))
    assert_size_stride(arg121_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg122_1, (64, ), (1, ))
    assert_size_stride(arg123_1, (64, ), (1, ))
    assert_size_stride(arg124_1, (64, ), (1, ))
    assert_size_stride(arg125_1, (64, ), (1, ))
    assert_size_stride(arg126_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg127_1, (64, ), (1, ))
    assert_size_stride(arg128_1, (64, ), (1, ))
    assert_size_stride(arg129_1, (64, ), (1, ))
    assert_size_stride(arg130_1, (64, ), (1, ))
    assert_size_stride(arg131_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg132_1, (64, ), (1, ))
    assert_size_stride(arg133_1, (64, ), (1, ))
    assert_size_stride(arg134_1, (64, ), (1, ))
    assert_size_stride(arg135_1, (64, ), (1, ))
    assert_size_stride(arg136_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, ), (1, ))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg147_1, (64, ), (1, ))
    assert_size_stride(arg148_1, (64, ), (1, ))
    assert_size_stride(arg149_1, (64, ), (1, ))
    assert_size_stride(arg150_1, (64, ), (1, ))
    assert_size_stride(arg151_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg152_1, (64, ), (1, ))
    assert_size_stride(arg153_1, (64, ), (1, ))
    assert_size_stride(arg154_1, (64, ), (1, ))
    assert_size_stride(arg155_1, (64, ), (1, ))
    assert_size_stride(arg156_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg157_1, (64, ), (1, ))
    assert_size_stride(arg158_1, (64, ), (1, ))
    assert_size_stride(arg159_1, (64, ), (1, ))
    assert_size_stride(arg160_1, (64, ), (1, ))
    assert_size_stride(arg161_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg167_1, (256, ), (1, ))
    assert_size_stride(arg168_1, (256, ), (1, ))
    assert_size_stride(arg169_1, (256, ), (1, ))
    assert_size_stride(arg170_1, (256, ), (1, ))
    assert_size_stride(arg171_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg172_1, (64, ), (1, ))
    assert_size_stride(arg173_1, (64, ), (1, ))
    assert_size_stride(arg174_1, (64, ), (1, ))
    assert_size_stride(arg175_1, (64, ), (1, ))
    assert_size_stride(arg176_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg177_1, (64, ), (1, ))
    assert_size_stride(arg178_1, (64, ), (1, ))
    assert_size_stride(arg179_1, (64, ), (1, ))
    assert_size_stride(arg180_1, (64, ), (1, ))
    assert_size_stride(arg181_1, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg182_1, (64, ), (1, ))
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (64, ), (1, ))
    assert_size_stride(arg185_1, (64, ), (1, ))
    assert_size_stride(arg186_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg197_1, (128, ), (1, ))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (128, ), (1, ))
    assert_size_stride(arg201_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg202_1, (128, ), (1, ))
    assert_size_stride(arg203_1, (128, ), (1, ))
    assert_size_stride(arg204_1, (128, ), (1, ))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (128, ), (1, ))
    assert_size_stride(arg209_1, (128, ), (1, ))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (512, ), (1, ))
    assert_size_stride(arg224_1, (512, ), (1, ))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg227_1, (128, ), (1, ))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (128, ), (1, ))
    assert_size_stride(arg231_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg232_1, (128, ), (1, ))
    assert_size_stride(arg233_1, (128, ), (1, ))
    assert_size_stride(arg234_1, (128, ), (1, ))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, ), (1, ))
    assert_size_stride(arg239_1, (128, ), (1, ))
    assert_size_stride(arg240_1, (128, ), (1, ))
    assert_size_stride(arg241_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg247_1, (512, ), (1, ))
    assert_size_stride(arg248_1, (512, ), (1, ))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (128, ), (1, ))
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg257_1, (128, ), (1, ))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (128, ), (1, ))
    assert_size_stride(arg261_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (128, ), (1, ))
    assert_size_stride(arg264_1, (128, ), (1, ))
    assert_size_stride(arg265_1, (128, ), (1, ))
    assert_size_stride(arg266_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg272_1, (512, ), (1, ))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg277_1, (128, ), (1, ))
    assert_size_stride(arg278_1, (128, ), (1, ))
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg282_1, (128, ), (1, ))
    assert_size_stride(arg283_1, (128, ), (1, ))
    assert_size_stride(arg284_1, (128, ), (1, ))
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg287_1, (128, ), (1, ))
    assert_size_stride(arg288_1, (128, ), (1, ))
    assert_size_stride(arg289_1, (128, ), (1, ))
    assert_size_stride(arg290_1, (128, ), (1, ))
    assert_size_stride(arg291_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (512, ), (1, ))
    assert_size_stride(arg300_1, (512, ), (1, ))
    assert_size_stride(arg301_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg302_1, (128, ), (1, ))
    assert_size_stride(arg303_1, (128, ), (1, ))
    assert_size_stride(arg304_1, (128, ), (1, ))
    assert_size_stride(arg305_1, (128, ), (1, ))
    assert_size_stride(arg306_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg307_1, (128, ), (1, ))
    assert_size_stride(arg308_1, (128, ), (1, ))
    assert_size_stride(arg309_1, (128, ), (1, ))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg312_1, (128, ), (1, ))
    assert_size_stride(arg313_1, (128, ), (1, ))
    assert_size_stride(arg314_1, (128, ), (1, ))
    assert_size_stride(arg315_1, (128, ), (1, ))
    assert_size_stride(arg316_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (512, ), (1, ))
    assert_size_stride(arg324_1, (512, ), (1, ))
    assert_size_stride(arg325_1, (512, ), (1, ))
    assert_size_stride(arg326_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (128, ), (1, ))
    assert_size_stride(arg329_1, (128, ), (1, ))
    assert_size_stride(arg330_1, (128, ), (1, ))
    assert_size_stride(arg331_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg332_1, (128, ), (1, ))
    assert_size_stride(arg333_1, (128, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg337_1, (128, ), (1, ))
    assert_size_stride(arg338_1, (128, ), (1, ))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (128, ), (1, ))
    assert_size_stride(arg341_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg352_1, (256, ), (1, ))
    assert_size_stride(arg353_1, (256, ), (1, ))
    assert_size_stride(arg354_1, (256, ), (1, ))
    assert_size_stride(arg355_1, (256, ), (1, ))
    assert_size_stride(arg356_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg357_1, (256, ), (1, ))
    assert_size_stride(arg358_1, (256, ), (1, ))
    assert_size_stride(arg359_1, (256, ), (1, ))
    assert_size_stride(arg360_1, (256, ), (1, ))
    assert_size_stride(arg361_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg362_1, (256, ), (1, ))
    assert_size_stride(arg363_1, (256, ), (1, ))
    assert_size_stride(arg364_1, (256, ), (1, ))
    assert_size_stride(arg365_1, (256, ), (1, ))
    assert_size_stride(arg366_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg367_1, (2048, ), (1, ))
    assert_size_stride(arg368_1, (2048, ), (1, ))
    assert_size_stride(arg369_1, (2048, ), (1, ))
    assert_size_stride(arg370_1, (2048, ), (1, ))
    assert_size_stride(arg371_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg372_1, (2048, ), (1, ))
    assert_size_stride(arg373_1, (2048, ), (1, ))
    assert_size_stride(arg374_1, (2048, ), (1, ))
    assert_size_stride(arg375_1, (2048, ), (1, ))
    assert_size_stride(arg376_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg382_1, (256, ), (1, ))
    assert_size_stride(arg383_1, (256, ), (1, ))
    assert_size_stride(arg384_1, (256, ), (1, ))
    assert_size_stride(arg385_1, (256, ), (1, ))
    assert_size_stride(arg386_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg387_1, (256, ), (1, ))
    assert_size_stride(arg388_1, (256, ), (1, ))
    assert_size_stride(arg389_1, (256, ), (1, ))
    assert_size_stride(arg390_1, (256, ), (1, ))
    assert_size_stride(arg391_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg392_1, (256, ), (1, ))
    assert_size_stride(arg393_1, (256, ), (1, ))
    assert_size_stride(arg394_1, (256, ), (1, ))
    assert_size_stride(arg395_1, (256, ), (1, ))
    assert_size_stride(arg396_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg397_1, (2048, ), (1, ))
    assert_size_stride(arg398_1, (2048, ), (1, ))
    assert_size_stride(arg399_1, (2048, ), (1, ))
    assert_size_stride(arg400_1, (2048, ), (1, ))
    assert_size_stride(arg401_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg402_1, (1024, ), (1, ))
    assert_size_stride(arg403_1, (1024, ), (1, ))
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg407_1, (256, ), (1, ))
    assert_size_stride(arg408_1, (256, ), (1, ))
    assert_size_stride(arg409_1, (256, ), (1, ))
    assert_size_stride(arg410_1, (256, ), (1, ))
    assert_size_stride(arg411_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg412_1, (256, ), (1, ))
    assert_size_stride(arg413_1, (256, ), (1, ))
    assert_size_stride(arg414_1, (256, ), (1, ))
    assert_size_stride(arg415_1, (256, ), (1, ))
    assert_size_stride(arg416_1, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg417_1, (256, ), (1, ))
    assert_size_stride(arg418_1, (256, ), (1, ))
    assert_size_stride(arg419_1, (256, ), (1, ))
    assert_size_stride(arg420_1, (256, ), (1, ))
    assert_size_stride(arg421_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg422_1, (2048, ), (1, ))
    assert_size_stride(arg423_1, (2048, ), (1, ))
    assert_size_stride(arg424_1, (2048, ), (1, ))
    assert_size_stride(arg425_1, (2048, ), (1, ))
    assert_size_stride(arg426_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg427_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 49, grid=grid(192, 49), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 1, 7168, 64))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((8, 64, 56, 56), (200704, 1, 3584, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 1605632, grid=grid(1605632), stream=stream0)
        # Topologically Sorted Source Nodes: [out_128], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [out_129, out_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((32, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Topologically Sorted Source Nodes: [sp_193], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg11_1, buf7, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [sp_193], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 32, 56, 56), (401408, 1, 7168, 128), 0), buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf8, (8, 32, 56, 56), (100352, 1, 1792, 32))
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sp_197], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg16_1, buf9, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [sp_197], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 32, 56, 56), (401408, 1, 7168, 128), 32), buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf10, (8, 32, 56, 56), (100352, 1, 1792, 32))
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sp_201], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg21_1, buf11, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [sp_201], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 32, 56, 56), (401408, 1, 7168, 128), 64), buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf12, (8, 32, 56, 56), (100352, 1, 1792, 32))
        buf17 = empty_strided_cuda((8, 128, 56, 56), (401408, 3136, 56, 1), torch.float32)
        buf13 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_6.run(buf6, buf13, 25088, 32, grid=grid(25088, 32), stream=stream0)
        buf14 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_194, sp_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf8, arg12_1, arg13_1, arg14_1, arg15_1, buf14, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf8
        buf15 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        # Topologically Sorted Source Nodes: [sp_198, sp_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf10, arg17_1, arg18_1, arg19_1, arg20_1, buf15, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf10
        buf16 = reinterpret_tensor(buf17, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        # Topologically Sorted Source Nodes: [sp_202, sp_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf12, arg22_1, arg23_1, arg24_1, arg25_1, buf16, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf12
        buf18 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf17, buf18, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del buf13
        del buf14
        del buf15
        del buf16
        del buf17
        # Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg26_1
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf4, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg31_1
        buf21 = buf19; del buf19  # reuse
        buf22 = reinterpret_tensor(buf3, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_133, input_10, out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf21, arg27_1, arg28_1, arg29_1, arg30_1, buf20, arg32_1, arg33_1, arg34_1, arg35_1, buf22, 6422528, grid=grid(6422528), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf20
        del buf21
        # Topologically Sorted Source Nodes: [out_135, out_136], Original ATen: [aten.relu, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg36_1
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [out_137, out_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf24, arg37_1, arg38_1, arg39_1, arg40_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        buf25 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [sp_205], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg41_1, buf25, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [sp_205], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(reinterpret_tensor(buf24, (8, 32, 56, 56), (401408, 1, 7168, 128), 0), buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf26, (8, 32, 56, 56), (100352, 1, 1792, 32))
        buf37 = reinterpret_tensor(buf18, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf18  # reuse
        buf27 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_206, sp_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf26, arg42_1, arg43_1, arg44_1, arg45_1, buf27, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [sp_208], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf27, buf24, buf28, 25088, 32, grid=grid(25088, 32), stream=stream0)
        buf29 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [sp_208, sp_209], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg46_1, buf29, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg46_1
        # Topologically Sorted Source Nodes: [sp_208, sp_209], Original ATen: [aten.add, aten.convolution]
        buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf30, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf28
        buf31 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        # Topologically Sorted Source Nodes: [sp_210, sp_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf30, arg47_1, arg48_1, arg49_1, arg50_1, buf31, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [sp_212], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf31, buf24, buf32, 25088, 32, grid=grid(25088, 32), stream=stream0)
        buf33 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [sp_212, sp_213], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg51_1, buf33, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg51_1
        # Topologically Sorted Source Nodes: [sp_212, sp_213], Original ATen: [aten.add, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf34, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf32
        buf35 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        # Topologically Sorted Source Nodes: [sp_214, sp_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf34, arg52_1, arg53_1, arg54_1, arg55_1, buf35, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf34
        buf36 = reinterpret_tensor(buf37, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Topologically Sorted Source Nodes: [out_139], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf24, buf36, 256, 3136, grid=grid(256, 3136), stream=stream0)
        buf38 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf37, buf38, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del buf27
        del buf31
        del buf35
        del buf36
        del buf37
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg56_1
        buf40 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf40, buf39, arg57_1, arg58_1, arg59_1, arg60_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf39
        # Topologically Sorted Source Nodes: [out_144], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 128, 56, 56), (401408, 1, 7168, 128))
        del arg61_1
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [out_145, out_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf42, arg62_1, arg63_1, arg64_1, arg65_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        buf43 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [sp_217], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg66_1, buf43, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [sp_217], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(reinterpret_tensor(buf42, (8, 32, 56, 56), (401408, 1, 7168, 128), 0), buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf44, (8, 32, 56, 56), (100352, 1, 1792, 32))
        buf55 = reinterpret_tensor(buf38, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf38  # reuse
        buf45 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_218, sp_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf44, arg67_1, arg68_1, arg69_1, arg70_1, buf45, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf46 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [sp_220], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf45, buf42, buf46, 25088, 32, grid=grid(25088, 32), stream=stream0)
        buf47 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [sp_220, sp_221], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg71_1, buf47, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [sp_220, sp_221], Original ATen: [aten.add, aten.convolution]
        buf48 = extern_kernels.convolution(buf46, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf48, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf46
        buf49 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        # Topologically Sorted Source Nodes: [sp_222, sp_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf48, arg72_1, arg73_1, arg74_1, arg75_1, buf49, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf50 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [sp_224], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf49, buf42, buf50, 25088, 32, grid=grid(25088, 32), stream=stream0)
        buf51 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [sp_224, sp_225], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg76_1, buf51, 128, 9, grid=grid(128, 9), stream=stream0)
        del arg76_1
        # Topologically Sorted Source Nodes: [sp_224, sp_225], Original ATen: [aten.add, aten.convolution]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf52, (8, 32, 56, 56), (100352, 1, 1792, 32))
        del buf51
        buf53 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        # Topologically Sorted Source Nodes: [sp_226, sp_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf52, arg77_1, arg78_1, arg79_1, arg80_1, buf53, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf54 = reinterpret_tensor(buf55, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Topologically Sorted Source Nodes: [out_147], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf42, buf54, 256, 3136, grid=grid(256, 3136), stream=stream0)
        buf56 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf55, buf56, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del buf45
        del buf49
        del buf53
        del buf54
        del buf55
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg81_1
        buf58 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [out_149, out_150, out_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf58, buf57, arg82_1, arg83_1, arg84_1, arg85_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del buf57
        # Topologically Sorted Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg86_1
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf60, arg87_1, arg88_1, arg89_1, arg90_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf61 = empty_strided_cuda((64, 8, 3, 3), (72, 1, 24, 8), torch.float32)
        # Topologically Sorted Source Nodes: [sp_229], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg91_1, buf61, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [sp_229], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 64, 56, 56), (802816, 1, 14336, 256), 0), buf61, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf62, (8, 64, 28, 28), (50176, 1, 1792, 64))
        buf63 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [sp_233], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg96_1, buf63, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [sp_233], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 64, 56, 56), (802816, 1, 14336, 256), 64), buf63, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf64, (8, 64, 28, 28), (50176, 1, 1792, 64))
        buf65 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [sp_237], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg101_1, buf65, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [sp_237], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 64, 56, 56), (802816, 1, 14336, 256), 128), buf65, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf66, (8, 64, 28, 28), (50176, 1, 1792, 64))
        buf71 = reinterpret_tensor(buf4, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf4  # reuse
        buf67 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_16.run(buf60, buf67, 6272, 64, grid=grid(6272, 64), stream=stream0)
        del buf60
        buf68 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_230, sp_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf62, arg92_1, arg93_1, arg94_1, arg95_1, buf68, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf62
        buf69 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_234, sp_235], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf64, arg97_1, arg98_1, arg99_1, arg100_1, buf69, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf64
        buf70 = reinterpret_tensor(buf71, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Topologically Sorted Source Nodes: [sp_238, sp_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf66, arg102_1, arg103_1, arg104_1, arg105_1, buf70, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del buf66
        buf72 = empty_strided_cuda((8, 256, 28, 28), (200704, 1, 7168, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf71, buf72, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf67
        del buf68
        del buf69
        del buf70
        del buf71
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg106_1
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf58, arg111_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg111_1
        del buf58
        buf75 = buf73; del buf73  # reuse
        buf76 = reinterpret_tensor(buf56, (8, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [out_157, input_12, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf75, arg107_1, arg108_1, arg109_1, arg110_1, buf74, arg112_1, arg113_1, arg114_1, arg115_1, buf76, 3211264, grid=grid(3211264), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        del buf74
        del buf75
        # Topologically Sorted Source Nodes: [out_159, out_160], Original ATen: [aten.relu, aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg116_1
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf78, arg117_1, arg118_1, arg119_1, arg120_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf79 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [sp_241], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg121_1, buf79, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [sp_241], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 64, 28, 28), (200704, 1, 7168, 256), 0), buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf80, (8, 64, 28, 28), (50176, 1, 1792, 64))
        buf91 = reinterpret_tensor(buf72, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf72  # reuse
        buf81 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_242, sp_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf80, arg122_1, arg123_1, arg124_1, arg125_1, buf81, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf82 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [sp_244], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf81, buf78, buf82, 6272, 64, grid=grid(6272, 64), stream=stream0)
        buf83 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [sp_244, sp_245], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg126_1, buf83, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [sp_244, sp_245], Original ATen: [aten.add, aten.convolution]
        buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf84, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del buf82
        buf85 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_246, sp_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf84, arg127_1, arg128_1, arg129_1, arg130_1, buf85, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf86 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [sp_248], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf85, buf78, buf86, 6272, 64, grid=grid(6272, 64), stream=stream0)
        buf87 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [sp_248, sp_249], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg131_1, buf87, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [sp_248, sp_249], Original ATen: [aten.add, aten.convolution]
        buf88 = extern_kernels.convolution(buf86, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf88, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del buf86
        buf89 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Topologically Sorted Source Nodes: [sp_250, sp_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf88, arg132_1, arg133_1, arg134_1, arg135_1, buf89, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del buf88
        buf90 = reinterpret_tensor(buf91, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Topologically Sorted Source Nodes: [out_163], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf78, buf90, 512, 784, grid=grid(512, 784), stream=stream0)
        buf92 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_164], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf91, buf92, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf81
        del buf85
        del buf89
        del buf90
        del buf91
        # Topologically Sorted Source Nodes: [out_164], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg136_1
        buf94 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [out_165, out_166, out_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf94, buf93, arg137_1, arg138_1, arg139_1, arg140_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf93
        # Topologically Sorted Source Nodes: [out_168], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg141_1
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [out_169, out_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf96, arg142_1, arg143_1, arg144_1, arg145_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        buf97 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [sp_253], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg146_1, buf97, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [sp_253], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(reinterpret_tensor(buf96, (8, 64, 28, 28), (200704, 1, 7168, 256), 0), buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf98, (8, 64, 28, 28), (50176, 1, 1792, 64))
        buf109 = reinterpret_tensor(buf92, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf92  # reuse
        buf99 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_254, sp_255], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf98, arg147_1, arg148_1, arg149_1, arg150_1, buf99, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf100 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [sp_256], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf99, buf96, buf100, 6272, 64, grid=grid(6272, 64), stream=stream0)
        buf101 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [sp_256, sp_257], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg151_1, buf101, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg151_1
        # Topologically Sorted Source Nodes: [sp_256, sp_257], Original ATen: [aten.add, aten.convolution]
        buf102 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf102, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del buf100
        buf103 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_258, sp_259], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf102, arg152_1, arg153_1, arg154_1, arg155_1, buf103, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        buf104 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [sp_260], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf103, buf96, buf104, 6272, 64, grid=grid(6272, 64), stream=stream0)
        buf105 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [sp_260, sp_261], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg156_1, buf105, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg156_1
        # Topologically Sorted Source Nodes: [sp_260, sp_261], Original ATen: [aten.add, aten.convolution]
        buf106 = extern_kernels.convolution(buf104, buf105, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf106, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del buf104
        buf107 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Topologically Sorted Source Nodes: [sp_262, sp_263], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf106, arg157_1, arg158_1, arg159_1, arg160_1, buf107, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        del buf106
        buf108 = reinterpret_tensor(buf109, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Topologically Sorted Source Nodes: [out_171], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf96, buf108, 512, 784, grid=grid(512, 784), stream=stream0)
        buf110 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [out_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf109, buf110, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf103
        del buf107
        del buf108
        del buf109
        del buf99
        # Topologically Sorted Source Nodes: [out_172], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg161_1
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [out_173, out_174, out_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf112, arg162_1, arg163_1, arg164_1, arg165_1, buf94, 3211264, grid=grid(3211264), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf94
        # Topologically Sorted Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 256, 28, 28), (200704, 1, 7168, 256))
        del arg166_1
        buf114 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [out_177, out_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf114, arg167_1, arg168_1, arg169_1, arg170_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        buf115 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [sp_265], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg171_1, buf115, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg171_1
        # Topologically Sorted Source Nodes: [sp_265], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(reinterpret_tensor(buf114, (8, 64, 28, 28), (200704, 1, 7168, 256), 0), buf115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf116, (8, 64, 28, 28), (50176, 1, 1792, 64))
        buf127 = reinterpret_tensor(buf110, (8, 256, 28, 28), (200704, 784, 28, 1), 0); del buf110  # reuse
        buf117 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_266, sp_267], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf116, arg172_1, arg173_1, arg174_1, arg175_1, buf117, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf118 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [sp_268], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf117, buf114, buf118, 6272, 64, grid=grid(6272, 64), stream=stream0)
        buf119 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [sp_268, sp_269], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg176_1, buf119, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [sp_268, sp_269], Original ATen: [aten.add, aten.convolution]
        buf120 = extern_kernels.convolution(buf118, buf119, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf120, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del buf118
        buf121 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_270, sp_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf120, arg177_1, arg178_1, arg179_1, arg180_1, buf121, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        buf122 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [sp_272], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf121, buf114, buf122, 6272, 64, grid=grid(6272, 64), stream=stream0)
        buf123 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [sp_272, sp_273], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg181_1, buf123, 512, 9, grid=grid(512, 9), stream=stream0)
        del arg181_1
        # Topologically Sorted Source Nodes: [sp_272, sp_273], Original ATen: [aten.add, aten.convolution]
        buf124 = extern_kernels.convolution(buf122, buf123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf124, (8, 64, 28, 28), (50176, 1, 1792, 64))
        del buf123
        buf125 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        # Topologically Sorted Source Nodes: [sp_274, sp_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf124, arg182_1, arg183_1, arg184_1, arg185_1, buf125, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf126 = reinterpret_tensor(buf127, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Topologically Sorted Source Nodes: [out_179], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf114, buf126, 512, 784, grid=grid(512, 784), stream=stream0)
        buf128 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf127, buf128, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del buf117
        del buf121
        del buf125
        del buf126
        del buf127
        # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg186_1
        buf130 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [out_181, out_182, out_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf130, buf129, arg187_1, arg188_1, arg189_1, arg190_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf129
        # Topologically Sorted Source Nodes: [out_184], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg191_1
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf132, arg192_1, arg193_1, arg194_1, arg195_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        buf133 = empty_strided_cuda((128, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Topologically Sorted Source Nodes: [sp_277], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg196_1, buf133, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg196_1
        # Topologically Sorted Source Nodes: [sp_277], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 128, 28, 28), (401408, 1, 14336, 512), 0), buf133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf134, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf135 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [sp_281], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg201_1, buf135, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg201_1
        # Topologically Sorted Source Nodes: [sp_281], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 128, 28, 28), (401408, 1, 14336, 512), 128), buf135, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf136, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf137 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [sp_285], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg206_1, buf137, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg206_1
        # Topologically Sorted Source Nodes: [sp_285], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 128, 28, 28), (401408, 1, 14336, 512), 256), buf137, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf138, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf143 = reinterpret_tensor(buf52, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf52  # reuse
        buf139 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_6], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_28.run(buf132, buf139, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del buf132
        buf140 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_278, sp_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf134, arg197_1, arg198_1, arg199_1, arg200_1, buf140, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf134
        buf141 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_282, sp_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf136, arg202_1, arg203_1, arg204_1, arg205_1, buf141, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del buf136
        buf142 = reinterpret_tensor(buf143, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_286, sp_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf138, arg207_1, arg208_1, arg209_1, arg210_1, buf142, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        del buf138
        buf144 = reinterpret_tensor(buf50, (8, 512, 14, 14), (100352, 1, 7168, 512), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [out_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf143, buf144, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf139
        del buf140
        del buf141
        del buf142
        del buf143
        # Topologically Sorted Source Nodes: [out_188], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg211_1
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf130, arg216_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg216_1
        del buf130
        buf147 = buf145; del buf145  # reuse
        buf148 = reinterpret_tensor(buf128, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [out_189, input_14, out_190, out_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_31.run(buf147, arg212_1, arg213_1, arg214_1, arg215_1, buf146, arg217_1, arg218_1, arg219_1, arg220_1, buf148, 1605632, grid=grid(1605632), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        del buf146
        del buf147
        # Topologically Sorted Source Nodes: [out_191, out_192], Original ATen: [aten.relu, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg221_1
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [out_193, out_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf150, arg222_1, arg223_1, arg224_1, arg225_1, 802816, grid=grid(802816), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf151 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [sp_289], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg226_1, buf151, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg226_1
        # Topologically Sorted Source Nodes: [sp_289], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(reinterpret_tensor(buf150, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf152, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf163 = reinterpret_tensor(buf144, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf144  # reuse
        buf153 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_290, sp_291], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf152, arg227_1, arg228_1, arg229_1, arg230_1, buf153, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        buf154 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [sp_292], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf153, buf150, buf154, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf155 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [sp_292, sp_293], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg231_1, buf155, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg231_1
        # Topologically Sorted Source Nodes: [sp_292, sp_293], Original ATen: [aten.add, aten.convolution]
        buf156 = extern_kernels.convolution(buf154, buf155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf156, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf154
        buf157 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_294, sp_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf156, arg232_1, arg233_1, arg234_1, arg235_1, buf157, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        buf158 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [sp_296], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf157, buf150, buf158, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf159 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [sp_296, sp_297], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg236_1, buf159, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg236_1
        # Topologically Sorted Source Nodes: [sp_296, sp_297], Original ATen: [aten.add, aten.convolution]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf160, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf158
        buf161 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_298, sp_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf160, arg237_1, arg238_1, arg239_1, arg240_1, buf161, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf160
        buf162 = reinterpret_tensor(buf163, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [out_195], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf150, buf162, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf164 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [out_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf163, buf164, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf153
        del buf157
        del buf161
        del buf162
        del buf163
        # Topologically Sorted Source Nodes: [out_196], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg241_1
        buf166 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [out_197, out_198, out_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf166, buf165, arg242_1, arg243_1, arg244_1, arg245_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del buf165
        # Topologically Sorted Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg246_1
        buf168 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [out_201, out_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf168, arg247_1, arg248_1, arg249_1, arg250_1, 802816, grid=grid(802816), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        buf169 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [sp_301], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg251_1, buf169, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg251_1
        # Topologically Sorted Source Nodes: [sp_301], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(reinterpret_tensor(buf168, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf170, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf181 = reinterpret_tensor(buf164, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf164  # reuse
        buf171 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_302, sp_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf170, arg252_1, arg253_1, arg254_1, arg255_1, buf171, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        buf172 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [sp_304], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf171, buf168, buf172, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf173 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [sp_304, sp_305], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg256_1, buf173, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg256_1
        # Topologically Sorted Source Nodes: [sp_304, sp_305], Original ATen: [aten.add, aten.convolution]
        buf174 = extern_kernels.convolution(buf172, buf173, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf174, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf172
        buf175 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_306, sp_307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf174, arg257_1, arg258_1, arg259_1, arg260_1, buf175, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        buf176 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [sp_308], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf175, buf168, buf176, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf177 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [sp_308, sp_309], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg261_1, buf177, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [sp_308, sp_309], Original ATen: [aten.add, aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf178, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf176
        buf179 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_310, sp_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf178, arg262_1, arg263_1, arg264_1, arg265_1, buf179, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        del buf178
        buf180 = reinterpret_tensor(buf181, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [out_203], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf168, buf180, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf182 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [out_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf181, buf182, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf171
        del buf175
        del buf179
        del buf180
        del buf181
        # Topologically Sorted Source Nodes: [out_204], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg266_1
        buf184 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [out_205, out_206, out_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf184, buf183, arg267_1, arg268_1, arg269_1, arg270_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        del buf183
        # Topologically Sorted Source Nodes: [out_208], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg271_1
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [out_209, out_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf186, arg272_1, arg273_1, arg274_1, arg275_1, 802816, grid=grid(802816), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf187 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [sp_313], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg276_1, buf187, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg276_1
        # Topologically Sorted Source Nodes: [sp_313], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf188, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf199 = reinterpret_tensor(buf182, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf182  # reuse
        buf189 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_314, sp_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf188, arg277_1, arg278_1, arg279_1, arg280_1, buf189, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        buf190 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [sp_316], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf189, buf186, buf190, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf191 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [sp_316, sp_317], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg281_1, buf191, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg281_1
        # Topologically Sorted Source Nodes: [sp_316, sp_317], Original ATen: [aten.add, aten.convolution]
        buf192 = extern_kernels.convolution(buf190, buf191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf192, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf190
        buf193 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_318, sp_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf192, arg282_1, arg283_1, arg284_1, arg285_1, buf193, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        buf194 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [sp_320], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf193, buf186, buf194, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf195 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [sp_320, sp_321], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg286_1, buf195, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg286_1
        # Topologically Sorted Source Nodes: [sp_320, sp_321], Original ATen: [aten.add, aten.convolution]
        buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf196, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf194
        buf197 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_322, sp_323], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf196, arg287_1, arg288_1, arg289_1, arg290_1, buf197, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf196
        buf198 = reinterpret_tensor(buf199, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [out_211], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf186, buf198, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf200 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [out_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf199, buf200, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf189
        del buf193
        del buf197
        del buf198
        del buf199
        # Topologically Sorted Source Nodes: [out_212], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg291_1
        buf202 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [out_213, out_214, out_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf202, buf201, arg292_1, arg293_1, arg294_1, arg295_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        del buf201
        # Topologically Sorted Source Nodes: [out_216], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg296_1
        buf204 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [out_217, out_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf204, arg297_1, arg298_1, arg299_1, arg300_1, 802816, grid=grid(802816), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        buf205 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [sp_325], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg301_1, buf205, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg301_1
        # Topologically Sorted Source Nodes: [sp_325], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(reinterpret_tensor(buf204, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf206, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf217 = reinterpret_tensor(buf200, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf200  # reuse
        buf207 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_326, sp_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf206, arg302_1, arg303_1, arg304_1, arg305_1, buf207, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        buf208 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [sp_328], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf207, buf204, buf208, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf209 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [sp_328, sp_329], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg306_1, buf209, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg306_1
        # Topologically Sorted Source Nodes: [sp_328, sp_329], Original ATen: [aten.add, aten.convolution]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf210, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf208
        buf211 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_330, sp_331], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf210, arg307_1, arg308_1, arg309_1, arg310_1, buf211, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        buf212 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [sp_332], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf211, buf204, buf212, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf213 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [sp_332, sp_333], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg311_1, buf213, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [sp_332, sp_333], Original ATen: [aten.add, aten.convolution]
        buf214 = extern_kernels.convolution(buf212, buf213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf214, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf212
        buf215 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_334, sp_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf214, arg312_1, arg313_1, arg314_1, arg315_1, buf215, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        del buf214
        buf216 = reinterpret_tensor(buf217, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [out_219], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf204, buf216, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf218 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [out_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf217, buf218, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf207
        del buf211
        del buf215
        del buf216
        del buf217
        # Topologically Sorted Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg316_1
        buf220 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [out_221, out_222, out_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf220, buf219, arg317_1, arg318_1, arg319_1, arg320_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        del buf219
        # Topologically Sorted Source Nodes: [out_224], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 512, 14, 14), (100352, 1, 7168, 512))
        del arg321_1
        buf222 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [out_225, out_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf222, arg322_1, arg323_1, arg324_1, arg325_1, 802816, grid=grid(802816), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        buf223 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [sp_337], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg326_1, buf223, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg326_1
        # Topologically Sorted Source Nodes: [sp_337], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(reinterpret_tensor(buf222, (8, 128, 14, 14), (100352, 1, 7168, 512), 0), buf223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf224, (8, 128, 14, 14), (25088, 1, 1792, 128))
        buf235 = reinterpret_tensor(buf218, (8, 512, 14, 14), (100352, 196, 14, 1), 0); del buf218  # reuse
        buf225 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_338, sp_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf224, arg327_1, arg328_1, arg329_1, arg330_1, buf225, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        buf226 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [sp_340], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf225, buf222, buf226, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf227 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [sp_340, sp_341], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg331_1, buf227, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg331_1
        # Topologically Sorted Source Nodes: [sp_340, sp_341], Original ATen: [aten.add, aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf228, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf226
        buf229 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_342, sp_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf228, arg332_1, arg333_1, arg334_1, arg335_1, buf229, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        buf230 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [sp_344], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf229, buf222, buf230, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf231 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [sp_344, sp_345], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg336_1, buf231, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg336_1
        # Topologically Sorted Source Nodes: [sp_344, sp_345], Original ATen: [aten.add, aten.convolution]
        buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf232, (8, 128, 14, 14), (25088, 1, 1792, 128))
        del buf230
        del buf231
        buf233 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        # Topologically Sorted Source Nodes: [sp_346, sp_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf232, arg337_1, arg338_1, arg339_1, arg340_1, buf233, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        del buf232
        buf234 = reinterpret_tensor(buf235, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Topologically Sorted Source Nodes: [out_227], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf222, buf234, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf236 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [out_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf235, buf236, 4096, 196, grid=grid(4096, 196), stream=stream0)
        del buf225
        del buf229
        del buf233
        del buf234
        del buf235
        # Topologically Sorted Source Nodes: [out_228], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg341_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg341_1
        buf238 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [out_229, out_230, out_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf238, buf237, arg342_1, arg343_1, arg344_1, arg345_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del buf237
        # Topologically Sorted Source Nodes: [out_232], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg346_1
        buf240 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [out_233, out_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf240, arg347_1, arg348_1, arg349_1, arg350_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        buf241 = empty_strided_cuda((256, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [sp_349], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg351_1, buf241, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg351_1
        # Topologically Sorted Source Nodes: [sp_349], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(reinterpret_tensor(buf240, (8, 256, 14, 14), (200704, 1, 14336, 1024), 0), buf241, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf242, (8, 256, 7, 7), (12544, 1, 1792, 256))
        buf243 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [sp_353], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg356_1, buf243, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg356_1
        # Topologically Sorted Source Nodes: [sp_353], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(reinterpret_tensor(buf240, (8, 256, 14, 14), (200704, 1, 14336, 1024), 256), buf243, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf244, (8, 256, 7, 7), (12544, 1, 1792, 256))
        buf245 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [sp_357], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg361_1, buf245, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg361_1
        # Topologically Sorted Source Nodes: [sp_357], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(reinterpret_tensor(buf240, (8, 256, 14, 14), (200704, 1, 14336, 1024), 512), buf245, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf246, (8, 256, 7, 7), (12544, 1, 1792, 256))
        buf251 = reinterpret_tensor(buf124, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf124  # reuse
        buf247 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_7], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_39.run(buf240, buf247, 392, 256, grid=grid(392, 256), stream=stream0)
        del buf240
        buf248 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_350, sp_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf242, arg352_1, arg353_1, arg354_1, arg355_1, buf248, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        del buf242
        buf249 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        # Topologically Sorted Source Nodes: [sp_354, sp_355], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf244, arg357_1, arg358_1, arg359_1, arg360_1, buf249, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        del buf244
        buf250 = reinterpret_tensor(buf251, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_358, sp_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf246, arg362_1, arg363_1, arg364_1, arg365_1, buf250, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del buf246
        buf252 = reinterpret_tensor(buf122, (8, 1024, 7, 7), (50176, 1, 7168, 1024), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf251, buf252, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del buf247
        del buf248
        del buf249
        del buf250
        del buf251
        # Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, arg366_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg366_1
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf238, arg371_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg371_1
        del buf238
        buf255 = buf253; del buf253  # reuse
        buf256 = reinterpret_tensor(buf236, (8, 2048, 7, 7), (100352, 1, 14336, 2048), 0); del buf236  # reuse
        # Topologically Sorted Source Nodes: [out_237, input_16, out_238, out_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42.run(buf255, arg367_1, arg368_1, arg369_1, arg370_1, buf254, arg372_1, arg373_1, arg374_1, arg375_1, buf256, 802816, grid=grid(802816), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        del arg370_1
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        del buf254
        del buf255
        # Topologically Sorted Source Nodes: [out_239, out_240], Original ATen: [aten.relu, aten.convolution]
        buf257 = extern_kernels.convolution(buf256, arg376_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg376_1
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf258, arg377_1, arg378_1, arg379_1, arg380_1, 401408, grid=grid(401408), stream=stream0)
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        buf259 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [sp_361], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg381_1, buf259, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg381_1
        # Topologically Sorted Source Nodes: [sp_361], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(reinterpret_tensor(buf258, (8, 256, 7, 7), (50176, 1, 7168, 1024), 0), buf259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf260, (8, 256, 7, 7), (12544, 1, 1792, 256))
        buf271 = reinterpret_tensor(buf252, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf252  # reuse
        buf261 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_362, sp_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf260, arg382_1, arg383_1, arg384_1, arg385_1, buf261, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        buf262 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [sp_364], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf261, buf258, buf262, 392, 256, grid=grid(392, 256), stream=stream0)
        buf263 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [sp_364, sp_365], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg386_1, buf263, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg386_1
        # Topologically Sorted Source Nodes: [sp_364, sp_365], Original ATen: [aten.add, aten.convolution]
        buf264 = extern_kernels.convolution(buf262, buf263, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf264, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del buf262
        buf265 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        # Topologically Sorted Source Nodes: [sp_366, sp_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf264, arg387_1, arg388_1, arg389_1, arg390_1, buf265, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        buf266 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [sp_368], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf265, buf258, buf266, 392, 256, grid=grid(392, 256), stream=stream0)
        buf267 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [sp_368, sp_369], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg391_1, buf267, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg391_1
        # Topologically Sorted Source Nodes: [sp_368, sp_369], Original ATen: [aten.add, aten.convolution]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf268, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del buf266
        buf269 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_370, sp_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf268, arg392_1, arg393_1, arg394_1, arg395_1, buf269, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        del buf268
        buf270 = reinterpret_tensor(buf271, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Topologically Sorted Source Nodes: [out_243], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf258, buf270, 2048, 49, grid=grid(2048, 49), stream=stream0)
        buf272 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [out_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf271, buf272, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del buf261
        del buf265
        del buf269
        del buf270
        del buf271
        # Topologically Sorted Source Nodes: [out_244], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, arg396_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg396_1
        buf274 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [out_245, out_246, out_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf274, buf273, arg397_1, arg398_1, arg399_1, arg400_1, 802816, grid=grid(802816), stream=stream0)
        del arg397_1
        del arg398_1
        del arg399_1
        del arg400_1
        del buf273
        # Topologically Sorted Source Nodes: [out_248], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, arg401_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
        del arg401_1
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [out_249, out_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf276, arg402_1, arg403_1, arg404_1, arg405_1, 401408, grid=grid(401408), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        buf277 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [sp_373], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg406_1, buf277, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg406_1
        # Topologically Sorted Source Nodes: [sp_373], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(reinterpret_tensor(buf276, (8, 256, 7, 7), (50176, 1, 7168, 1024), 0), buf277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf278, (8, 256, 7, 7), (12544, 1, 1792, 256))
        buf289 = reinterpret_tensor(buf272, (8, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf272  # reuse
        buf279 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_374, sp_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf278, arg407_1, arg408_1, arg409_1, arg410_1, buf279, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        del arg410_1
        buf280 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [sp_376], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf279, buf276, buf280, 392, 256, grid=grid(392, 256), stream=stream0)
        buf281 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [sp_376, sp_377], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg411_1, buf281, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg411_1
        # Topologically Sorted Source Nodes: [sp_376, sp_377], Original ATen: [aten.add, aten.convolution]
        buf282 = extern_kernels.convolution(buf280, buf281, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf282, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del buf280
        buf283 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        # Topologically Sorted Source Nodes: [sp_378, sp_379], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf282, arg412_1, arg413_1, arg414_1, arg415_1, buf283, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        buf284 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [sp_380], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf283, buf276, buf284, 392, 256, grid=grid(392, 256), stream=stream0)
        buf285 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [sp_380, sp_381], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg416_1, buf285, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg416_1
        # Topologically Sorted Source Nodes: [sp_380, sp_381], Original ATen: [aten.add, aten.convolution]
        buf286 = extern_kernels.convolution(buf284, buf285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf286, (8, 256, 7, 7), (12544, 1, 1792, 256))
        del buf284
        del buf285
        buf287 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Topologically Sorted Source Nodes: [sp_382, sp_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf286, arg417_1, arg418_1, arg419_1, arg420_1, buf287, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg417_1
        del arg418_1
        del arg419_1
        del arg420_1
        del buf286
        buf288 = reinterpret_tensor(buf289, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Topologically Sorted Source Nodes: [out_251], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf276, buf288, 2048, 49, grid=grid(2048, 49), stream=stream0)
        buf290 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [out_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf289, buf290, 8192, 49, grid=grid(8192, 49), stream=stream0)
        del buf279
        del buf283
        del buf287
        del buf288
        del buf289
        # Topologically Sorted Source Nodes: [out_252], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, arg421_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg421_1
        del buf290
        buf293 = empty_strided_cuda((8, 2048, 1, 1), (2048, 1, 16384, 16384), torch.float32)
        # Topologically Sorted Source Nodes: [out_253, out_254, out_255, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_48.run(buf291, arg422_1, arg423_1, arg424_1, arg425_1, buf274, buf293, 16384, 49, grid=grid(16384), stream=stream0)
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        del buf274
        del buf291
        buf294 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg427_1, reinterpret_tensor(buf293, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg426_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf294)
        del arg426_1
        del arg427_1
        del buf293
    return (buf294, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2next50', benchmark_compiled_module)
