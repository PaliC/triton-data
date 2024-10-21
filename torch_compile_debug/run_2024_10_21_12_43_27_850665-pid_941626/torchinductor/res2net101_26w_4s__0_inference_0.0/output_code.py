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
#   x_7 => convolution_170
# Graph fragment:
#   %convolution_170 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
#   x_7 => convolution_170
# Graph fragment:
#   %convolution_170 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
#   x_8 => add_432, mul_511, mul_512, sub_170
#   x_9 => relu_166
# Graph fragment:
#   %sub_170 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_170, %unsqueeze_1361), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_170, %unsqueeze_1363), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_511, %unsqueeze_1365), kwargs = {})
#   %add_432 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_512, %unsqueeze_1367), kwargs = {})
#   %relu_166 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_432,), kwargs = {})
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
#   x_8 => add_432, mul_511, mul_512, sub_170
#   x_9 => relu_166
# Graph fragment:
#   %sub_170 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_170, %unsqueeze_1361), kwargs = {})
#   %mul_511 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_170, %unsqueeze_1363), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_511, %unsqueeze_1365), kwargs = {})
#   %add_432 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_512, %unsqueeze_1367), kwargs = {})
#   %relu_166 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_432,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_166, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/u3/cu3k45t7rrpifuypkng5onncsk3b3wl2dzs6kyhqgo7sw3mj66n6.py
# Topologically Sorted Source Nodes: [out_265, out_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_265 => add_434, mul_514, mul_515, sub_171
#   out_266 => relu_167
# Graph fragment:
#   %sub_171 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_171, %unsqueeze_1369), kwargs = {})
#   %mul_514 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_171, %unsqueeze_1371), kwargs = {})
#   %mul_515 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_514, %unsqueeze_1373), kwargs = {})
#   %add_434 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_515, %unsqueeze_1375), kwargs = {})
#   %relu_167 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_434,), kwargs = {})
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
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 104
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


# kernel path: /tmp/torchinductor_sahanp/ss/cssjrmvb2lpfnzvvilt2qdpbopc3vj5lbz2thuznl36xp6mwk6jx.py
# Topologically Sorted Source Nodes: [sp_397], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_397 => convolution_172
# Graph fragment:
#   %convolution_172 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_668, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 676
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (26*x2) + (234*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/re/crehglnj7axzejz5ahylfg54skzkp2sy4koqoelrxk2hrhumlabg.py
# Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_683, [3, 3], [1, 1], [1, 1]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp11 = tl.load(in_ptr0 + ((-5850) + x3 + (104*y5)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5746) + x3 + (104*y5)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-5642) + x3 + (104*y5)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-26) + x3 + (104*y5)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (78 + x3 + (104*y5)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (182 + x3 + (104*y5)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (5798 + x3 + (104*y5)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5902 + x3 + (104*y5)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (6006 + x3 + (104*y5)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*y0) + ((-1)*y1) + (y0*y1) + (((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57)))*((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))) + ((-1)*y0*((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))) + ((-1)*y1*((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57)))) + ((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57))) + ((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y6 + (3136*x3) + (326144*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/m2/cm2mqkhby5gjfocdm3y2b44dh5cyrlud4idyjbqfdnpch5vixrzl.py
# Topologically Sorted Source Nodes: [sp_398, sp_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_398 => add_436, mul_517, mul_518, sub_172
#   sp_399 => relu_168
# Graph fragment:
#   %sub_172 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_172, %unsqueeze_1377), kwargs = {})
#   %mul_517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_172, %unsqueeze_1379), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_517, %unsqueeze_1381), kwargs = {})
#   %add_436 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_518, %unsqueeze_1383), kwargs = {})
#   %relu_168 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_436,), kwargs = {})
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
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (y0 + (26*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (3136*y0) + (326144*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4i/c4ilviexxcnats6xmubu5qi2y6i5fuouhpnhiidzr4tx5ri2buec.py
# Topologically Sorted Source Nodes: [out_268], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_268 => convolution_175
# Graph fragment:
#   %convolution_175 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    ynumel = 832
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (326144*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xx/cxx4x6yndyw5nnibgriubhzn46v6fabjfvtib4f32ouv5srfthxx.py
# Topologically Sorted Source Nodes: [out_269, input_10, out_270, out_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_444, mul_529, mul_530, sub_176
#   out_269 => add_442, mul_526, mul_527, sub_175
#   out_270 => add_445
#   out_271 => relu_171
# Graph fragment:
#   %sub_175 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_175, %unsqueeze_1401), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_175, %unsqueeze_1403), kwargs = {})
#   %mul_527 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_526, %unsqueeze_1405), kwargs = {})
#   %add_442 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_527, %unsqueeze_1407), kwargs = {})
#   %sub_176 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_176, %unsqueeze_1409), kwargs = {})
#   %mul_529 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_176, %unsqueeze_1411), kwargs = {})
#   %mul_530 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_529, %unsqueeze_1413), kwargs = {})
#   %add_444 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_530, %unsqueeze_1415), kwargs = {})
#   %add_445 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_442, %add_444), kwargs = {})
#   %relu_171 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_445,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/b3/cb3znjayvqwvjtfqbcltgpq6x7dvbgmebsawxnmycaxrouezy764.py
# Topologically Sorted Source Nodes: [sp_412], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_412 => add_450
# Graph fragment:
#   %add_450 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_173, %getitem_693), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_10(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (26 + x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jo/cjot7xrcafwjsczo7qvymphp6q4jwkpjlrtp2cpvc5azxfesiuch.py
# Topologically Sorted Source Nodes: [sp_416], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_416 => add_453
# Graph fragment:
#   %add_453 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_174, %getitem_698), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_11(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 26
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (52 + x2 + (104*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (26*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ln/clnojb2etdbyyncfmttfn4n5esbl5zgkxhs5z5djimy4fck6q2hf.py
# Topologically Sorted Source Nodes: [out_275], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_275 => cat_34
# Graph fragment:
#   %cat_34 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_173, %relu_174, %relu_175, %getitem_703], 1), kwargs = {})
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
    ynumel = 208
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 26
    y1 = (yindex // 26)
    tmp0 = tl.load(in_ptr0 + (78 + y0 + (104*x2) + (326144*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y0) + (326144*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2c/c2czh36ngygmmtpxk25kxdedpmd63eier2lruzeaoobw5th4jt3v.py
# Topologically Sorted Source Nodes: [out_277, out_278, out_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_277 => add_457, mul_544, mul_545, sub_181
#   out_278 => add_458
#   out_279 => relu_176
# Graph fragment:
#   %sub_181 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_181, %unsqueeze_1449), kwargs = {})
#   %mul_544 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_181, %unsqueeze_1451), kwargs = {})
#   %mul_545 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_544, %unsqueeze_1453), kwargs = {})
#   %add_457 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_545, %unsqueeze_1455), kwargs = {})
#   %add_458 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_457, %relu_171), kwargs = {})
#   %relu_176 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_458,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/ll/clliuuyz2rbwoe6zglkgnnmrdvxcszsw4lvhntc2xnidp3aweebm.py
# Topologically Sorted Source Nodes: [out_289, out_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_289 => add_473, mul_562, mul_563, sub_187
#   out_290 => relu_182
# Graph fragment:
#   %sub_187 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_187, %unsqueeze_1497), kwargs = {})
#   %mul_562 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_187, %unsqueeze_1499), kwargs = {})
#   %mul_563 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_562, %unsqueeze_1501), kwargs = {})
#   %add_473 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_563, %unsqueeze_1503), kwargs = {})
#   %relu_182 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_473,), kwargs = {})
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
    xnumel = 5218304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 208
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


# kernel path: /tmp/torchinductor_sahanp/o5/co5kov6oa7yqf3dkqtbo5cvsq2d6lm2qsux5ujni5xynir6hpivk.py
# Topologically Sorted Source Nodes: [sp_433], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_433 => convolution_188
# Graph fragment:
#   %convolution_188 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_728, %arg91_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2704
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (52*x2) + (468*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/2s/c2syiwhzecdphjoffiicyzzsg3irngfhouubyv3uujg3prqmizgw.py
# Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_743, [3, 3], [2, 2], [1, 1]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp11 = tl.load(in_ptr0 + ((-11700) + x3 + (416*y0) + (23296*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-11492) + x3 + (416*y0) + (23296*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-11284) + x3 + (416*y0) + (23296*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-52) + x3 + (416*y0) + (23296*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (156 + x3 + (416*y0) + (23296*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (364 + x3 + (416*y0) + (23296*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (11596 + x3 + (416*y0) + (23296*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (11804 + x3 + (416*y0) + (23296*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (12012 + x3 + (416*y0) + (23296*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57)))*((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))) + ((-2)*y0*((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))) + ((-2)*y1*((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57)))) + (4*y0*y1) + ((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57))) + ((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (784*x3) + (163072*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qv/cqvfj6f4admcvjc52s32hmwe5tcm67akyr6xsezyqnpc7tgfdfr6.py
# Topologically Sorted Source Nodes: [sp_434, sp_435], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_434 => add_475, mul_565, mul_566, sub_188
#   sp_435 => relu_183
# Graph fragment:
#   %sub_188 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_188, %unsqueeze_1505), kwargs = {})
#   %mul_565 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_188, %unsqueeze_1507), kwargs = {})
#   %mul_566 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_565, %unsqueeze_1509), kwargs = {})
#   %add_475 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_566, %unsqueeze_1511), kwargs = {})
#   %relu_183 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_475,), kwargs = {})
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
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (y0 + (52*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (784*y0) + (163072*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zm/czmnr2mugefhrslka75v63lliqvooe5le4yjv5olqpzkc5ykoj4u.py
# Topologically Sorted Source Nodes: [out_292], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_292 => convolution_191
# Graph fragment:
#   %convolution_191 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_36, %arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    ynumel = 1664
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (163072*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ay/cay3mgtetgjqws32liuom3vqgzjtbk7qljzmu5q3nryziq2637wh.py
# Topologically Sorted Source Nodes: [out_293, input_12, out_294, out_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_12 => add_483, mul_577, mul_578, sub_192
#   out_293 => add_481, mul_574, mul_575, sub_191
#   out_294 => add_484
#   out_295 => relu_186
# Graph fragment:
#   %sub_191 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_191, %unsqueeze_1529), kwargs = {})
#   %mul_574 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_191, %unsqueeze_1531), kwargs = {})
#   %mul_575 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_574, %unsqueeze_1533), kwargs = {})
#   %add_481 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_575, %unsqueeze_1535), kwargs = {})
#   %sub_192 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_192, %unsqueeze_1537), kwargs = {})
#   %mul_577 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_192, %unsqueeze_1539), kwargs = {})
#   %mul_578 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_577, %unsqueeze_1541), kwargs = {})
#   %add_483 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_578, %unsqueeze_1543), kwargs = {})
#   %add_484 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_481, %add_483), kwargs = {})
#   %relu_186 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_484,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/xw/cxwwmeudwhqii4ynto636a2swcvag5uaz63gllap47mwwjsde7br.py
# Topologically Sorted Source Nodes: [out_297, out_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_297 => add_486, mul_580, mul_581, sub_193
#   out_298 => relu_187
# Graph fragment:
#   %sub_193 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_193, %unsqueeze_1545), kwargs = {})
#   %mul_580 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_193, %unsqueeze_1547), kwargs = {})
#   %mul_581 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_580, %unsqueeze_1549), kwargs = {})
#   %add_486 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_581, %unsqueeze_1551), kwargs = {})
#   %relu_187 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_486,), kwargs = {})
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
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 208
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


# kernel path: /tmp/torchinductor_sahanp/am/camxsfg475jk2njcm5qbh6zph26af4qfnxh7ov7slcjslbyfwhnj.py
# Topologically Sorted Source Nodes: [sp_448], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_448 => add_489
# Graph fragment:
#   %add_489 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_188, %getitem_753), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_21(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (52 + x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fa/cfaeiyljixwrhl5xbsg3ud2hof724z7ogsmugezacyykypbcki5w.py
# Topologically Sorted Source Nodes: [sp_452], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_452 => add_492
# Graph fragment:
#   %add_492 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_189, %getitem_758), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_22(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 52
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (104 + x2 + (208*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (52*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ki/cki6wj2wznrjys3xkjtobgw5uaiazxfecbtdhhoiz6zqnpygl7ky.py
# Topologically Sorted Source Nodes: [out_299], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_299 => cat_37
# Graph fragment:
#   %cat_37 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_188, %relu_189, %relu_190, %getitem_763], 1), kwargs = {})
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
    ynumel = 416
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 52
    y1 = (yindex // 52)
    tmp0 = tl.load(in_ptr0 + (156 + y0 + (208*x2) + (163072*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y0) + (163072*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/cy/ccy65qcswuvqrezjv2dgq7yuxznvtzfh6awuj6gdc76etne3bkpz.py
# Topologically Sorted Source Nodes: [out_301, out_302, out_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_301 => add_496, mul_592, mul_593, sub_197
#   out_302 => add_497
#   out_303 => relu_191
# Graph fragment:
#   %sub_197 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_197, %unsqueeze_1577), kwargs = {})
#   %mul_592 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_197, %unsqueeze_1579), kwargs = {})
#   %mul_593 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_592, %unsqueeze_1581), kwargs = {})
#   %add_496 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_593, %unsqueeze_1583), kwargs = {})
#   %add_497 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_496, %relu_186), kwargs = {})
#   %relu_191 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_497,), kwargs = {})
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
# Topologically Sorted Source Nodes: [out_309, out_310, out_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_309 => add_509, mul_607, mul_608, sub_202
#   out_310 => add_510
#   out_311 => relu_196
# Graph fragment:
#   %sub_202 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_202, %unsqueeze_1617), kwargs = {})
#   %mul_607 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_202, %unsqueeze_1619), kwargs = {})
#   %mul_608 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_607, %unsqueeze_1621), kwargs = {})
#   %add_509 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_608, %unsqueeze_1623), kwargs = {})
#   %add_510 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_509, %relu_191), kwargs = {})
#   %relu_196 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_510,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/ce/cce2zacp5s3hglnfji2mwxtipp5kq5fixfq7xufmiqeym2pizg5r.py
# Topologically Sorted Source Nodes: [out_321, out_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_321 => add_525, mul_625, mul_626, sub_208
#   out_322 => relu_202
# Graph fragment:
#   %sub_208 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_208, %unsqueeze_1665), kwargs = {})
#   %mul_625 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_208, %unsqueeze_1667), kwargs = {})
#   %mul_626 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_625, %unsqueeze_1669), kwargs = {})
#   %add_525 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_626, %unsqueeze_1671), kwargs = {})
#   %relu_202 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_525,), kwargs = {})
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
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 416
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


# kernel path: /tmp/torchinductor_sahanp/he/chedeas5sj6orftezfpyqdvznkvaxhddpoldv42zkjpfk6glujoi.py
# Topologically Sorted Source Nodes: [sp_481], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_481 => convolution_209
# Graph fragment:
#   %convolution_209 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_808, %arg196_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10816
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (104*x2) + (936*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wd/cwd4ucpsea5ycsf57j452ukr4lusrcpwqi7kvh2q5y4g7iftio2o.py
# Topologically Sorted Source Nodes: [avg_pool2d_6], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_6 => avg_pool2d_6
# Graph fragment:
#   %avg_pool2d_6 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_823, [3, 3], [2, 2], [1, 1]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp11 = tl.load(in_ptr0 + ((-11752) + x3 + (832*y0) + (23296*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-11336) + x3 + (832*y0) + (23296*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-10920) + x3 + (832*y0) + (23296*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-104) + x3 + (832*y0) + (23296*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (312 + x3 + (832*y0) + (23296*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (728 + x3 + (832*y0) + (23296*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (11544 + x3 + (832*y0) + (23296*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (11960 + x3 + (832*y0) + (23296*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (12376 + x3 + (832*y0) + (23296*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29)))*((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))) + ((-2)*y0*((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))) + ((-2)*y1*((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29)))) + (4*y0*y1) + ((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29))) + ((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (196*x3) + (81536*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kc/ckcoabllv77npvvwwjfjvcvw62l3wmecxn37b4ny7svdeu4dgqtj.py
# Topologically Sorted Source Nodes: [sp_482, sp_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_482 => add_527, mul_628, mul_629, sub_209
#   sp_483 => relu_203
# Graph fragment:
#   %sub_209 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_209, %unsqueeze_1673), kwargs = {})
#   %mul_628 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_209, %unsqueeze_1675), kwargs = {})
#   %mul_629 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_628, %unsqueeze_1677), kwargs = {})
#   %add_527 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_629, %unsqueeze_1679), kwargs = {})
#   %relu_203 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_527,), kwargs = {})
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
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (y0 + (104*x2) + (20384*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (196*y0) + (81536*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/pg/cpgmukg5muh5qr3hlzxr35wzwaqjlzsl3tj2oeczq4dgpifoy5kj.py
# Topologically Sorted Source Nodes: [out_324], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_324 => convolution_212
# Graph fragment:
#   %convolution_212 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_40, %arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    ynumel = 3328
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 416
    y1 = (yindex // 416)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (416*x2) + (81536*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xg/cxgvwgahmzqc62tuyt7rdok6y7slncvczxpejsiotugnitp4prii.py
# Topologically Sorted Source Nodes: [out_325, input_14, out_326, out_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_14 => add_535, mul_640, mul_641, sub_213
#   out_325 => add_533, mul_637, mul_638, sub_212
#   out_326 => add_536
#   out_327 => relu_206
# Graph fragment:
#   %sub_212 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_212, %unsqueeze_1697), kwargs = {})
#   %mul_637 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_212, %unsqueeze_1699), kwargs = {})
#   %mul_638 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_637, %unsqueeze_1701), kwargs = {})
#   %add_533 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_638, %unsqueeze_1703), kwargs = {})
#   %sub_213 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_213, %unsqueeze_1705), kwargs = {})
#   %mul_640 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_213, %unsqueeze_1707), kwargs = {})
#   %mul_641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_640, %unsqueeze_1709), kwargs = {})
#   %add_535 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_641, %unsqueeze_1711), kwargs = {})
#   %add_536 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_533, %add_535), kwargs = {})
#   %relu_206 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_536,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/d5/cd5qz36hv2nnpohsri2divxpxsqz5kx7nqcldjp7qdgbb4p4woop.py
# Topologically Sorted Source Nodes: [out_329, out_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_329 => add_538, mul_643, mul_644, sub_214
#   out_330 => relu_207
# Graph fragment:
#   %sub_214 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_214, %unsqueeze_1713), kwargs = {})
#   %mul_643 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_214, %unsqueeze_1715), kwargs = {})
#   %mul_644 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_643, %unsqueeze_1717), kwargs = {})
#   %add_538 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_644, %unsqueeze_1719), kwargs = {})
#   %relu_207 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_538,), kwargs = {})
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
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 416
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


# kernel path: /tmp/torchinductor_sahanp/25/c25d4ivdm5k6mmi24rsd54ixoan6tj7fpnogerj3ae5taoltfgrd.py
# Topologically Sorted Source Nodes: [sp_496], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_496 => add_541
# Graph fragment:
#   %add_541 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_208, %getitem_833), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_33(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (104 + x2 + (416*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/v3/cv3ggrtur2t6codsqzewm2j3w52qtouniktka4wszssjzfa3rv3n.py
# Topologically Sorted Source Nodes: [sp_500], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_500 => add_544
# Graph fragment:
#   %add_544 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_209, %getitem_838), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_34(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 104
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (208 + x2 + (416*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (104*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hc/chcokusl3ugdt5eezbzadspvz32grnmbqedc4a2tw57zwhdaxzad.py
# Topologically Sorted Source Nodes: [out_331], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_331 => cat_41
# Graph fragment:
#   %cat_41 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_208, %relu_209, %relu_210, %getitem_843], 1), kwargs = {})
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
    ynumel = 832
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 104
    y1 = (yindex // 104)
    tmp0 = tl.load(in_ptr0 + (312 + y0 + (416*x2) + (81536*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y0) + (81536*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/so/csofprbecbpcjkp5h2u5b3lsj24pq2zqswwwxv34vxzpexzcrc74.py
# Topologically Sorted Source Nodes: [out_333, out_334, out_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_333 => add_548, mul_655, mul_656, sub_218
#   out_334 => add_549
#   out_335 => relu_211
# Graph fragment:
#   %sub_218 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_218, %unsqueeze_1745), kwargs = {})
#   %mul_655 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_218, %unsqueeze_1747), kwargs = {})
#   %mul_656 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_655, %unsqueeze_1749), kwargs = {})
#   %add_548 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_656, %unsqueeze_1751), kwargs = {})
#   %add_549 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_548, %relu_206), kwargs = {})
#   %relu_211 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_549,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/xl/cxle36oke3bx2siubfeoqgwvf3ctpextjeqvvs2fy5dfiszm7fz4.py
# Topologically Sorted Source Nodes: [out_505, out_506], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_505 => add_824, mul_973, mul_974, sub_324
#   out_506 => relu_317
# Graph fragment:
#   %sub_324 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_324, %unsqueeze_2593), kwargs = {})
#   %mul_973 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_324, %unsqueeze_2595), kwargs = {})
#   %mul_974 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_973, %unsqueeze_2597), kwargs = {})
#   %add_824 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_974, %unsqueeze_2599), kwargs = {})
#   %relu_317 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_824,), kwargs = {})
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
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 832
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


# kernel path: /tmp/torchinductor_sahanp/re/cre5sfmqrca5vark55pujivfcfruldhs3e7ztxumk2yqw4oxqukj.py
# Topologically Sorted Source Nodes: [sp_757], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_757 => convolution_325
# Graph fragment:
#   %convolution_325 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_1268, %arg776_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 43264
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (208*x2) + (1872*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h7/ch7yb3dziwjy46che6bo6ikgqfwvv3rimit2mj4oowchiaoyehrj.py
# Topologically Sorted Source Nodes: [avg_pool2d_7], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_7 => avg_pool2d_7
# Graph fragment:
#   %avg_pool2d_7 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_1283, [3, 3], [2, 2], [1, 1]), kwargs = {})
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
    xnumel = 208
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
    tmp11 = tl.load(in_ptr0 + ((-11856) + x3 + (1664*y0) + (23296*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-11024) + x3 + (1664*y0) + (23296*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-10192) + x3 + (1664*y0) + (23296*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-208) + x3 + (1664*y0) + (23296*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (624 + x3 + (1664*y0) + (23296*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1456 + x3 + (1664*y0) + (23296*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (11440 + x3 + (1664*y0) + (23296*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (12272 + x3 + (1664*y0) + (23296*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (13104 + x3 + (1664*y0) + (23296*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15)))*((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))) + ((-2)*y0*((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))) + ((-2)*y1*((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15)))) + (4*y0*y1) + ((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15))) + ((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (49*x3) + (40768*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fw/cfwebtdajpojlkteqcvll6c7mfykjdfn5mp3mbwfcdvumaaczibb.py
# Topologically Sorted Source Nodes: [sp_758, sp_759], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_758 => add_826, mul_976, mul_977, sub_325
#   sp_759 => relu_318
# Graph fragment:
#   %sub_325 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_325, %unsqueeze_2601), kwargs = {})
#   %mul_976 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_325, %unsqueeze_2603), kwargs = {})
#   %mul_977 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_976, %unsqueeze_2605), kwargs = {})
#   %add_826 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_977, %unsqueeze_2607), kwargs = {})
#   %relu_318 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_826,), kwargs = {})
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
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (y0 + (208*x2) + (10192*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (49*y0) + (40768*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/6j/c6jehjgv4qtqi5diu7bpwa7ykhs5st22wzahh5iavbu6c7m2dwf6.py
# Topologically Sorted Source Nodes: [out_508], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_508 => convolution_328
# Graph fragment:
#   %convolution_328 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_63, %arg791_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    ynumel = 6656
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 832
    y1 = (yindex // 832)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (832*x2) + (40768*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3y/c3ybzhn7z6qmw2ms2lblx6pzp4gnetrh7hjxrr7passm3ianpeit.py
# Topologically Sorted Source Nodes: [out_509, input_16, out_510, out_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_834, mul_988, mul_989, sub_329
#   out_509 => add_832, mul_985, mul_986, sub_328
#   out_510 => add_835
#   out_511 => relu_321
# Graph fragment:
#   %sub_328 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_328, %unsqueeze_2625), kwargs = {})
#   %mul_985 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_328, %unsqueeze_2627), kwargs = {})
#   %mul_986 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_985, %unsqueeze_2629), kwargs = {})
#   %add_832 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_986, %unsqueeze_2631), kwargs = {})
#   %sub_329 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_329, %unsqueeze_2633), kwargs = {})
#   %mul_988 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_329, %unsqueeze_2635), kwargs = {})
#   %mul_989 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_988, %unsqueeze_2637), kwargs = {})
#   %add_834 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_989, %unsqueeze_2639), kwargs = {})
#   %add_835 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_832, %add_834), kwargs = {})
#   %relu_321 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_835,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/s2/cs2pnlcuvwudl4eupdln7jlankghh3xp7a7o5aofpdw4it3r7jh7.py
# Topologically Sorted Source Nodes: [out_513, out_514], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_513 => add_837, mul_991, mul_992, sub_330
#   out_514 => relu_322
# Graph fragment:
#   %sub_330 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_330, %unsqueeze_2641), kwargs = {})
#   %mul_991 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_330, %unsqueeze_2643), kwargs = {})
#   %mul_992 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_991, %unsqueeze_2645), kwargs = {})
#   %add_837 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_992, %unsqueeze_2647), kwargs = {})
#   %relu_322 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_837,), kwargs = {})
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
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 832
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


# kernel path: /tmp/torchinductor_sahanp/eg/cegv6rxvguftjdr36zaur3huaw4unat64ajcbilj3ffvd4gw7abn.py
# Topologically Sorted Source Nodes: [sp_772], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_772 => add_840
# Graph fragment:
#   %add_840 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_323, %getitem_1293), kwargs = {})
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
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (208 + x2 + (832*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vp/cvpfitsqxjos5sbuvtdzmogxfkrhkz7sozc37jvjhuadbsaejlte.py
# Topologically Sorted Source Nodes: [sp_776], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_776 => add_843
# Graph fragment:
#   %add_843 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_324, %getitem_1298), kwargs = {})
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
    xnumel = 208
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (416 + x2 + (832*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (208*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n7/cn77aa6cm44p77b5gezz3uxen6vsq35hjwfkdymcddrq4nwv5thr.py
# Topologically Sorted Source Nodes: [out_515], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_515 => cat_64
# Graph fragment:
#   %cat_64 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_323, %relu_324, %relu_325, %getitem_1303], 1), kwargs = {})
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
    ynumel = 1664
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 208
    y1 = (yindex // 208)
    tmp0 = tl.load(in_ptr0 + (624 + y0 + (832*x2) + (40768*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y0) + (40768*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g5/cg5g2lkzaaxjlnp4jgir6if7wg7zsakagfqg33hec2ceuicohlkg.py
# Topologically Sorted Source Nodes: [out_517, out_518, out_519], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_517 => add_847, mul_1003, mul_1004, sub_334
#   out_518 => add_848
#   out_519 => relu_326
# Graph fragment:
#   %sub_334 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_334, %unsqueeze_2673), kwargs = {})
#   %mul_1003 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_334, %unsqueeze_2675), kwargs = {})
#   %mul_1004 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1003, %unsqueeze_2677), kwargs = {})
#   %add_847 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1004, %unsqueeze_2679), kwargs = {})
#   %add_848 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_847, %relu_321), kwargs = {})
#   %relu_326 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_848,), kwargs = {})
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
# Topologically Sorted Source Nodes: [out_525, out_526, out_527, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   out_525 => add_860, mul_1018, mul_1019, sub_339
#   out_526 => add_861
#   out_527 => relu_331
#   x_11 => mean_1
# Graph fragment:
#   %sub_339 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_339, %unsqueeze_2713), kwargs = {})
#   %mul_1018 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_339, %unsqueeze_2715), kwargs = {})
#   %mul_1019 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1018, %unsqueeze_2717), kwargs = {})
#   %add_860 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1019, %unsqueeze_2719), kwargs = {})
#   %add_861 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_860, %relu_326), kwargs = {})
#   %relu_331 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_861,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_331, [-1, -2], True), kwargs = {})
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (104, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg7_1, (104, ), (1, ))
    assert_size_stride(arg8_1, (104, ), (1, ))
    assert_size_stride(arg9_1, (104, ), (1, ))
    assert_size_stride(arg10_1, (104, ), (1, ))
    assert_size_stride(arg11_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg12_1, (26, ), (1, ))
    assert_size_stride(arg13_1, (26, ), (1, ))
    assert_size_stride(arg14_1, (26, ), (1, ))
    assert_size_stride(arg15_1, (26, ), (1, ))
    assert_size_stride(arg16_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg17_1, (26, ), (1, ))
    assert_size_stride(arg18_1, (26, ), (1, ))
    assert_size_stride(arg19_1, (26, ), (1, ))
    assert_size_stride(arg20_1, (26, ), (1, ))
    assert_size_stride(arg21_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg22_1, (26, ), (1, ))
    assert_size_stride(arg23_1, (26, ), (1, ))
    assert_size_stride(arg24_1, (26, ), (1, ))
    assert_size_stride(arg25_1, (26, ), (1, ))
    assert_size_stride(arg26_1, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg37_1, (104, ), (1, ))
    assert_size_stride(arg38_1, (104, ), (1, ))
    assert_size_stride(arg39_1, (104, ), (1, ))
    assert_size_stride(arg40_1, (104, ), (1, ))
    assert_size_stride(arg41_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg42_1, (26, ), (1, ))
    assert_size_stride(arg43_1, (26, ), (1, ))
    assert_size_stride(arg44_1, (26, ), (1, ))
    assert_size_stride(arg45_1, (26, ), (1, ))
    assert_size_stride(arg46_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg47_1, (26, ), (1, ))
    assert_size_stride(arg48_1, (26, ), (1, ))
    assert_size_stride(arg49_1, (26, ), (1, ))
    assert_size_stride(arg50_1, (26, ), (1, ))
    assert_size_stride(arg51_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg52_1, (26, ), (1, ))
    assert_size_stride(arg53_1, (26, ), (1, ))
    assert_size_stride(arg54_1, (26, ), (1, ))
    assert_size_stride(arg55_1, (26, ), (1, ))
    assert_size_stride(arg56_1, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (256, ), (1, ))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (104, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg62_1, (104, ), (1, ))
    assert_size_stride(arg63_1, (104, ), (1, ))
    assert_size_stride(arg64_1, (104, ), (1, ))
    assert_size_stride(arg65_1, (104, ), (1, ))
    assert_size_stride(arg66_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg67_1, (26, ), (1, ))
    assert_size_stride(arg68_1, (26, ), (1, ))
    assert_size_stride(arg69_1, (26, ), (1, ))
    assert_size_stride(arg70_1, (26, ), (1, ))
    assert_size_stride(arg71_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg72_1, (26, ), (1, ))
    assert_size_stride(arg73_1, (26, ), (1, ))
    assert_size_stride(arg74_1, (26, ), (1, ))
    assert_size_stride(arg75_1, (26, ), (1, ))
    assert_size_stride(arg76_1, (26, 26, 3, 3), (234, 9, 3, 1))
    assert_size_stride(arg77_1, (26, ), (1, ))
    assert_size_stride(arg78_1, (26, ), (1, ))
    assert_size_stride(arg79_1, (26, ), (1, ))
    assert_size_stride(arg80_1, (26, ), (1, ))
    assert_size_stride(arg81_1, (256, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (256, ), (1, ))
    assert_size_stride(arg84_1, (256, ), (1, ))
    assert_size_stride(arg85_1, (256, ), (1, ))
    assert_size_stride(arg86_1, (208, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg87_1, (208, ), (1, ))
    assert_size_stride(arg88_1, (208, ), (1, ))
    assert_size_stride(arg89_1, (208, ), (1, ))
    assert_size_stride(arg90_1, (208, ), (1, ))
    assert_size_stride(arg91_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg92_1, (52, ), (1, ))
    assert_size_stride(arg93_1, (52, ), (1, ))
    assert_size_stride(arg94_1, (52, ), (1, ))
    assert_size_stride(arg95_1, (52, ), (1, ))
    assert_size_stride(arg96_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg97_1, (52, ), (1, ))
    assert_size_stride(arg98_1, (52, ), (1, ))
    assert_size_stride(arg99_1, (52, ), (1, ))
    assert_size_stride(arg100_1, (52, ), (1, ))
    assert_size_stride(arg101_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg102_1, (52, ), (1, ))
    assert_size_stride(arg103_1, (52, ), (1, ))
    assert_size_stride(arg104_1, (52, ), (1, ))
    assert_size_stride(arg105_1, (52, ), (1, ))
    assert_size_stride(arg106_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg117_1, (208, ), (1, ))
    assert_size_stride(arg118_1, (208, ), (1, ))
    assert_size_stride(arg119_1, (208, ), (1, ))
    assert_size_stride(arg120_1, (208, ), (1, ))
    assert_size_stride(arg121_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg122_1, (52, ), (1, ))
    assert_size_stride(arg123_1, (52, ), (1, ))
    assert_size_stride(arg124_1, (52, ), (1, ))
    assert_size_stride(arg125_1, (52, ), (1, ))
    assert_size_stride(arg126_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg127_1, (52, ), (1, ))
    assert_size_stride(arg128_1, (52, ), (1, ))
    assert_size_stride(arg129_1, (52, ), (1, ))
    assert_size_stride(arg130_1, (52, ), (1, ))
    assert_size_stride(arg131_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg132_1, (52, ), (1, ))
    assert_size_stride(arg133_1, (52, ), (1, ))
    assert_size_stride(arg134_1, (52, ), (1, ))
    assert_size_stride(arg135_1, (52, ), (1, ))
    assert_size_stride(arg136_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (512, ), (1, ))
    assert_size_stride(arg139_1, (512, ), (1, ))
    assert_size_stride(arg140_1, (512, ), (1, ))
    assert_size_stride(arg141_1, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg142_1, (208, ), (1, ))
    assert_size_stride(arg143_1, (208, ), (1, ))
    assert_size_stride(arg144_1, (208, ), (1, ))
    assert_size_stride(arg145_1, (208, ), (1, ))
    assert_size_stride(arg146_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg147_1, (52, ), (1, ))
    assert_size_stride(arg148_1, (52, ), (1, ))
    assert_size_stride(arg149_1, (52, ), (1, ))
    assert_size_stride(arg150_1, (52, ), (1, ))
    assert_size_stride(arg151_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg152_1, (52, ), (1, ))
    assert_size_stride(arg153_1, (52, ), (1, ))
    assert_size_stride(arg154_1, (52, ), (1, ))
    assert_size_stride(arg155_1, (52, ), (1, ))
    assert_size_stride(arg156_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg157_1, (52, ), (1, ))
    assert_size_stride(arg158_1, (52, ), (1, ))
    assert_size_stride(arg159_1, (52, ), (1, ))
    assert_size_stride(arg160_1, (52, ), (1, ))
    assert_size_stride(arg161_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (512, ), (1, ))
    assert_size_stride(arg164_1, (512, ), (1, ))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (208, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg167_1, (208, ), (1, ))
    assert_size_stride(arg168_1, (208, ), (1, ))
    assert_size_stride(arg169_1, (208, ), (1, ))
    assert_size_stride(arg170_1, (208, ), (1, ))
    assert_size_stride(arg171_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg172_1, (52, ), (1, ))
    assert_size_stride(arg173_1, (52, ), (1, ))
    assert_size_stride(arg174_1, (52, ), (1, ))
    assert_size_stride(arg175_1, (52, ), (1, ))
    assert_size_stride(arg176_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg177_1, (52, ), (1, ))
    assert_size_stride(arg178_1, (52, ), (1, ))
    assert_size_stride(arg179_1, (52, ), (1, ))
    assert_size_stride(arg180_1, (52, ), (1, ))
    assert_size_stride(arg181_1, (52, 52, 3, 3), (468, 9, 3, 1))
    assert_size_stride(arg182_1, (52, ), (1, ))
    assert_size_stride(arg183_1, (52, ), (1, ))
    assert_size_stride(arg184_1, (52, ), (1, ))
    assert_size_stride(arg185_1, (52, ), (1, ))
    assert_size_stride(arg186_1, (512, 208, 1, 1), (208, 1, 1, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (416, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg192_1, (416, ), (1, ))
    assert_size_stride(arg193_1, (416, ), (1, ))
    assert_size_stride(arg194_1, (416, ), (1, ))
    assert_size_stride(arg195_1, (416, ), (1, ))
    assert_size_stride(arg196_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg197_1, (104, ), (1, ))
    assert_size_stride(arg198_1, (104, ), (1, ))
    assert_size_stride(arg199_1, (104, ), (1, ))
    assert_size_stride(arg200_1, (104, ), (1, ))
    assert_size_stride(arg201_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg202_1, (104, ), (1, ))
    assert_size_stride(arg203_1, (104, ), (1, ))
    assert_size_stride(arg204_1, (104, ), (1, ))
    assert_size_stride(arg205_1, (104, ), (1, ))
    assert_size_stride(arg206_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg207_1, (104, ), (1, ))
    assert_size_stride(arg208_1, (104, ), (1, ))
    assert_size_stride(arg209_1, (104, ), (1, ))
    assert_size_stride(arg210_1, (104, ), (1, ))
    assert_size_stride(arg211_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (1024, ), (1, ))
    assert_size_stride(arg220_1, (1024, ), (1, ))
    assert_size_stride(arg221_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg222_1, (416, ), (1, ))
    assert_size_stride(arg223_1, (416, ), (1, ))
    assert_size_stride(arg224_1, (416, ), (1, ))
    assert_size_stride(arg225_1, (416, ), (1, ))
    assert_size_stride(arg226_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg227_1, (104, ), (1, ))
    assert_size_stride(arg228_1, (104, ), (1, ))
    assert_size_stride(arg229_1, (104, ), (1, ))
    assert_size_stride(arg230_1, (104, ), (1, ))
    assert_size_stride(arg231_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg232_1, (104, ), (1, ))
    assert_size_stride(arg233_1, (104, ), (1, ))
    assert_size_stride(arg234_1, (104, ), (1, ))
    assert_size_stride(arg235_1, (104, ), (1, ))
    assert_size_stride(arg236_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg237_1, (104, ), (1, ))
    assert_size_stride(arg238_1, (104, ), (1, ))
    assert_size_stride(arg239_1, (104, ), (1, ))
    assert_size_stride(arg240_1, (104, ), (1, ))
    assert_size_stride(arg241_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg247_1, (416, ), (1, ))
    assert_size_stride(arg248_1, (416, ), (1, ))
    assert_size_stride(arg249_1, (416, ), (1, ))
    assert_size_stride(arg250_1, (416, ), (1, ))
    assert_size_stride(arg251_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg252_1, (104, ), (1, ))
    assert_size_stride(arg253_1, (104, ), (1, ))
    assert_size_stride(arg254_1, (104, ), (1, ))
    assert_size_stride(arg255_1, (104, ), (1, ))
    assert_size_stride(arg256_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg257_1, (104, ), (1, ))
    assert_size_stride(arg258_1, (104, ), (1, ))
    assert_size_stride(arg259_1, (104, ), (1, ))
    assert_size_stride(arg260_1, (104, ), (1, ))
    assert_size_stride(arg261_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg262_1, (104, ), (1, ))
    assert_size_stride(arg263_1, (104, ), (1, ))
    assert_size_stride(arg264_1, (104, ), (1, ))
    assert_size_stride(arg265_1, (104, ), (1, ))
    assert_size_stride(arg266_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg267_1, (1024, ), (1, ))
    assert_size_stride(arg268_1, (1024, ), (1, ))
    assert_size_stride(arg269_1, (1024, ), (1, ))
    assert_size_stride(arg270_1, (1024, ), (1, ))
    assert_size_stride(arg271_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg272_1, (416, ), (1, ))
    assert_size_stride(arg273_1, (416, ), (1, ))
    assert_size_stride(arg274_1, (416, ), (1, ))
    assert_size_stride(arg275_1, (416, ), (1, ))
    assert_size_stride(arg276_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg277_1, (104, ), (1, ))
    assert_size_stride(arg278_1, (104, ), (1, ))
    assert_size_stride(arg279_1, (104, ), (1, ))
    assert_size_stride(arg280_1, (104, ), (1, ))
    assert_size_stride(arg281_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg282_1, (104, ), (1, ))
    assert_size_stride(arg283_1, (104, ), (1, ))
    assert_size_stride(arg284_1, (104, ), (1, ))
    assert_size_stride(arg285_1, (104, ), (1, ))
    assert_size_stride(arg286_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg287_1, (104, ), (1, ))
    assert_size_stride(arg288_1, (104, ), (1, ))
    assert_size_stride(arg289_1, (104, ), (1, ))
    assert_size_stride(arg290_1, (104, ), (1, ))
    assert_size_stride(arg291_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg292_1, (1024, ), (1, ))
    assert_size_stride(arg293_1, (1024, ), (1, ))
    assert_size_stride(arg294_1, (1024, ), (1, ))
    assert_size_stride(arg295_1, (1024, ), (1, ))
    assert_size_stride(arg296_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg297_1, (416, ), (1, ))
    assert_size_stride(arg298_1, (416, ), (1, ))
    assert_size_stride(arg299_1, (416, ), (1, ))
    assert_size_stride(arg300_1, (416, ), (1, ))
    assert_size_stride(arg301_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg302_1, (104, ), (1, ))
    assert_size_stride(arg303_1, (104, ), (1, ))
    assert_size_stride(arg304_1, (104, ), (1, ))
    assert_size_stride(arg305_1, (104, ), (1, ))
    assert_size_stride(arg306_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg307_1, (104, ), (1, ))
    assert_size_stride(arg308_1, (104, ), (1, ))
    assert_size_stride(arg309_1, (104, ), (1, ))
    assert_size_stride(arg310_1, (104, ), (1, ))
    assert_size_stride(arg311_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg312_1, (104, ), (1, ))
    assert_size_stride(arg313_1, (104, ), (1, ))
    assert_size_stride(arg314_1, (104, ), (1, ))
    assert_size_stride(arg315_1, (104, ), (1, ))
    assert_size_stride(arg316_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg322_1, (416, ), (1, ))
    assert_size_stride(arg323_1, (416, ), (1, ))
    assert_size_stride(arg324_1, (416, ), (1, ))
    assert_size_stride(arg325_1, (416, ), (1, ))
    assert_size_stride(arg326_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg327_1, (104, ), (1, ))
    assert_size_stride(arg328_1, (104, ), (1, ))
    assert_size_stride(arg329_1, (104, ), (1, ))
    assert_size_stride(arg330_1, (104, ), (1, ))
    assert_size_stride(arg331_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg332_1, (104, ), (1, ))
    assert_size_stride(arg333_1, (104, ), (1, ))
    assert_size_stride(arg334_1, (104, ), (1, ))
    assert_size_stride(arg335_1, (104, ), (1, ))
    assert_size_stride(arg336_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg337_1, (104, ), (1, ))
    assert_size_stride(arg338_1, (104, ), (1, ))
    assert_size_stride(arg339_1, (104, ), (1, ))
    assert_size_stride(arg340_1, (104, ), (1, ))
    assert_size_stride(arg341_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg342_1, (1024, ), (1, ))
    assert_size_stride(arg343_1, (1024, ), (1, ))
    assert_size_stride(arg344_1, (1024, ), (1, ))
    assert_size_stride(arg345_1, (1024, ), (1, ))
    assert_size_stride(arg346_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg347_1, (416, ), (1, ))
    assert_size_stride(arg348_1, (416, ), (1, ))
    assert_size_stride(arg349_1, (416, ), (1, ))
    assert_size_stride(arg350_1, (416, ), (1, ))
    assert_size_stride(arg351_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg352_1, (104, ), (1, ))
    assert_size_stride(arg353_1, (104, ), (1, ))
    assert_size_stride(arg354_1, (104, ), (1, ))
    assert_size_stride(arg355_1, (104, ), (1, ))
    assert_size_stride(arg356_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg357_1, (104, ), (1, ))
    assert_size_stride(arg358_1, (104, ), (1, ))
    assert_size_stride(arg359_1, (104, ), (1, ))
    assert_size_stride(arg360_1, (104, ), (1, ))
    assert_size_stride(arg361_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg362_1, (104, ), (1, ))
    assert_size_stride(arg363_1, (104, ), (1, ))
    assert_size_stride(arg364_1, (104, ), (1, ))
    assert_size_stride(arg365_1, (104, ), (1, ))
    assert_size_stride(arg366_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg367_1, (1024, ), (1, ))
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (1024, ), (1, ))
    assert_size_stride(arg371_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg372_1, (416, ), (1, ))
    assert_size_stride(arg373_1, (416, ), (1, ))
    assert_size_stride(arg374_1, (416, ), (1, ))
    assert_size_stride(arg375_1, (416, ), (1, ))
    assert_size_stride(arg376_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg377_1, (104, ), (1, ))
    assert_size_stride(arg378_1, (104, ), (1, ))
    assert_size_stride(arg379_1, (104, ), (1, ))
    assert_size_stride(arg380_1, (104, ), (1, ))
    assert_size_stride(arg381_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg382_1, (104, ), (1, ))
    assert_size_stride(arg383_1, (104, ), (1, ))
    assert_size_stride(arg384_1, (104, ), (1, ))
    assert_size_stride(arg385_1, (104, ), (1, ))
    assert_size_stride(arg386_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg387_1, (104, ), (1, ))
    assert_size_stride(arg388_1, (104, ), (1, ))
    assert_size_stride(arg389_1, (104, ), (1, ))
    assert_size_stride(arg390_1, (104, ), (1, ))
    assert_size_stride(arg391_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (1024, ), (1, ))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg397_1, (416, ), (1, ))
    assert_size_stride(arg398_1, (416, ), (1, ))
    assert_size_stride(arg399_1, (416, ), (1, ))
    assert_size_stride(arg400_1, (416, ), (1, ))
    assert_size_stride(arg401_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg402_1, (104, ), (1, ))
    assert_size_stride(arg403_1, (104, ), (1, ))
    assert_size_stride(arg404_1, (104, ), (1, ))
    assert_size_stride(arg405_1, (104, ), (1, ))
    assert_size_stride(arg406_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg407_1, (104, ), (1, ))
    assert_size_stride(arg408_1, (104, ), (1, ))
    assert_size_stride(arg409_1, (104, ), (1, ))
    assert_size_stride(arg410_1, (104, ), (1, ))
    assert_size_stride(arg411_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg412_1, (104, ), (1, ))
    assert_size_stride(arg413_1, (104, ), (1, ))
    assert_size_stride(arg414_1, (104, ), (1, ))
    assert_size_stride(arg415_1, (104, ), (1, ))
    assert_size_stride(arg416_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg417_1, (1024, ), (1, ))
    assert_size_stride(arg418_1, (1024, ), (1, ))
    assert_size_stride(arg419_1, (1024, ), (1, ))
    assert_size_stride(arg420_1, (1024, ), (1, ))
    assert_size_stride(arg421_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg422_1, (416, ), (1, ))
    assert_size_stride(arg423_1, (416, ), (1, ))
    assert_size_stride(arg424_1, (416, ), (1, ))
    assert_size_stride(arg425_1, (416, ), (1, ))
    assert_size_stride(arg426_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg427_1, (104, ), (1, ))
    assert_size_stride(arg428_1, (104, ), (1, ))
    assert_size_stride(arg429_1, (104, ), (1, ))
    assert_size_stride(arg430_1, (104, ), (1, ))
    assert_size_stride(arg431_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg432_1, (104, ), (1, ))
    assert_size_stride(arg433_1, (104, ), (1, ))
    assert_size_stride(arg434_1, (104, ), (1, ))
    assert_size_stride(arg435_1, (104, ), (1, ))
    assert_size_stride(arg436_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg437_1, (104, ), (1, ))
    assert_size_stride(arg438_1, (104, ), (1, ))
    assert_size_stride(arg439_1, (104, ), (1, ))
    assert_size_stride(arg440_1, (104, ), (1, ))
    assert_size_stride(arg441_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg442_1, (1024, ), (1, ))
    assert_size_stride(arg443_1, (1024, ), (1, ))
    assert_size_stride(arg444_1, (1024, ), (1, ))
    assert_size_stride(arg445_1, (1024, ), (1, ))
    assert_size_stride(arg446_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg447_1, (416, ), (1, ))
    assert_size_stride(arg448_1, (416, ), (1, ))
    assert_size_stride(arg449_1, (416, ), (1, ))
    assert_size_stride(arg450_1, (416, ), (1, ))
    assert_size_stride(arg451_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg452_1, (104, ), (1, ))
    assert_size_stride(arg453_1, (104, ), (1, ))
    assert_size_stride(arg454_1, (104, ), (1, ))
    assert_size_stride(arg455_1, (104, ), (1, ))
    assert_size_stride(arg456_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg457_1, (104, ), (1, ))
    assert_size_stride(arg458_1, (104, ), (1, ))
    assert_size_stride(arg459_1, (104, ), (1, ))
    assert_size_stride(arg460_1, (104, ), (1, ))
    assert_size_stride(arg461_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg462_1, (104, ), (1, ))
    assert_size_stride(arg463_1, (104, ), (1, ))
    assert_size_stride(arg464_1, (104, ), (1, ))
    assert_size_stride(arg465_1, (104, ), (1, ))
    assert_size_stride(arg466_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, ), (1, ))
    assert_size_stride(arg471_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg472_1, (416, ), (1, ))
    assert_size_stride(arg473_1, (416, ), (1, ))
    assert_size_stride(arg474_1, (416, ), (1, ))
    assert_size_stride(arg475_1, (416, ), (1, ))
    assert_size_stride(arg476_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg477_1, (104, ), (1, ))
    assert_size_stride(arg478_1, (104, ), (1, ))
    assert_size_stride(arg479_1, (104, ), (1, ))
    assert_size_stride(arg480_1, (104, ), (1, ))
    assert_size_stride(arg481_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg482_1, (104, ), (1, ))
    assert_size_stride(arg483_1, (104, ), (1, ))
    assert_size_stride(arg484_1, (104, ), (1, ))
    assert_size_stride(arg485_1, (104, ), (1, ))
    assert_size_stride(arg486_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg487_1, (104, ), (1, ))
    assert_size_stride(arg488_1, (104, ), (1, ))
    assert_size_stride(arg489_1, (104, ), (1, ))
    assert_size_stride(arg490_1, (104, ), (1, ))
    assert_size_stride(arg491_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg492_1, (1024, ), (1, ))
    assert_size_stride(arg493_1, (1024, ), (1, ))
    assert_size_stride(arg494_1, (1024, ), (1, ))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg497_1, (416, ), (1, ))
    assert_size_stride(arg498_1, (416, ), (1, ))
    assert_size_stride(arg499_1, (416, ), (1, ))
    assert_size_stride(arg500_1, (416, ), (1, ))
    assert_size_stride(arg501_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg502_1, (104, ), (1, ))
    assert_size_stride(arg503_1, (104, ), (1, ))
    assert_size_stride(arg504_1, (104, ), (1, ))
    assert_size_stride(arg505_1, (104, ), (1, ))
    assert_size_stride(arg506_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg507_1, (104, ), (1, ))
    assert_size_stride(arg508_1, (104, ), (1, ))
    assert_size_stride(arg509_1, (104, ), (1, ))
    assert_size_stride(arg510_1, (104, ), (1, ))
    assert_size_stride(arg511_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg512_1, (104, ), (1, ))
    assert_size_stride(arg513_1, (104, ), (1, ))
    assert_size_stride(arg514_1, (104, ), (1, ))
    assert_size_stride(arg515_1, (104, ), (1, ))
    assert_size_stride(arg516_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg517_1, (1024, ), (1, ))
    assert_size_stride(arg518_1, (1024, ), (1, ))
    assert_size_stride(arg519_1, (1024, ), (1, ))
    assert_size_stride(arg520_1, (1024, ), (1, ))
    assert_size_stride(arg521_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg522_1, (416, ), (1, ))
    assert_size_stride(arg523_1, (416, ), (1, ))
    assert_size_stride(arg524_1, (416, ), (1, ))
    assert_size_stride(arg525_1, (416, ), (1, ))
    assert_size_stride(arg526_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg527_1, (104, ), (1, ))
    assert_size_stride(arg528_1, (104, ), (1, ))
    assert_size_stride(arg529_1, (104, ), (1, ))
    assert_size_stride(arg530_1, (104, ), (1, ))
    assert_size_stride(arg531_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg532_1, (104, ), (1, ))
    assert_size_stride(arg533_1, (104, ), (1, ))
    assert_size_stride(arg534_1, (104, ), (1, ))
    assert_size_stride(arg535_1, (104, ), (1, ))
    assert_size_stride(arg536_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg537_1, (104, ), (1, ))
    assert_size_stride(arg538_1, (104, ), (1, ))
    assert_size_stride(arg539_1, (104, ), (1, ))
    assert_size_stride(arg540_1, (104, ), (1, ))
    assert_size_stride(arg541_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg542_1, (1024, ), (1, ))
    assert_size_stride(arg543_1, (1024, ), (1, ))
    assert_size_stride(arg544_1, (1024, ), (1, ))
    assert_size_stride(arg545_1, (1024, ), (1, ))
    assert_size_stride(arg546_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg547_1, (416, ), (1, ))
    assert_size_stride(arg548_1, (416, ), (1, ))
    assert_size_stride(arg549_1, (416, ), (1, ))
    assert_size_stride(arg550_1, (416, ), (1, ))
    assert_size_stride(arg551_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg552_1, (104, ), (1, ))
    assert_size_stride(arg553_1, (104, ), (1, ))
    assert_size_stride(arg554_1, (104, ), (1, ))
    assert_size_stride(arg555_1, (104, ), (1, ))
    assert_size_stride(arg556_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg557_1, (104, ), (1, ))
    assert_size_stride(arg558_1, (104, ), (1, ))
    assert_size_stride(arg559_1, (104, ), (1, ))
    assert_size_stride(arg560_1, (104, ), (1, ))
    assert_size_stride(arg561_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg562_1, (104, ), (1, ))
    assert_size_stride(arg563_1, (104, ), (1, ))
    assert_size_stride(arg564_1, (104, ), (1, ))
    assert_size_stride(arg565_1, (104, ), (1, ))
    assert_size_stride(arg566_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg567_1, (1024, ), (1, ))
    assert_size_stride(arg568_1, (1024, ), (1, ))
    assert_size_stride(arg569_1, (1024, ), (1, ))
    assert_size_stride(arg570_1, (1024, ), (1, ))
    assert_size_stride(arg571_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg572_1, (416, ), (1, ))
    assert_size_stride(arg573_1, (416, ), (1, ))
    assert_size_stride(arg574_1, (416, ), (1, ))
    assert_size_stride(arg575_1, (416, ), (1, ))
    assert_size_stride(arg576_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg577_1, (104, ), (1, ))
    assert_size_stride(arg578_1, (104, ), (1, ))
    assert_size_stride(arg579_1, (104, ), (1, ))
    assert_size_stride(arg580_1, (104, ), (1, ))
    assert_size_stride(arg581_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg582_1, (104, ), (1, ))
    assert_size_stride(arg583_1, (104, ), (1, ))
    assert_size_stride(arg584_1, (104, ), (1, ))
    assert_size_stride(arg585_1, (104, ), (1, ))
    assert_size_stride(arg586_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg587_1, (104, ), (1, ))
    assert_size_stride(arg588_1, (104, ), (1, ))
    assert_size_stride(arg589_1, (104, ), (1, ))
    assert_size_stride(arg590_1, (104, ), (1, ))
    assert_size_stride(arg591_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg592_1, (1024, ), (1, ))
    assert_size_stride(arg593_1, (1024, ), (1, ))
    assert_size_stride(arg594_1, (1024, ), (1, ))
    assert_size_stride(arg595_1, (1024, ), (1, ))
    assert_size_stride(arg596_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg597_1, (416, ), (1, ))
    assert_size_stride(arg598_1, (416, ), (1, ))
    assert_size_stride(arg599_1, (416, ), (1, ))
    assert_size_stride(arg600_1, (416, ), (1, ))
    assert_size_stride(arg601_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg602_1, (104, ), (1, ))
    assert_size_stride(arg603_1, (104, ), (1, ))
    assert_size_stride(arg604_1, (104, ), (1, ))
    assert_size_stride(arg605_1, (104, ), (1, ))
    assert_size_stride(arg606_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg607_1, (104, ), (1, ))
    assert_size_stride(arg608_1, (104, ), (1, ))
    assert_size_stride(arg609_1, (104, ), (1, ))
    assert_size_stride(arg610_1, (104, ), (1, ))
    assert_size_stride(arg611_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg612_1, (104, ), (1, ))
    assert_size_stride(arg613_1, (104, ), (1, ))
    assert_size_stride(arg614_1, (104, ), (1, ))
    assert_size_stride(arg615_1, (104, ), (1, ))
    assert_size_stride(arg616_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg617_1, (1024, ), (1, ))
    assert_size_stride(arg618_1, (1024, ), (1, ))
    assert_size_stride(arg619_1, (1024, ), (1, ))
    assert_size_stride(arg620_1, (1024, ), (1, ))
    assert_size_stride(arg621_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg622_1, (416, ), (1, ))
    assert_size_stride(arg623_1, (416, ), (1, ))
    assert_size_stride(arg624_1, (416, ), (1, ))
    assert_size_stride(arg625_1, (416, ), (1, ))
    assert_size_stride(arg626_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg627_1, (104, ), (1, ))
    assert_size_stride(arg628_1, (104, ), (1, ))
    assert_size_stride(arg629_1, (104, ), (1, ))
    assert_size_stride(arg630_1, (104, ), (1, ))
    assert_size_stride(arg631_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg632_1, (104, ), (1, ))
    assert_size_stride(arg633_1, (104, ), (1, ))
    assert_size_stride(arg634_1, (104, ), (1, ))
    assert_size_stride(arg635_1, (104, ), (1, ))
    assert_size_stride(arg636_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg637_1, (104, ), (1, ))
    assert_size_stride(arg638_1, (104, ), (1, ))
    assert_size_stride(arg639_1, (104, ), (1, ))
    assert_size_stride(arg640_1, (104, ), (1, ))
    assert_size_stride(arg641_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg642_1, (1024, ), (1, ))
    assert_size_stride(arg643_1, (1024, ), (1, ))
    assert_size_stride(arg644_1, (1024, ), (1, ))
    assert_size_stride(arg645_1, (1024, ), (1, ))
    assert_size_stride(arg646_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg647_1, (416, ), (1, ))
    assert_size_stride(arg648_1, (416, ), (1, ))
    assert_size_stride(arg649_1, (416, ), (1, ))
    assert_size_stride(arg650_1, (416, ), (1, ))
    assert_size_stride(arg651_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg652_1, (104, ), (1, ))
    assert_size_stride(arg653_1, (104, ), (1, ))
    assert_size_stride(arg654_1, (104, ), (1, ))
    assert_size_stride(arg655_1, (104, ), (1, ))
    assert_size_stride(arg656_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg657_1, (104, ), (1, ))
    assert_size_stride(arg658_1, (104, ), (1, ))
    assert_size_stride(arg659_1, (104, ), (1, ))
    assert_size_stride(arg660_1, (104, ), (1, ))
    assert_size_stride(arg661_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg662_1, (104, ), (1, ))
    assert_size_stride(arg663_1, (104, ), (1, ))
    assert_size_stride(arg664_1, (104, ), (1, ))
    assert_size_stride(arg665_1, (104, ), (1, ))
    assert_size_stride(arg666_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg667_1, (1024, ), (1, ))
    assert_size_stride(arg668_1, (1024, ), (1, ))
    assert_size_stride(arg669_1, (1024, ), (1, ))
    assert_size_stride(arg670_1, (1024, ), (1, ))
    assert_size_stride(arg671_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg672_1, (416, ), (1, ))
    assert_size_stride(arg673_1, (416, ), (1, ))
    assert_size_stride(arg674_1, (416, ), (1, ))
    assert_size_stride(arg675_1, (416, ), (1, ))
    assert_size_stride(arg676_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg677_1, (104, ), (1, ))
    assert_size_stride(arg678_1, (104, ), (1, ))
    assert_size_stride(arg679_1, (104, ), (1, ))
    assert_size_stride(arg680_1, (104, ), (1, ))
    assert_size_stride(arg681_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg682_1, (104, ), (1, ))
    assert_size_stride(arg683_1, (104, ), (1, ))
    assert_size_stride(arg684_1, (104, ), (1, ))
    assert_size_stride(arg685_1, (104, ), (1, ))
    assert_size_stride(arg686_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg687_1, (104, ), (1, ))
    assert_size_stride(arg688_1, (104, ), (1, ))
    assert_size_stride(arg689_1, (104, ), (1, ))
    assert_size_stride(arg690_1, (104, ), (1, ))
    assert_size_stride(arg691_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg692_1, (1024, ), (1, ))
    assert_size_stride(arg693_1, (1024, ), (1, ))
    assert_size_stride(arg694_1, (1024, ), (1, ))
    assert_size_stride(arg695_1, (1024, ), (1, ))
    assert_size_stride(arg696_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg697_1, (416, ), (1, ))
    assert_size_stride(arg698_1, (416, ), (1, ))
    assert_size_stride(arg699_1, (416, ), (1, ))
    assert_size_stride(arg700_1, (416, ), (1, ))
    assert_size_stride(arg701_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg702_1, (104, ), (1, ))
    assert_size_stride(arg703_1, (104, ), (1, ))
    assert_size_stride(arg704_1, (104, ), (1, ))
    assert_size_stride(arg705_1, (104, ), (1, ))
    assert_size_stride(arg706_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg707_1, (104, ), (1, ))
    assert_size_stride(arg708_1, (104, ), (1, ))
    assert_size_stride(arg709_1, (104, ), (1, ))
    assert_size_stride(arg710_1, (104, ), (1, ))
    assert_size_stride(arg711_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg712_1, (104, ), (1, ))
    assert_size_stride(arg713_1, (104, ), (1, ))
    assert_size_stride(arg714_1, (104, ), (1, ))
    assert_size_stride(arg715_1, (104, ), (1, ))
    assert_size_stride(arg716_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg717_1, (1024, ), (1, ))
    assert_size_stride(arg718_1, (1024, ), (1, ))
    assert_size_stride(arg719_1, (1024, ), (1, ))
    assert_size_stride(arg720_1, (1024, ), (1, ))
    assert_size_stride(arg721_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg722_1, (416, ), (1, ))
    assert_size_stride(arg723_1, (416, ), (1, ))
    assert_size_stride(arg724_1, (416, ), (1, ))
    assert_size_stride(arg725_1, (416, ), (1, ))
    assert_size_stride(arg726_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg727_1, (104, ), (1, ))
    assert_size_stride(arg728_1, (104, ), (1, ))
    assert_size_stride(arg729_1, (104, ), (1, ))
    assert_size_stride(arg730_1, (104, ), (1, ))
    assert_size_stride(arg731_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg732_1, (104, ), (1, ))
    assert_size_stride(arg733_1, (104, ), (1, ))
    assert_size_stride(arg734_1, (104, ), (1, ))
    assert_size_stride(arg735_1, (104, ), (1, ))
    assert_size_stride(arg736_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg737_1, (104, ), (1, ))
    assert_size_stride(arg738_1, (104, ), (1, ))
    assert_size_stride(arg739_1, (104, ), (1, ))
    assert_size_stride(arg740_1, (104, ), (1, ))
    assert_size_stride(arg741_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg742_1, (1024, ), (1, ))
    assert_size_stride(arg743_1, (1024, ), (1, ))
    assert_size_stride(arg744_1, (1024, ), (1, ))
    assert_size_stride(arg745_1, (1024, ), (1, ))
    assert_size_stride(arg746_1, (416, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg747_1, (416, ), (1, ))
    assert_size_stride(arg748_1, (416, ), (1, ))
    assert_size_stride(arg749_1, (416, ), (1, ))
    assert_size_stride(arg750_1, (416, ), (1, ))
    assert_size_stride(arg751_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg752_1, (104, ), (1, ))
    assert_size_stride(arg753_1, (104, ), (1, ))
    assert_size_stride(arg754_1, (104, ), (1, ))
    assert_size_stride(arg755_1, (104, ), (1, ))
    assert_size_stride(arg756_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg757_1, (104, ), (1, ))
    assert_size_stride(arg758_1, (104, ), (1, ))
    assert_size_stride(arg759_1, (104, ), (1, ))
    assert_size_stride(arg760_1, (104, ), (1, ))
    assert_size_stride(arg761_1, (104, 104, 3, 3), (936, 9, 3, 1))
    assert_size_stride(arg762_1, (104, ), (1, ))
    assert_size_stride(arg763_1, (104, ), (1, ))
    assert_size_stride(arg764_1, (104, ), (1, ))
    assert_size_stride(arg765_1, (104, ), (1, ))
    assert_size_stride(arg766_1, (1024, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg767_1, (1024, ), (1, ))
    assert_size_stride(arg768_1, (1024, ), (1, ))
    assert_size_stride(arg769_1, (1024, ), (1, ))
    assert_size_stride(arg770_1, (1024, ), (1, ))
    assert_size_stride(arg771_1, (832, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg772_1, (832, ), (1, ))
    assert_size_stride(arg773_1, (832, ), (1, ))
    assert_size_stride(arg774_1, (832, ), (1, ))
    assert_size_stride(arg775_1, (832, ), (1, ))
    assert_size_stride(arg776_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg777_1, (208, ), (1, ))
    assert_size_stride(arg778_1, (208, ), (1, ))
    assert_size_stride(arg779_1, (208, ), (1, ))
    assert_size_stride(arg780_1, (208, ), (1, ))
    assert_size_stride(arg781_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg782_1, (208, ), (1, ))
    assert_size_stride(arg783_1, (208, ), (1, ))
    assert_size_stride(arg784_1, (208, ), (1, ))
    assert_size_stride(arg785_1, (208, ), (1, ))
    assert_size_stride(arg786_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg787_1, (208, ), (1, ))
    assert_size_stride(arg788_1, (208, ), (1, ))
    assert_size_stride(arg789_1, (208, ), (1, ))
    assert_size_stride(arg790_1, (208, ), (1, ))
    assert_size_stride(arg791_1, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg792_1, (2048, ), (1, ))
    assert_size_stride(arg793_1, (2048, ), (1, ))
    assert_size_stride(arg794_1, (2048, ), (1, ))
    assert_size_stride(arg795_1, (2048, ), (1, ))
    assert_size_stride(arg796_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg797_1, (2048, ), (1, ))
    assert_size_stride(arg798_1, (2048, ), (1, ))
    assert_size_stride(arg799_1, (2048, ), (1, ))
    assert_size_stride(arg800_1, (2048, ), (1, ))
    assert_size_stride(arg801_1, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg802_1, (832, ), (1, ))
    assert_size_stride(arg803_1, (832, ), (1, ))
    assert_size_stride(arg804_1, (832, ), (1, ))
    assert_size_stride(arg805_1, (832, ), (1, ))
    assert_size_stride(arg806_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg807_1, (208, ), (1, ))
    assert_size_stride(arg808_1, (208, ), (1, ))
    assert_size_stride(arg809_1, (208, ), (1, ))
    assert_size_stride(arg810_1, (208, ), (1, ))
    assert_size_stride(arg811_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg812_1, (208, ), (1, ))
    assert_size_stride(arg813_1, (208, ), (1, ))
    assert_size_stride(arg814_1, (208, ), (1, ))
    assert_size_stride(arg815_1, (208, ), (1, ))
    assert_size_stride(arg816_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg817_1, (208, ), (1, ))
    assert_size_stride(arg818_1, (208, ), (1, ))
    assert_size_stride(arg819_1, (208, ), (1, ))
    assert_size_stride(arg820_1, (208, ), (1, ))
    assert_size_stride(arg821_1, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg822_1, (2048, ), (1, ))
    assert_size_stride(arg823_1, (2048, ), (1, ))
    assert_size_stride(arg824_1, (2048, ), (1, ))
    assert_size_stride(arg825_1, (2048, ), (1, ))
    assert_size_stride(arg826_1, (832, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg827_1, (832, ), (1, ))
    assert_size_stride(arg828_1, (832, ), (1, ))
    assert_size_stride(arg829_1, (832, ), (1, ))
    assert_size_stride(arg830_1, (832, ), (1, ))
    assert_size_stride(arg831_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg832_1, (208, ), (1, ))
    assert_size_stride(arg833_1, (208, ), (1, ))
    assert_size_stride(arg834_1, (208, ), (1, ))
    assert_size_stride(arg835_1, (208, ), (1, ))
    assert_size_stride(arg836_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg837_1, (208, ), (1, ))
    assert_size_stride(arg838_1, (208, ), (1, ))
    assert_size_stride(arg839_1, (208, ), (1, ))
    assert_size_stride(arg840_1, (208, ), (1, ))
    assert_size_stride(arg841_1, (208, 208, 3, 3), (1872, 9, 3, 1))
    assert_size_stride(arg842_1, (208, ), (1, ))
    assert_size_stride(arg843_1, (208, ), (1, ))
    assert_size_stride(arg844_1, (208, ), (1, ))
    assert_size_stride(arg845_1, (208, ), (1, ))
    assert_size_stride(arg846_1, (2048, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg847_1, (2048, ), (1, ))
    assert_size_stride(arg848_1, (2048, ), (1, ))
    assert_size_stride(arg849_1, (2048, ), (1, ))
    assert_size_stride(arg850_1, (2048, ), (1, ))
    assert_size_stride(arg851_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg852_1, (1000, ), (1, ))
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
        # Topologically Sorted Source Nodes: [out_264], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 104, 56, 56), (326144, 1, 5824, 104))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [out_265, out_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((26, 26, 3, 3), (234, 1, 78, 26), torch.float32)
        # Topologically Sorted Source Nodes: [sp_397], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg11_1, buf7, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [sp_397], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 26, 56, 56), (81536, 1, 1456, 26))
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sp_401], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg16_1, buf9, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [sp_401], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 26, 56, 56), (326144, 1, 5824, 104), 26), buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 26, 56, 56), (81536, 1, 1456, 26))
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sp_405], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg21_1, buf11, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [sp_405], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 26, 56, 56), (326144, 1, 5824, 104), 52), buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 26, 56, 56), (81536, 1, 1456, 26))
        buf17 = empty_strided_cuda((8, 104, 56, 56), (326144, 3136, 56, 1), torch.float32)
        buf13 = reinterpret_tensor(buf17, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_6.run(buf6, buf13, 25088, 26, grid=grid(25088, 26), stream=stream0)
        buf14 = reinterpret_tensor(buf17, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_398, sp_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf8, arg12_1, arg13_1, arg14_1, arg15_1, buf14, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf8
        buf15 = reinterpret_tensor(buf17, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        # Topologically Sorted Source Nodes: [sp_402, sp_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf10, arg17_1, arg18_1, arg19_1, arg20_1, buf15, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf10
        buf16 = reinterpret_tensor(buf17, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        # Topologically Sorted Source Nodes: [sp_406, sp_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf12, arg22_1, arg23_1, arg24_1, arg25_1, buf16, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf12
        buf18 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [out_268], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf17, buf18, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf13
        del buf14
        del buf15
        del buf16
        del buf17
        # Topologically Sorted Source Nodes: [out_268], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg26_1
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf4, arg31_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg31_1
        buf21 = buf19; del buf19  # reuse
        buf22 = reinterpret_tensor(buf3, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_269, input_10, out_270, out_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
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
        # Topologically Sorted Source Nodes: [out_271, out_272], Original ATen: [aten.relu, aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 104, 56, 56), (326144, 1, 5824, 104))
        del arg36_1
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [out_273, out_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf24, arg37_1, arg38_1, arg39_1, arg40_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        buf25 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [sp_409], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg41_1, buf25, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [sp_409], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(reinterpret_tensor(buf24, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 26, 56, 56), (81536, 1, 1456, 26))
        buf37 = reinterpret_tensor(buf18, (8, 104, 56, 56), (326144, 3136, 56, 1), 0); del buf18  # reuse
        buf27 = reinterpret_tensor(buf37, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_410, sp_411], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf26, arg42_1, arg43_1, arg44_1, arg45_1, buf27, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [sp_412], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf27, buf24, buf28, 25088, 26, grid=grid(25088, 26), stream=stream0)
        buf29 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [sp_412, sp_413], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg46_1, buf29, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg46_1
        # Topologically Sorted Source Nodes: [sp_412, sp_413], Original ATen: [aten.add, aten.convolution]
        buf30 = extern_kernels.convolution(buf28, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 26, 56, 56), (81536, 1, 1456, 26))
        del buf28
        buf31 = reinterpret_tensor(buf37, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        # Topologically Sorted Source Nodes: [sp_414, sp_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf30, arg47_1, arg48_1, arg49_1, arg50_1, buf31, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [sp_416], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf31, buf24, buf32, 25088, 26, grid=grid(25088, 26), stream=stream0)
        buf33 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [sp_416, sp_417], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg51_1, buf33, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg51_1
        # Topologically Sorted Source Nodes: [sp_416, sp_417], Original ATen: [aten.add, aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 26, 56, 56), (81536, 1, 1456, 26))
        del buf32
        buf35 = reinterpret_tensor(buf37, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        # Topologically Sorted Source Nodes: [sp_418, sp_419], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf34, arg52_1, arg53_1, arg54_1, arg55_1, buf35, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf34
        buf36 = reinterpret_tensor(buf37, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Topologically Sorted Source Nodes: [out_275], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf24, buf36, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf38 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_276], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf37, buf38, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf27
        del buf31
        del buf35
        del buf36
        del buf37
        # Topologically Sorted Source Nodes: [out_276], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg56_1
        buf40 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [out_277, out_278, out_279], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf40, buf39, arg57_1, arg58_1, arg59_1, arg60_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        del buf39
        # Topologically Sorted Source Nodes: [out_280], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 104, 56, 56), (326144, 1, 5824, 104))
        del arg61_1
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [out_281, out_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf42, arg62_1, arg63_1, arg64_1, arg65_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        buf43 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [sp_421], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg66_1, buf43, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [sp_421], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(reinterpret_tensor(buf42, (8, 26, 56, 56), (326144, 1, 5824, 104), 0), buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 26, 56, 56), (81536, 1, 1456, 26))
        buf55 = reinterpret_tensor(buf38, (8, 104, 56, 56), (326144, 3136, 56, 1), 0); del buf38  # reuse
        buf45 = reinterpret_tensor(buf55, (8, 26, 56, 56), (326144, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_422, sp_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf44, arg67_1, arg68_1, arg69_1, arg70_1, buf45, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf46 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [sp_424], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf45, buf42, buf46, 25088, 26, grid=grid(25088, 26), stream=stream0)
        buf47 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [sp_424, sp_425], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg71_1, buf47, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [sp_424, sp_425], Original ATen: [aten.add, aten.convolution]
        buf48 = extern_kernels.convolution(buf46, buf47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 26, 56, 56), (81536, 1, 1456, 26))
        del buf46
        buf49 = reinterpret_tensor(buf55, (8, 26, 56, 56), (326144, 3136, 56, 1), 81536)  # alias
        # Topologically Sorted Source Nodes: [sp_426, sp_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf48, arg72_1, arg73_1, arg74_1, arg75_1, buf49, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf50 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [sp_428], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf49, buf42, buf50, 25088, 26, grid=grid(25088, 26), stream=stream0)
        buf51 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [sp_428, sp_429], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg76_1, buf51, 676, 9, grid=grid(676, 9), stream=stream0)
        del arg76_1
        # Topologically Sorted Source Nodes: [sp_428, sp_429], Original ATen: [aten.add, aten.convolution]
        buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 26, 56, 56), (81536, 1, 1456, 26))
        del buf51
        buf53 = reinterpret_tensor(buf55, (8, 26, 56, 56), (326144, 3136, 56, 1), 163072)  # alias
        # Topologically Sorted Source Nodes: [sp_430, sp_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf52, arg77_1, arg78_1, arg79_1, arg80_1, buf53, 208, 3136, grid=grid(208, 3136), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf54 = reinterpret_tensor(buf55, (8, 26, 56, 56), (326144, 3136, 56, 1), 244608)  # alias
        # Topologically Sorted Source Nodes: [out_283], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf42, buf54, 208, 3136, grid=grid(208, 3136), stream=stream0)
        buf56 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [out_284], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf55, buf56, 832, 3136, grid=grid(832, 3136), stream=stream0)
        del buf45
        del buf49
        del buf53
        del buf54
        del buf55
        # Topologically Sorted Source Nodes: [out_284], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg81_1
        del buf56
        buf58 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [out_285, out_286, out_287], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf58, buf57, arg82_1, arg83_1, arg84_1, arg85_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        del buf57
        # Topologically Sorted Source Nodes: [out_288], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 208, 56, 56), (652288, 1, 11648, 208))
        del arg86_1
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [out_289, out_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf60, arg87_1, arg88_1, arg89_1, arg90_1, 5218304, grid=grid(5218304), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf61 = empty_strided_cuda((52, 52, 3, 3), (468, 1, 156, 52), torch.float32)
        # Topologically Sorted Source Nodes: [sp_433], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg91_1, buf61, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [sp_433], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 52, 56, 56), (652288, 1, 11648, 208), 0), buf61, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 52, 28, 28), (40768, 1, 1456, 52))
        buf63 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [sp_437], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg96_1, buf63, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [sp_437], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 52, 56, 56), (652288, 1, 11648, 208), 52), buf63, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 52, 28, 28), (40768, 1, 1456, 52))
        buf65 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [sp_441], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg101_1, buf65, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg101_1
        # Topologically Sorted Source Nodes: [sp_441], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(reinterpret_tensor(buf60, (8, 52, 56, 56), (652288, 1, 11648, 208), 104), buf65, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 52, 28, 28), (40768, 1, 1456, 52))
        buf71 = empty_strided_cuda((8, 208, 28, 28), (163072, 784, 28, 1), torch.float32)
        buf67 = reinterpret_tensor(buf71, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_16.run(buf60, buf67, 6272, 52, grid=grid(6272, 52), stream=stream0)
        del buf60
        buf68 = reinterpret_tensor(buf71, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_434, sp_435], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf62, arg92_1, arg93_1, arg94_1, arg95_1, buf68, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf62
        buf69 = reinterpret_tensor(buf71, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_438, sp_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf64, arg97_1, arg98_1, arg99_1, arg100_1, buf69, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf64
        buf70 = reinterpret_tensor(buf71, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Topologically Sorted Source Nodes: [sp_442, sp_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf66, arg102_1, arg103_1, arg104_1, arg105_1, buf70, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del buf66
        buf72 = empty_strided_cuda((8, 208, 28, 28), (163072, 1, 5824, 208), torch.float32)
        # Topologically Sorted Source Nodes: [out_292], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf71, buf72, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf67
        del buf68
        del buf69
        del buf70
        del buf71
        # Topologically Sorted Source Nodes: [out_292], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg106_1
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf58, arg111_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg111_1
        del buf58
        buf75 = buf73; del buf73  # reuse
        buf76 = empty_strided_cuda((8, 512, 28, 28), (401408, 1, 14336, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_293, input_12, out_294, out_295], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
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
        # Topologically Sorted Source Nodes: [out_295, out_296], Original ATen: [aten.relu, aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 208, 28, 28), (163072, 1, 5824, 208))
        del arg116_1
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [out_297, out_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf78, arg117_1, arg118_1, arg119_1, arg120_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf79 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [sp_445], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg121_1, buf79, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [sp_445], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(reinterpret_tensor(buf78, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 52, 28, 28), (40768, 1, 1456, 52))
        buf91 = reinterpret_tensor(buf72, (8, 208, 28, 28), (163072, 784, 28, 1), 0); del buf72  # reuse
        buf81 = reinterpret_tensor(buf91, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_446, sp_447], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf80, arg122_1, arg123_1, arg124_1, arg125_1, buf81, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf82 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [sp_448], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf81, buf78, buf82, 6272, 52, grid=grid(6272, 52), stream=stream0)
        buf83 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [sp_448, sp_449], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg126_1, buf83, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [sp_448, sp_449], Original ATen: [aten.add, aten.convolution]
        buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 52, 28, 28), (40768, 1, 1456, 52))
        del buf82
        buf85 = reinterpret_tensor(buf91, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_450, sp_451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf84, arg127_1, arg128_1, arg129_1, arg130_1, buf85, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf86 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [sp_452], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf85, buf78, buf86, 6272, 52, grid=grid(6272, 52), stream=stream0)
        buf87 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [sp_452, sp_453], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg131_1, buf87, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [sp_452, sp_453], Original ATen: [aten.add, aten.convolution]
        buf88 = extern_kernels.convolution(buf86, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 52, 28, 28), (40768, 1, 1456, 52))
        del buf86
        buf89 = reinterpret_tensor(buf91, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Topologically Sorted Source Nodes: [sp_454, sp_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf88, arg132_1, arg133_1, arg134_1, arg135_1, buf89, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del buf88
        buf90 = reinterpret_tensor(buf91, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Topologically Sorted Source Nodes: [out_299], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf78, buf90, 416, 784, grid=grid(416, 784), stream=stream0)
        buf92 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [out_300], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf91, buf92, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf81
        del buf85
        del buf89
        del buf90
        del buf91
        # Topologically Sorted Source Nodes: [out_300], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg136_1
        buf94 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [out_301, out_302, out_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf94, buf93, arg137_1, arg138_1, arg139_1, arg140_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf93
        # Topologically Sorted Source Nodes: [out_304], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 208, 28, 28), (163072, 1, 5824, 208))
        del arg141_1
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [out_305, out_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf96, arg142_1, arg143_1, arg144_1, arg145_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        buf97 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [sp_457], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg146_1, buf97, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [sp_457], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(reinterpret_tensor(buf96, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 52, 28, 28), (40768, 1, 1456, 52))
        buf109 = reinterpret_tensor(buf92, (8, 208, 28, 28), (163072, 784, 28, 1), 0); del buf92  # reuse
        buf99 = reinterpret_tensor(buf109, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_458, sp_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf98, arg147_1, arg148_1, arg149_1, arg150_1, buf99, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf100 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [sp_460], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf99, buf96, buf100, 6272, 52, grid=grid(6272, 52), stream=stream0)
        buf101 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [sp_460, sp_461], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg151_1, buf101, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg151_1
        # Topologically Sorted Source Nodes: [sp_460, sp_461], Original ATen: [aten.add, aten.convolution]
        buf102 = extern_kernels.convolution(buf100, buf101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 52, 28, 28), (40768, 1, 1456, 52))
        del buf100
        buf103 = reinterpret_tensor(buf109, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_462, sp_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf102, arg152_1, arg153_1, arg154_1, arg155_1, buf103, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        buf104 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [sp_464], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf103, buf96, buf104, 6272, 52, grid=grid(6272, 52), stream=stream0)
        buf105 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [sp_464, sp_465], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg156_1, buf105, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg156_1
        # Topologically Sorted Source Nodes: [sp_464, sp_465], Original ATen: [aten.add, aten.convolution]
        buf106 = extern_kernels.convolution(buf104, buf105, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 52, 28, 28), (40768, 1, 1456, 52))
        del buf104
        buf107 = reinterpret_tensor(buf109, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Topologically Sorted Source Nodes: [sp_466, sp_467], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf106, arg157_1, arg158_1, arg159_1, arg160_1, buf107, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        del buf106
        buf108 = reinterpret_tensor(buf109, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Topologically Sorted Source Nodes: [out_307], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf96, buf108, 416, 784, grid=grid(416, 784), stream=stream0)
        buf110 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [out_308], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf109, buf110, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf103
        del buf107
        del buf108
        del buf109
        del buf99
        # Topologically Sorted Source Nodes: [out_308], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg161_1
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [out_309, out_310, out_311], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_25.run(buf112, arg162_1, arg163_1, arg164_1, arg165_1, buf94, 3211264, grid=grid(3211264), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf94
        # Topologically Sorted Source Nodes: [out_312], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 208, 28, 28), (163072, 1, 5824, 208))
        del arg166_1
        buf114 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [out_313, out_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf114, arg167_1, arg168_1, arg169_1, arg170_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        buf115 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [sp_469], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg171_1, buf115, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg171_1
        # Topologically Sorted Source Nodes: [sp_469], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(reinterpret_tensor(buf114, (8, 52, 28, 28), (163072, 1, 5824, 208), 0), buf115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 52, 28, 28), (40768, 1, 1456, 52))
        buf127 = reinterpret_tensor(buf110, (8, 208, 28, 28), (163072, 784, 28, 1), 0); del buf110  # reuse
        buf117 = reinterpret_tensor(buf127, (8, 52, 28, 28), (163072, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_470, sp_471], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf116, arg172_1, arg173_1, arg174_1, arg175_1, buf117, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf118 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [sp_472], Original ATen: [aten.add]
        triton_poi_fused_add_21.run(buf117, buf114, buf118, 6272, 52, grid=grid(6272, 52), stream=stream0)
        buf119 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [sp_472, sp_473], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg176_1, buf119, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [sp_472, sp_473], Original ATen: [aten.add, aten.convolution]
        buf120 = extern_kernels.convolution(buf118, buf119, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (8, 52, 28, 28), (40768, 1, 1456, 52))
        del buf118
        buf121 = reinterpret_tensor(buf127, (8, 52, 28, 28), (163072, 784, 28, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_474, sp_475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf120, arg177_1, arg178_1, arg179_1, arg180_1, buf121, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        buf122 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [sp_476], Original ATen: [aten.add]
        triton_poi_fused_add_22.run(buf121, buf114, buf122, 6272, 52, grid=grid(6272, 52), stream=stream0)
        buf123 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [sp_476, sp_477], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_15.run(arg181_1, buf123, 2704, 9, grid=grid(2704, 9), stream=stream0)
        del arg181_1
        # Topologically Sorted Source Nodes: [sp_476, sp_477], Original ATen: [aten.add, aten.convolution]
        buf124 = extern_kernels.convolution(buf122, buf123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 52, 28, 28), (40768, 1, 1456, 52))
        del buf123
        buf125 = reinterpret_tensor(buf127, (8, 52, 28, 28), (163072, 784, 28, 1), 81536)  # alias
        # Topologically Sorted Source Nodes: [sp_478, sp_479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf124, arg182_1, arg183_1, arg184_1, arg185_1, buf125, 416, 784, grid=grid(416, 784), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        buf126 = reinterpret_tensor(buf127, (8, 52, 28, 28), (163072, 784, 28, 1), 122304)  # alias
        # Topologically Sorted Source Nodes: [out_315], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf114, buf126, 416, 784, grid=grid(416, 784), stream=stream0)
        buf128 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [out_316], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(buf127, buf128, 1664, 784, grid=grid(1664, 784), stream=stream0)
        del buf117
        del buf121
        del buf125
        del buf126
        del buf127
        # Topologically Sorted Source Nodes: [out_316], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg186_1
        del buf128
        buf130 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [out_317, out_318, out_319], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf130, buf129, arg187_1, arg188_1, arg189_1, arg190_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del buf129
        # Topologically Sorted Source Nodes: [out_320], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 416, 28, 28), (326144, 1, 11648, 416))
        del arg191_1
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [out_321, out_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf132, arg192_1, arg193_1, arg194_1, arg195_1, 2609152, grid=grid(2609152), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        buf133 = empty_strided_cuda((104, 104, 3, 3), (936, 1, 312, 104), torch.float32)
        # Topologically Sorted Source Nodes: [sp_481], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg196_1, buf133, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg196_1
        # Topologically Sorted Source Nodes: [sp_481], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 104, 28, 28), (326144, 1, 11648, 416), 0), buf133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf135 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [sp_485], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg201_1, buf135, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg201_1
        # Topologically Sorted Source Nodes: [sp_485], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 104, 28, 28), (326144, 1, 11648, 416), 104), buf135, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf137 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [sp_489], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg206_1, buf137, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg206_1
        # Topologically Sorted Source Nodes: [sp_489], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(reinterpret_tensor(buf132, (8, 104, 28, 28), (326144, 1, 11648, 416), 208), buf137, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf143 = reinterpret_tensor(buf52, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf52  # reuse
        buf139 = reinterpret_tensor(buf143, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_6], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_28.run(buf132, buf139, 1568, 104, grid=grid(1568, 104), stream=stream0)
        del buf132
        buf140 = reinterpret_tensor(buf143, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_482, sp_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf134, arg197_1, arg198_1, arg199_1, arg200_1, buf140, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf134
        buf141 = reinterpret_tensor(buf143, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_486, sp_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf136, arg202_1, arg203_1, arg204_1, arg205_1, buf141, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        del buf136
        buf142 = reinterpret_tensor(buf143, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_490, sp_491], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf138, arg207_1, arg208_1, arg209_1, arg210_1, buf142, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        del buf138
        buf144 = reinterpret_tensor(buf50, (8, 416, 14, 14), (81536, 1, 5824, 416), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [out_324], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf143, buf144, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf139
        del buf140
        del buf141
        del buf142
        del buf143
        # Topologically Sorted Source Nodes: [out_324], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg211_1
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf130, arg216_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg216_1
        del buf130
        buf147 = buf145; del buf145  # reuse
        buf148 = reinterpret_tensor(buf4, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [out_325, input_14, out_326, out_327], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
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
        # Topologically Sorted Source Nodes: [out_327, out_328], Original ATen: [aten.relu, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg221_1
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [out_329, out_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf150, arg222_1, arg223_1, arg224_1, arg225_1, 652288, grid=grid(652288), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf151 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [sp_493], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg226_1, buf151, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg226_1
        # Topologically Sorted Source Nodes: [sp_493], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(reinterpret_tensor(buf150, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf163 = reinterpret_tensor(buf144, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf144  # reuse
        buf153 = reinterpret_tensor(buf163, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_494, sp_495], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf152, arg227_1, arg228_1, arg229_1, arg230_1, buf153, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        buf154 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [sp_496], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf153, buf150, buf154, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf155 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [sp_496, sp_497], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg231_1, buf155, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg231_1
        # Topologically Sorted Source Nodes: [sp_496, sp_497], Original ATen: [aten.add, aten.convolution]
        buf156 = extern_kernels.convolution(buf154, buf155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf154
        buf157 = reinterpret_tensor(buf163, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_498, sp_499], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf156, arg232_1, arg233_1, arg234_1, arg235_1, buf157, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        buf158 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [sp_500], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf157, buf150, buf158, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf159 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [sp_500, sp_501], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg236_1, buf159, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg236_1
        # Topologically Sorted Source Nodes: [sp_500, sp_501], Original ATen: [aten.add, aten.convolution]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf158
        buf161 = reinterpret_tensor(buf163, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_502, sp_503], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf160, arg237_1, arg238_1, arg239_1, arg240_1, buf161, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf160
        buf162 = reinterpret_tensor(buf163, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_331], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf150, buf162, 832, 196, grid=grid(832, 196), stream=stream0)
        buf164 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [out_332], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf163, buf164, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf153
        del buf157
        del buf161
        del buf162
        del buf163
        # Topologically Sorted Source Nodes: [out_332], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg241_1
        buf166 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [out_333, out_334, out_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf166, buf165, arg242_1, arg243_1, arg244_1, arg245_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del buf165
        # Topologically Sorted Source Nodes: [out_336], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg246_1
        buf168 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [out_337, out_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf168, arg247_1, arg248_1, arg249_1, arg250_1, 652288, grid=grid(652288), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        buf169 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [sp_505], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg251_1, buf169, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg251_1
        # Topologically Sorted Source Nodes: [sp_505], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(reinterpret_tensor(buf168, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf181 = reinterpret_tensor(buf164, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf164  # reuse
        buf171 = reinterpret_tensor(buf181, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_506, sp_507], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf170, arg252_1, arg253_1, arg254_1, arg255_1, buf171, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        buf172 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [sp_508], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf171, buf168, buf172, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf173 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [sp_508, sp_509], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg256_1, buf173, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg256_1
        # Topologically Sorted Source Nodes: [sp_508, sp_509], Original ATen: [aten.add, aten.convolution]
        buf174 = extern_kernels.convolution(buf172, buf173, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf172
        buf175 = reinterpret_tensor(buf181, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_510, sp_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf174, arg257_1, arg258_1, arg259_1, arg260_1, buf175, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        buf176 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [sp_512], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf175, buf168, buf176, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf177 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [sp_512, sp_513], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg261_1, buf177, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [sp_512, sp_513], Original ATen: [aten.add, aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf176
        buf179 = reinterpret_tensor(buf181, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_514, sp_515], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf178, arg262_1, arg263_1, arg264_1, arg265_1, buf179, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        del buf178
        buf180 = reinterpret_tensor(buf181, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_339], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf168, buf180, 832, 196, grid=grid(832, 196), stream=stream0)
        buf182 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [out_340], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf181, buf182, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf171
        del buf175
        del buf179
        del buf180
        del buf181
        # Topologically Sorted Source Nodes: [out_340], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg266_1
        buf184 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [out_341, out_342, out_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf184, buf183, arg267_1, arg268_1, arg269_1, arg270_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        del buf183
        # Topologically Sorted Source Nodes: [out_344], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg271_1
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [out_345, out_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf186, arg272_1, arg273_1, arg274_1, arg275_1, 652288, grid=grid(652288), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf187 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [sp_517], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg276_1, buf187, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg276_1
        # Topologically Sorted Source Nodes: [sp_517], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(reinterpret_tensor(buf186, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf199 = reinterpret_tensor(buf182, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf182  # reuse
        buf189 = reinterpret_tensor(buf199, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_518, sp_519], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf188, arg277_1, arg278_1, arg279_1, arg280_1, buf189, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        buf190 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [sp_520], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf189, buf186, buf190, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf191 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [sp_520, sp_521], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg281_1, buf191, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg281_1
        # Topologically Sorted Source Nodes: [sp_520, sp_521], Original ATen: [aten.add, aten.convolution]
        buf192 = extern_kernels.convolution(buf190, buf191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf190
        buf193 = reinterpret_tensor(buf199, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_522, sp_523], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf192, arg282_1, arg283_1, arg284_1, arg285_1, buf193, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        buf194 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [sp_524], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf193, buf186, buf194, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf195 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [sp_524, sp_525], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg286_1, buf195, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg286_1
        # Topologically Sorted Source Nodes: [sp_524, sp_525], Original ATen: [aten.add, aten.convolution]
        buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf194
        buf197 = reinterpret_tensor(buf199, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_526, sp_527], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf196, arg287_1, arg288_1, arg289_1, arg290_1, buf197, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf196
        buf198 = reinterpret_tensor(buf199, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_347], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf186, buf198, 832, 196, grid=grid(832, 196), stream=stream0)
        buf200 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [out_348], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf199, buf200, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf189
        del buf193
        del buf197
        del buf198
        del buf199
        # Topologically Sorted Source Nodes: [out_348], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg291_1
        buf202 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [out_349, out_350, out_351], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf202, buf201, arg292_1, arg293_1, arg294_1, arg295_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        del buf201
        # Topologically Sorted Source Nodes: [out_352], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg296_1
        buf204 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [out_353, out_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf204, arg297_1, arg298_1, arg299_1, arg300_1, 652288, grid=grid(652288), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        buf205 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [sp_529], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg301_1, buf205, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg301_1
        # Topologically Sorted Source Nodes: [sp_529], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(reinterpret_tensor(buf204, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf217 = reinterpret_tensor(buf200, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf200  # reuse
        buf207 = reinterpret_tensor(buf217, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_530, sp_531], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf206, arg302_1, arg303_1, arg304_1, arg305_1, buf207, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        buf208 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [sp_532], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf207, buf204, buf208, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf209 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [sp_532, sp_533], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg306_1, buf209, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg306_1
        # Topologically Sorted Source Nodes: [sp_532, sp_533], Original ATen: [aten.add, aten.convolution]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf208
        buf211 = reinterpret_tensor(buf217, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_534, sp_535], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf210, arg307_1, arg308_1, arg309_1, arg310_1, buf211, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        buf212 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [sp_536], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf211, buf204, buf212, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf213 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [sp_536, sp_537], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg311_1, buf213, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [sp_536, sp_537], Original ATen: [aten.add, aten.convolution]
        buf214 = extern_kernels.convolution(buf212, buf213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf212
        buf215 = reinterpret_tensor(buf217, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_538, sp_539], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf214, arg312_1, arg313_1, arg314_1, arg315_1, buf215, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        del buf214
        buf216 = reinterpret_tensor(buf217, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_355], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf204, buf216, 832, 196, grid=grid(832, 196), stream=stream0)
        buf218 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [out_356], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf217, buf218, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf207
        del buf211
        del buf215
        del buf216
        del buf217
        # Topologically Sorted Source Nodes: [out_356], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg316_1
        buf220 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [out_357, out_358, out_359], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf220, buf219, arg317_1, arg318_1, arg319_1, arg320_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        del buf219
        # Topologically Sorted Source Nodes: [out_360], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg321_1
        buf222 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [out_361, out_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf222, arg322_1, arg323_1, arg324_1, arg325_1, 652288, grid=grid(652288), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        buf223 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [sp_541], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg326_1, buf223, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg326_1
        # Topologically Sorted Source Nodes: [sp_541], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(reinterpret_tensor(buf222, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf235 = reinterpret_tensor(buf218, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf218  # reuse
        buf225 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_542, sp_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf224, arg327_1, arg328_1, arg329_1, arg330_1, buf225, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        buf226 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [sp_544], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf225, buf222, buf226, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf227 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [sp_544, sp_545], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg331_1, buf227, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg331_1
        # Topologically Sorted Source Nodes: [sp_544, sp_545], Original ATen: [aten.add, aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf226
        buf229 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_546, sp_547], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf228, arg332_1, arg333_1, arg334_1, arg335_1, buf229, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        buf230 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [sp_548], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf229, buf222, buf230, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf231 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [sp_548, sp_549], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg336_1, buf231, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg336_1
        # Topologically Sorted Source Nodes: [sp_548, sp_549], Original ATen: [aten.add, aten.convolution]
        buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf230
        buf233 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_550, sp_551], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf232, arg337_1, arg338_1, arg339_1, arg340_1, buf233, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        del buf232
        buf234 = reinterpret_tensor(buf235, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_363], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf222, buf234, 832, 196, grid=grid(832, 196), stream=stream0)
        buf236 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [out_364], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf235, buf236, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf225
        del buf229
        del buf233
        del buf234
        del buf235
        # Topologically Sorted Source Nodes: [out_364], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg341_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg341_1
        buf238 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [out_365, out_366, out_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf238, buf237, arg342_1, arg343_1, arg344_1, arg345_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del buf237
        # Topologically Sorted Source Nodes: [out_368], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg346_1
        buf240 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [out_369, out_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf240, arg347_1, arg348_1, arg349_1, arg350_1, 652288, grid=grid(652288), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        buf241 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [sp_553], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg351_1, buf241, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg351_1
        # Topologically Sorted Source Nodes: [sp_553], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(reinterpret_tensor(buf240, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf253 = reinterpret_tensor(buf236, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf236  # reuse
        buf243 = reinterpret_tensor(buf253, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_554, sp_555], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf242, arg352_1, arg353_1, arg354_1, arg355_1, buf243, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        buf244 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [sp_556], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf243, buf240, buf244, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf245 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [sp_556, sp_557], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg356_1, buf245, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg356_1
        # Topologically Sorted Source Nodes: [sp_556, sp_557], Original ATen: [aten.add, aten.convolution]
        buf246 = extern_kernels.convolution(buf244, buf245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf244
        buf247 = reinterpret_tensor(buf253, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_558, sp_559], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf246, arg357_1, arg358_1, arg359_1, arg360_1, buf247, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        buf248 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [sp_560], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf247, buf240, buf248, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf249 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [sp_560, sp_561], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg361_1, buf249, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg361_1
        # Topologically Sorted Source Nodes: [sp_560, sp_561], Original ATen: [aten.add, aten.convolution]
        buf250 = extern_kernels.convolution(buf248, buf249, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf248
        buf251 = reinterpret_tensor(buf253, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_562, sp_563], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf250, arg362_1, arg363_1, arg364_1, arg365_1, buf251, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del buf250
        buf252 = reinterpret_tensor(buf253, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_371], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf240, buf252, 832, 196, grid=grid(832, 196), stream=stream0)
        buf254 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [out_372], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf253, buf254, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf243
        del buf247
        del buf251
        del buf252
        del buf253
        # Topologically Sorted Source Nodes: [out_372], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, arg366_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg366_1
        buf256 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [out_373, out_374, out_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf256, buf255, arg367_1, arg368_1, arg369_1, arg370_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        del arg370_1
        del buf255
        # Topologically Sorted Source Nodes: [out_376], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, arg371_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg371_1
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [out_377, out_378], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf258, arg372_1, arg373_1, arg374_1, arg375_1, 652288, grid=grid(652288), stream=stream0)
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        buf259 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [sp_565], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg376_1, buf259, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg376_1
        # Topologically Sorted Source Nodes: [sp_565], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(reinterpret_tensor(buf258, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf271 = reinterpret_tensor(buf254, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf254  # reuse
        buf261 = reinterpret_tensor(buf271, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_566, sp_567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf260, arg377_1, arg378_1, arg379_1, arg380_1, buf261, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        buf262 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [sp_568], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf261, buf258, buf262, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf263 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [sp_568, sp_569], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg381_1, buf263, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg381_1
        # Topologically Sorted Source Nodes: [sp_568, sp_569], Original ATen: [aten.add, aten.convolution]
        buf264 = extern_kernels.convolution(buf262, buf263, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf262
        buf265 = reinterpret_tensor(buf271, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_570, sp_571], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf264, arg382_1, arg383_1, arg384_1, arg385_1, buf265, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        buf266 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [sp_572], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf265, buf258, buf266, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf267 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [sp_572, sp_573], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg386_1, buf267, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg386_1
        # Topologically Sorted Source Nodes: [sp_572, sp_573], Original ATen: [aten.add, aten.convolution]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf266
        buf269 = reinterpret_tensor(buf271, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_574, sp_575], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf268, arg387_1, arg388_1, arg389_1, arg390_1, buf269, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        del buf268
        buf270 = reinterpret_tensor(buf271, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_379], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf258, buf270, 832, 196, grid=grid(832, 196), stream=stream0)
        buf272 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [out_380], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf271, buf272, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf261
        del buf265
        del buf269
        del buf270
        del buf271
        # Topologically Sorted Source Nodes: [out_380], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, arg391_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg391_1
        buf274 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [out_381, out_382, out_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf274, buf273, arg392_1, arg393_1, arg394_1, arg395_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        del buf273
        # Topologically Sorted Source Nodes: [out_384], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, arg396_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg396_1
        buf276 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [out_385, out_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf276, arg397_1, arg398_1, arg399_1, arg400_1, 652288, grid=grid(652288), stream=stream0)
        del arg397_1
        del arg398_1
        del arg399_1
        del arg400_1
        buf277 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [sp_577], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg401_1, buf277, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg401_1
        # Topologically Sorted Source Nodes: [sp_577], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(reinterpret_tensor(buf276, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf289 = reinterpret_tensor(buf272, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf272  # reuse
        buf279 = reinterpret_tensor(buf289, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_578, sp_579], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf278, arg402_1, arg403_1, arg404_1, arg405_1, buf279, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        buf280 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [sp_580], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf279, buf276, buf280, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf281 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [sp_580, sp_581], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg406_1, buf281, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg406_1
        # Topologically Sorted Source Nodes: [sp_580, sp_581], Original ATen: [aten.add, aten.convolution]
        buf282 = extern_kernels.convolution(buf280, buf281, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf280
        buf283 = reinterpret_tensor(buf289, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_582, sp_583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf282, arg407_1, arg408_1, arg409_1, arg410_1, buf283, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        del arg410_1
        buf284 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [sp_584], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf283, buf276, buf284, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf285 = buf281; del buf281  # reuse
        # Topologically Sorted Source Nodes: [sp_584, sp_585], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg411_1, buf285, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg411_1
        # Topologically Sorted Source Nodes: [sp_584, sp_585], Original ATen: [aten.add, aten.convolution]
        buf286 = extern_kernels.convolution(buf284, buf285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf284
        buf287 = reinterpret_tensor(buf289, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_586, sp_587], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf286, arg412_1, arg413_1, arg414_1, arg415_1, buf287, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        del buf286
        buf288 = reinterpret_tensor(buf289, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_387], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf276, buf288, 832, 196, grid=grid(832, 196), stream=stream0)
        buf290 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [out_388], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf289, buf290, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf279
        del buf283
        del buf287
        del buf288
        del buf289
        # Topologically Sorted Source Nodes: [out_388], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, arg416_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg416_1
        buf292 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [out_389, out_390, out_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf292, buf291, arg417_1, arg418_1, arg419_1, arg420_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg417_1
        del arg418_1
        del arg419_1
        del arg420_1
        del buf291
        # Topologically Sorted Source Nodes: [out_392], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, arg421_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg421_1
        buf294 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [out_393, out_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf294, arg422_1, arg423_1, arg424_1, arg425_1, 652288, grid=grid(652288), stream=stream0)
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        buf295 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [sp_589], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg426_1, buf295, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg426_1
        # Topologically Sorted Source Nodes: [sp_589], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(reinterpret_tensor(buf294, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf295, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf307 = reinterpret_tensor(buf290, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf290  # reuse
        buf297 = reinterpret_tensor(buf307, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_590, sp_591], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf296, arg427_1, arg428_1, arg429_1, arg430_1, buf297, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg427_1
        del arg428_1
        del arg429_1
        del arg430_1
        buf298 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [sp_592], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf297, buf294, buf298, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf299 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [sp_592, sp_593], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg431_1, buf299, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg431_1
        # Topologically Sorted Source Nodes: [sp_592, sp_593], Original ATen: [aten.add, aten.convolution]
        buf300 = extern_kernels.convolution(buf298, buf299, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf298
        buf301 = reinterpret_tensor(buf307, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_594, sp_595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf300, arg432_1, arg433_1, arg434_1, arg435_1, buf301, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg432_1
        del arg433_1
        del arg434_1
        del arg435_1
        buf302 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [sp_596], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf301, buf294, buf302, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf303 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [sp_596, sp_597], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg436_1, buf303, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg436_1
        # Topologically Sorted Source Nodes: [sp_596, sp_597], Original ATen: [aten.add, aten.convolution]
        buf304 = extern_kernels.convolution(buf302, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf302
        buf305 = reinterpret_tensor(buf307, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_598, sp_599], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf304, arg437_1, arg438_1, arg439_1, arg440_1, buf305, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg437_1
        del arg438_1
        del arg439_1
        del arg440_1
        del buf304
        buf306 = reinterpret_tensor(buf307, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_395], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf294, buf306, 832, 196, grid=grid(832, 196), stream=stream0)
        buf308 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [out_396], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf307, buf308, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf297
        del buf301
        del buf305
        del buf306
        del buf307
        # Topologically Sorted Source Nodes: [out_396], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, arg441_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg441_1
        buf310 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [out_397, out_398, out_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf310, buf309, arg442_1, arg443_1, arg444_1, arg445_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        del buf309
        # Topologically Sorted Source Nodes: [out_400], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, arg446_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg446_1
        buf312 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [out_401, out_402], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf312, arg447_1, arg448_1, arg449_1, arg450_1, 652288, grid=grid(652288), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        buf313 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [sp_601], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg451_1, buf313, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg451_1
        # Topologically Sorted Source Nodes: [sp_601], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(reinterpret_tensor(buf312, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf313, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf325 = reinterpret_tensor(buf308, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf308  # reuse
        buf315 = reinterpret_tensor(buf325, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_602, sp_603], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf314, arg452_1, arg453_1, arg454_1, arg455_1, buf315, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        buf316 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [sp_604], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf315, buf312, buf316, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf317 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [sp_604, sp_605], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg456_1, buf317, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg456_1
        # Topologically Sorted Source Nodes: [sp_604, sp_605], Original ATen: [aten.add, aten.convolution]
        buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf316
        buf319 = reinterpret_tensor(buf325, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_606, sp_607], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf318, arg457_1, arg458_1, arg459_1, arg460_1, buf319, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg457_1
        del arg458_1
        del arg459_1
        del arg460_1
        buf320 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [sp_608], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf319, buf312, buf320, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf321 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [sp_608, sp_609], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg461_1, buf321, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg461_1
        # Topologically Sorted Source Nodes: [sp_608, sp_609], Original ATen: [aten.add, aten.convolution]
        buf322 = extern_kernels.convolution(buf320, buf321, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf320
        buf323 = reinterpret_tensor(buf325, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_610, sp_611], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf322, arg462_1, arg463_1, arg464_1, arg465_1, buf323, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg462_1
        del arg463_1
        del arg464_1
        del arg465_1
        del buf322
        buf324 = reinterpret_tensor(buf325, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_403], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf312, buf324, 832, 196, grid=grid(832, 196), stream=stream0)
        buf326 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [out_404], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf325, buf326, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf315
        del buf319
        del buf323
        del buf324
        del buf325
        # Topologically Sorted Source Nodes: [out_404], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, arg466_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg466_1
        buf328 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [out_405, out_406, out_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf328, buf327, arg467_1, arg468_1, arg469_1, arg470_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg467_1
        del arg468_1
        del arg469_1
        del arg470_1
        del buf327
        # Topologically Sorted Source Nodes: [out_408], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, arg471_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg471_1
        buf330 = buf329; del buf329  # reuse
        # Topologically Sorted Source Nodes: [out_409, out_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf330, arg472_1, arg473_1, arg474_1, arg475_1, 652288, grid=grid(652288), stream=stream0)
        del arg472_1
        del arg473_1
        del arg474_1
        del arg475_1
        buf331 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [sp_613], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg476_1, buf331, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg476_1
        # Topologically Sorted Source Nodes: [sp_613], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(reinterpret_tensor(buf330, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf343 = reinterpret_tensor(buf326, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf326  # reuse
        buf333 = reinterpret_tensor(buf343, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_614, sp_615], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf332, arg477_1, arg478_1, arg479_1, arg480_1, buf333, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg477_1
        del arg478_1
        del arg479_1
        del arg480_1
        buf334 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [sp_616], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf333, buf330, buf334, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf335 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [sp_616, sp_617], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg481_1, buf335, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg481_1
        # Topologically Sorted Source Nodes: [sp_616, sp_617], Original ATen: [aten.add, aten.convolution]
        buf336 = extern_kernels.convolution(buf334, buf335, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf334
        buf337 = reinterpret_tensor(buf343, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_618, sp_619], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf336, arg482_1, arg483_1, arg484_1, arg485_1, buf337, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg482_1
        del arg483_1
        del arg484_1
        del arg485_1
        buf338 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [sp_620], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf337, buf330, buf338, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf339 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [sp_620, sp_621], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg486_1, buf339, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg486_1
        # Topologically Sorted Source Nodes: [sp_620, sp_621], Original ATen: [aten.add, aten.convolution]
        buf340 = extern_kernels.convolution(buf338, buf339, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf338
        buf341 = reinterpret_tensor(buf343, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_622, sp_623], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf340, arg487_1, arg488_1, arg489_1, arg490_1, buf341, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg487_1
        del arg488_1
        del arg489_1
        del arg490_1
        del buf340
        buf342 = reinterpret_tensor(buf343, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_411], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf330, buf342, 832, 196, grid=grid(832, 196), stream=stream0)
        buf344 = buf330; del buf330  # reuse
        # Topologically Sorted Source Nodes: [out_412], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf343, buf344, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf333
        del buf337
        del buf341
        del buf342
        del buf343
        # Topologically Sorted Source Nodes: [out_412], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, arg491_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg491_1
        buf346 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [out_413, out_414, out_415], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf346, buf345, arg492_1, arg493_1, arg494_1, arg495_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg492_1
        del arg493_1
        del arg494_1
        del arg495_1
        del buf345
        # Topologically Sorted Source Nodes: [out_416], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, arg496_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg496_1
        buf348 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [out_417, out_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf348, arg497_1, arg498_1, arg499_1, arg500_1, 652288, grid=grid(652288), stream=stream0)
        del arg497_1
        del arg498_1
        del arg499_1
        del arg500_1
        buf349 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [sp_625], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg501_1, buf349, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg501_1
        # Topologically Sorted Source Nodes: [sp_625], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(reinterpret_tensor(buf348, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf349, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf361 = reinterpret_tensor(buf344, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf344  # reuse
        buf351 = reinterpret_tensor(buf361, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_626, sp_627], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf350, arg502_1, arg503_1, arg504_1, arg505_1, buf351, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg502_1
        del arg503_1
        del arg504_1
        del arg505_1
        buf352 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [sp_628], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf351, buf348, buf352, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf353 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [sp_628, sp_629], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg506_1, buf353, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg506_1
        # Topologically Sorted Source Nodes: [sp_628, sp_629], Original ATen: [aten.add, aten.convolution]
        buf354 = extern_kernels.convolution(buf352, buf353, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf352
        buf355 = reinterpret_tensor(buf361, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_630, sp_631], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf354, arg507_1, arg508_1, arg509_1, arg510_1, buf355, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg507_1
        del arg508_1
        del arg509_1
        del arg510_1
        buf356 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [sp_632], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf355, buf348, buf356, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf357 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [sp_632, sp_633], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg511_1, buf357, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg511_1
        # Topologically Sorted Source Nodes: [sp_632, sp_633], Original ATen: [aten.add, aten.convolution]
        buf358 = extern_kernels.convolution(buf356, buf357, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf356
        buf359 = reinterpret_tensor(buf361, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_634, sp_635], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf358, arg512_1, arg513_1, arg514_1, arg515_1, buf359, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg512_1
        del arg513_1
        del arg514_1
        del arg515_1
        del buf358
        buf360 = reinterpret_tensor(buf361, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_419], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf348, buf360, 832, 196, grid=grid(832, 196), stream=stream0)
        buf362 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [out_420], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf361, buf362, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf351
        del buf355
        del buf359
        del buf360
        del buf361
        # Topologically Sorted Source Nodes: [out_420], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, arg516_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg516_1
        buf364 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [out_421, out_422, out_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf364, buf363, arg517_1, arg518_1, arg519_1, arg520_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg517_1
        del arg518_1
        del arg519_1
        del arg520_1
        del buf363
        # Topologically Sorted Source Nodes: [out_424], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, arg521_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg521_1
        buf366 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [out_425, out_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf366, arg522_1, arg523_1, arg524_1, arg525_1, 652288, grid=grid(652288), stream=stream0)
        del arg522_1
        del arg523_1
        del arg524_1
        del arg525_1
        buf367 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [sp_637], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg526_1, buf367, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg526_1
        # Topologically Sorted Source Nodes: [sp_637], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(reinterpret_tensor(buf366, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf379 = reinterpret_tensor(buf362, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf362  # reuse
        buf369 = reinterpret_tensor(buf379, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_638, sp_639], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf368, arg527_1, arg528_1, arg529_1, arg530_1, buf369, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg527_1
        del arg528_1
        del arg529_1
        del arg530_1
        buf370 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [sp_640], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf369, buf366, buf370, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf371 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [sp_640, sp_641], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg531_1, buf371, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg531_1
        # Topologically Sorted Source Nodes: [sp_640, sp_641], Original ATen: [aten.add, aten.convolution]
        buf372 = extern_kernels.convolution(buf370, buf371, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf370
        buf373 = reinterpret_tensor(buf379, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_642, sp_643], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf372, arg532_1, arg533_1, arg534_1, arg535_1, buf373, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg532_1
        del arg533_1
        del arg534_1
        del arg535_1
        buf374 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [sp_644], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf373, buf366, buf374, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf375 = buf371; del buf371  # reuse
        # Topologically Sorted Source Nodes: [sp_644, sp_645], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg536_1, buf375, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg536_1
        # Topologically Sorted Source Nodes: [sp_644, sp_645], Original ATen: [aten.add, aten.convolution]
        buf376 = extern_kernels.convolution(buf374, buf375, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf374
        buf377 = reinterpret_tensor(buf379, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_646, sp_647], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf376, arg537_1, arg538_1, arg539_1, arg540_1, buf377, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg537_1
        del arg538_1
        del arg539_1
        del arg540_1
        del buf376
        buf378 = reinterpret_tensor(buf379, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_427], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf366, buf378, 832, 196, grid=grid(832, 196), stream=stream0)
        buf380 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [out_428], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf379, buf380, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf369
        del buf373
        del buf377
        del buf378
        del buf379
        # Topologically Sorted Source Nodes: [out_428], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, arg541_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg541_1
        buf382 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [out_429, out_430, out_431], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf382, buf381, arg542_1, arg543_1, arg544_1, arg545_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg542_1
        del arg543_1
        del arg544_1
        del arg545_1
        del buf381
        # Topologically Sorted Source Nodes: [out_432], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, arg546_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg546_1
        buf384 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [out_433, out_434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf384, arg547_1, arg548_1, arg549_1, arg550_1, 652288, grid=grid(652288), stream=stream0)
        del arg547_1
        del arg548_1
        del arg549_1
        del arg550_1
        buf385 = buf375; del buf375  # reuse
        # Topologically Sorted Source Nodes: [sp_649], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg551_1, buf385, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg551_1
        # Topologically Sorted Source Nodes: [sp_649], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(reinterpret_tensor(buf384, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf385, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf397 = reinterpret_tensor(buf380, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf380  # reuse
        buf387 = reinterpret_tensor(buf397, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_650, sp_651], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf386, arg552_1, arg553_1, arg554_1, arg555_1, buf387, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg552_1
        del arg553_1
        del arg554_1
        del arg555_1
        buf388 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [sp_652], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf387, buf384, buf388, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf389 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [sp_652, sp_653], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg556_1, buf389, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg556_1
        # Topologically Sorted Source Nodes: [sp_652, sp_653], Original ATen: [aten.add, aten.convolution]
        buf390 = extern_kernels.convolution(buf388, buf389, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf388
        buf391 = reinterpret_tensor(buf397, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_654, sp_655], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf390, arg557_1, arg558_1, arg559_1, arg560_1, buf391, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg557_1
        del arg558_1
        del arg559_1
        del arg560_1
        buf392 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [sp_656], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf391, buf384, buf392, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf393 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [sp_656, sp_657], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg561_1, buf393, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg561_1
        # Topologically Sorted Source Nodes: [sp_656, sp_657], Original ATen: [aten.add, aten.convolution]
        buf394 = extern_kernels.convolution(buf392, buf393, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf392
        buf395 = reinterpret_tensor(buf397, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_658, sp_659], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf394, arg562_1, arg563_1, arg564_1, arg565_1, buf395, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg562_1
        del arg563_1
        del arg564_1
        del arg565_1
        del buf394
        buf396 = reinterpret_tensor(buf397, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_435], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf384, buf396, 832, 196, grid=grid(832, 196), stream=stream0)
        buf398 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [out_436], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf397, buf398, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf387
        del buf391
        del buf395
        del buf396
        del buf397
        # Topologically Sorted Source Nodes: [out_436], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, arg566_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg566_1
        buf400 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [out_437, out_438, out_439], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf400, buf399, arg567_1, arg568_1, arg569_1, arg570_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg567_1
        del arg568_1
        del arg569_1
        del arg570_1
        del buf399
        # Topologically Sorted Source Nodes: [out_440], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, arg571_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg571_1
        buf402 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [out_441, out_442], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf402, arg572_1, arg573_1, arg574_1, arg575_1, 652288, grid=grid(652288), stream=stream0)
        del arg572_1
        del arg573_1
        del arg574_1
        del arg575_1
        buf403 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [sp_661], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg576_1, buf403, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg576_1
        # Topologically Sorted Source Nodes: [sp_661], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(reinterpret_tensor(buf402, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf403, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf415 = reinterpret_tensor(buf398, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf398  # reuse
        buf405 = reinterpret_tensor(buf415, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_662, sp_663], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf404, arg577_1, arg578_1, arg579_1, arg580_1, buf405, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg577_1
        del arg578_1
        del arg579_1
        del arg580_1
        buf406 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [sp_664], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf405, buf402, buf406, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf407 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [sp_664, sp_665], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg581_1, buf407, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg581_1
        # Topologically Sorted Source Nodes: [sp_664, sp_665], Original ATen: [aten.add, aten.convolution]
        buf408 = extern_kernels.convolution(buf406, buf407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf406
        buf409 = reinterpret_tensor(buf415, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_666, sp_667], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf408, arg582_1, arg583_1, arg584_1, arg585_1, buf409, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg582_1
        del arg583_1
        del arg584_1
        del arg585_1
        buf410 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [sp_668], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf409, buf402, buf410, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf411 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [sp_668, sp_669], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg586_1, buf411, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg586_1
        # Topologically Sorted Source Nodes: [sp_668, sp_669], Original ATen: [aten.add, aten.convolution]
        buf412 = extern_kernels.convolution(buf410, buf411, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf410
        buf413 = reinterpret_tensor(buf415, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_670, sp_671], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf412, arg587_1, arg588_1, arg589_1, arg590_1, buf413, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg587_1
        del arg588_1
        del arg589_1
        del arg590_1
        del buf412
        buf414 = reinterpret_tensor(buf415, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_443], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf402, buf414, 832, 196, grid=grid(832, 196), stream=stream0)
        buf416 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [out_444], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf415, buf416, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf405
        del buf409
        del buf413
        del buf414
        del buf415
        # Topologically Sorted Source Nodes: [out_444], Original ATen: [aten.convolution]
        buf417 = extern_kernels.convolution(buf416, arg591_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf417, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg591_1
        buf418 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [out_445, out_446, out_447], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf418, buf417, arg592_1, arg593_1, arg594_1, arg595_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg592_1
        del arg593_1
        del arg594_1
        del arg595_1
        del buf417
        # Topologically Sorted Source Nodes: [out_448], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, arg596_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg596_1
        buf420 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [out_449, out_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf420, arg597_1, arg598_1, arg599_1, arg600_1, 652288, grid=grid(652288), stream=stream0)
        del arg597_1
        del arg598_1
        del arg599_1
        del arg600_1
        buf421 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [sp_673], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg601_1, buf421, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg601_1
        # Topologically Sorted Source Nodes: [sp_673], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(reinterpret_tensor(buf420, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf421, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf433 = reinterpret_tensor(buf416, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf416  # reuse
        buf423 = reinterpret_tensor(buf433, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_674, sp_675], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf422, arg602_1, arg603_1, arg604_1, arg605_1, buf423, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg602_1
        del arg603_1
        del arg604_1
        del arg605_1
        buf424 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [sp_676], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf423, buf420, buf424, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf425 = buf421; del buf421  # reuse
        # Topologically Sorted Source Nodes: [sp_676, sp_677], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg606_1, buf425, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg606_1
        # Topologically Sorted Source Nodes: [sp_676, sp_677], Original ATen: [aten.add, aten.convolution]
        buf426 = extern_kernels.convolution(buf424, buf425, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf424
        buf427 = reinterpret_tensor(buf433, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_678, sp_679], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf426, arg607_1, arg608_1, arg609_1, arg610_1, buf427, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg607_1
        del arg608_1
        del arg609_1
        del arg610_1
        buf428 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [sp_680], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf427, buf420, buf428, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf429 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [sp_680, sp_681], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg611_1, buf429, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg611_1
        # Topologically Sorted Source Nodes: [sp_680, sp_681], Original ATen: [aten.add, aten.convolution]
        buf430 = extern_kernels.convolution(buf428, buf429, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf428
        buf431 = reinterpret_tensor(buf433, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_682, sp_683], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf430, arg612_1, arg613_1, arg614_1, arg615_1, buf431, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg612_1
        del arg613_1
        del arg614_1
        del arg615_1
        del buf430
        buf432 = reinterpret_tensor(buf433, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_451], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf420, buf432, 832, 196, grid=grid(832, 196), stream=stream0)
        buf434 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [out_452], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf433, buf434, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf423
        del buf427
        del buf431
        del buf432
        del buf433
        # Topologically Sorted Source Nodes: [out_452], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, arg616_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg616_1
        buf436 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [out_453, out_454, out_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf436, buf435, arg617_1, arg618_1, arg619_1, arg620_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg617_1
        del arg618_1
        del arg619_1
        del arg620_1
        del buf435
        # Topologically Sorted Source Nodes: [out_456], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, arg621_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg621_1
        buf438 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [out_457, out_458], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf438, arg622_1, arg623_1, arg624_1, arg625_1, 652288, grid=grid(652288), stream=stream0)
        del arg622_1
        del arg623_1
        del arg624_1
        del arg625_1
        buf439 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [sp_685], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg626_1, buf439, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg626_1
        # Topologically Sorted Source Nodes: [sp_685], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(reinterpret_tensor(buf438, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf439, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf451 = reinterpret_tensor(buf434, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf434  # reuse
        buf441 = reinterpret_tensor(buf451, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_686, sp_687], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf440, arg627_1, arg628_1, arg629_1, arg630_1, buf441, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg627_1
        del arg628_1
        del arg629_1
        del arg630_1
        buf442 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [sp_688], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf441, buf438, buf442, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf443 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [sp_688, sp_689], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg631_1, buf443, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg631_1
        # Topologically Sorted Source Nodes: [sp_688, sp_689], Original ATen: [aten.add, aten.convolution]
        buf444 = extern_kernels.convolution(buf442, buf443, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf442
        buf445 = reinterpret_tensor(buf451, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_690, sp_691], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf444, arg632_1, arg633_1, arg634_1, arg635_1, buf445, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg632_1
        del arg633_1
        del arg634_1
        del arg635_1
        buf446 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [sp_692], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf445, buf438, buf446, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf447 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [sp_692, sp_693], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg636_1, buf447, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg636_1
        # Topologically Sorted Source Nodes: [sp_692, sp_693], Original ATen: [aten.add, aten.convolution]
        buf448 = extern_kernels.convolution(buf446, buf447, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf446
        buf449 = reinterpret_tensor(buf451, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_694, sp_695], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf448, arg637_1, arg638_1, arg639_1, arg640_1, buf449, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg637_1
        del arg638_1
        del arg639_1
        del arg640_1
        del buf448
        buf450 = reinterpret_tensor(buf451, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_459], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf438, buf450, 832, 196, grid=grid(832, 196), stream=stream0)
        buf452 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [out_460], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf451, buf452, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf441
        del buf445
        del buf449
        del buf450
        del buf451
        # Topologically Sorted Source Nodes: [out_460], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, arg641_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg641_1
        buf454 = buf436; del buf436  # reuse
        # Topologically Sorted Source Nodes: [out_461, out_462, out_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf454, buf453, arg642_1, arg643_1, arg644_1, arg645_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg642_1
        del arg643_1
        del arg644_1
        del arg645_1
        del buf453
        # Topologically Sorted Source Nodes: [out_464], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf454, arg646_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg646_1
        buf456 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [out_465, out_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf456, arg647_1, arg648_1, arg649_1, arg650_1, 652288, grid=grid(652288), stream=stream0)
        del arg647_1
        del arg648_1
        del arg649_1
        del arg650_1
        buf457 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [sp_697], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg651_1, buf457, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg651_1
        # Topologically Sorted Source Nodes: [sp_697], Original ATen: [aten.convolution]
        buf458 = extern_kernels.convolution(reinterpret_tensor(buf456, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf457, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf458, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf469 = reinterpret_tensor(buf452, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf452  # reuse
        buf459 = reinterpret_tensor(buf469, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_698, sp_699], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf458, arg652_1, arg653_1, arg654_1, arg655_1, buf459, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg652_1
        del arg653_1
        del arg654_1
        del arg655_1
        buf460 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [sp_700], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf459, buf456, buf460, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf461 = buf457; del buf457  # reuse
        # Topologically Sorted Source Nodes: [sp_700, sp_701], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg656_1, buf461, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg656_1
        # Topologically Sorted Source Nodes: [sp_700, sp_701], Original ATen: [aten.add, aten.convolution]
        buf462 = extern_kernels.convolution(buf460, buf461, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf460
        buf463 = reinterpret_tensor(buf469, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_702, sp_703], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf462, arg657_1, arg658_1, arg659_1, arg660_1, buf463, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg657_1
        del arg658_1
        del arg659_1
        del arg660_1
        buf464 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [sp_704], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf463, buf456, buf464, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf465 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [sp_704, sp_705], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg661_1, buf465, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg661_1
        # Topologically Sorted Source Nodes: [sp_704, sp_705], Original ATen: [aten.add, aten.convolution]
        buf466 = extern_kernels.convolution(buf464, buf465, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf464
        buf467 = reinterpret_tensor(buf469, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_706, sp_707], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf466, arg662_1, arg663_1, arg664_1, arg665_1, buf467, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg662_1
        del arg663_1
        del arg664_1
        del arg665_1
        del buf466
        buf468 = reinterpret_tensor(buf469, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_467], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf456, buf468, 832, 196, grid=grid(832, 196), stream=stream0)
        buf470 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [out_468], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf469, buf470, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf459
        del buf463
        del buf467
        del buf468
        del buf469
        # Topologically Sorted Source Nodes: [out_468], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, arg666_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg666_1
        buf472 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [out_469, out_470, out_471], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf472, buf471, arg667_1, arg668_1, arg669_1, arg670_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg667_1
        del arg668_1
        del arg669_1
        del arg670_1
        del buf471
        # Topologically Sorted Source Nodes: [out_472], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, arg671_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg671_1
        buf474 = buf473; del buf473  # reuse
        # Topologically Sorted Source Nodes: [out_473, out_474], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf474, arg672_1, arg673_1, arg674_1, arg675_1, 652288, grid=grid(652288), stream=stream0)
        del arg672_1
        del arg673_1
        del arg674_1
        del arg675_1
        buf475 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [sp_709], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg676_1, buf475, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg676_1
        # Topologically Sorted Source Nodes: [sp_709], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(reinterpret_tensor(buf474, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf475, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf487 = reinterpret_tensor(buf470, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf470  # reuse
        buf477 = reinterpret_tensor(buf487, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_710, sp_711], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf476, arg677_1, arg678_1, arg679_1, arg680_1, buf477, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg677_1
        del arg678_1
        del arg679_1
        del arg680_1
        buf478 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [sp_712], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf477, buf474, buf478, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf479 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [sp_712, sp_713], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg681_1, buf479, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg681_1
        # Topologically Sorted Source Nodes: [sp_712, sp_713], Original ATen: [aten.add, aten.convolution]
        buf480 = extern_kernels.convolution(buf478, buf479, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf478
        buf481 = reinterpret_tensor(buf487, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_714, sp_715], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf480, arg682_1, arg683_1, arg684_1, arg685_1, buf481, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg682_1
        del arg683_1
        del arg684_1
        del arg685_1
        buf482 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [sp_716], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf481, buf474, buf482, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf483 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [sp_716, sp_717], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg686_1, buf483, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg686_1
        # Topologically Sorted Source Nodes: [sp_716, sp_717], Original ATen: [aten.add, aten.convolution]
        buf484 = extern_kernels.convolution(buf482, buf483, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf482
        buf485 = reinterpret_tensor(buf487, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_718, sp_719], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf484, arg687_1, arg688_1, arg689_1, arg690_1, buf485, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg687_1
        del arg688_1
        del arg689_1
        del arg690_1
        del buf484
        buf486 = reinterpret_tensor(buf487, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_475], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf474, buf486, 832, 196, grid=grid(832, 196), stream=stream0)
        buf488 = buf474; del buf474  # reuse
        # Topologically Sorted Source Nodes: [out_476], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf487, buf488, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf477
        del buf481
        del buf485
        del buf486
        del buf487
        # Topologically Sorted Source Nodes: [out_476], Original ATen: [aten.convolution]
        buf489 = extern_kernels.convolution(buf488, arg691_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf489, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg691_1
        buf490 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [out_477, out_478, out_479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf490, buf489, arg692_1, arg693_1, arg694_1, arg695_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg692_1
        del arg693_1
        del arg694_1
        del arg695_1
        del buf489
        # Topologically Sorted Source Nodes: [out_480], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, arg696_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg696_1
        buf492 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [out_481, out_482], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf492, arg697_1, arg698_1, arg699_1, arg700_1, 652288, grid=grid(652288), stream=stream0)
        del arg697_1
        del arg698_1
        del arg699_1
        del arg700_1
        buf493 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [sp_721], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg701_1, buf493, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg701_1
        # Topologically Sorted Source Nodes: [sp_721], Original ATen: [aten.convolution]
        buf494 = extern_kernels.convolution(reinterpret_tensor(buf492, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf493, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf494, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf505 = reinterpret_tensor(buf488, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf488  # reuse
        buf495 = reinterpret_tensor(buf505, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_722, sp_723], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf494, arg702_1, arg703_1, arg704_1, arg705_1, buf495, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg702_1
        del arg703_1
        del arg704_1
        del arg705_1
        buf496 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [sp_724], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf495, buf492, buf496, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf497 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [sp_724, sp_725], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg706_1, buf497, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg706_1
        # Topologically Sorted Source Nodes: [sp_724, sp_725], Original ATen: [aten.add, aten.convolution]
        buf498 = extern_kernels.convolution(buf496, buf497, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf496
        buf499 = reinterpret_tensor(buf505, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_726, sp_727], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf498, arg707_1, arg708_1, arg709_1, arg710_1, buf499, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg707_1
        del arg708_1
        del arg709_1
        del arg710_1
        buf500 = buf498; del buf498  # reuse
        # Topologically Sorted Source Nodes: [sp_728], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf499, buf492, buf500, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf501 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [sp_728, sp_729], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg711_1, buf501, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg711_1
        # Topologically Sorted Source Nodes: [sp_728, sp_729], Original ATen: [aten.add, aten.convolution]
        buf502 = extern_kernels.convolution(buf500, buf501, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf500
        buf503 = reinterpret_tensor(buf505, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_730, sp_731], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf502, arg712_1, arg713_1, arg714_1, arg715_1, buf503, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg712_1
        del arg713_1
        del arg714_1
        del arg715_1
        del buf502
        buf504 = reinterpret_tensor(buf505, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_483], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf492, buf504, 832, 196, grid=grid(832, 196), stream=stream0)
        buf506 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [out_484], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf505, buf506, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf495
        del buf499
        del buf503
        del buf504
        del buf505
        # Topologically Sorted Source Nodes: [out_484], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, arg716_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg716_1
        buf508 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [out_485, out_486, out_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf508, buf507, arg717_1, arg718_1, arg719_1, arg720_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg717_1
        del arg718_1
        del arg719_1
        del arg720_1
        del buf507
        # Topologically Sorted Source Nodes: [out_488], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, arg721_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg721_1
        buf510 = buf509; del buf509  # reuse
        # Topologically Sorted Source Nodes: [out_489, out_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf510, arg722_1, arg723_1, arg724_1, arg725_1, 652288, grid=grid(652288), stream=stream0)
        del arg722_1
        del arg723_1
        del arg724_1
        del arg725_1
        buf511 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [sp_733], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg726_1, buf511, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg726_1
        # Topologically Sorted Source Nodes: [sp_733], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(reinterpret_tensor(buf510, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf511, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf523 = reinterpret_tensor(buf506, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf506  # reuse
        buf513 = reinterpret_tensor(buf523, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_734, sp_735], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf512, arg727_1, arg728_1, arg729_1, arg730_1, buf513, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg727_1
        del arg728_1
        del arg729_1
        del arg730_1
        buf514 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [sp_736], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf513, buf510, buf514, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf515 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [sp_736, sp_737], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg731_1, buf515, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg731_1
        # Topologically Sorted Source Nodes: [sp_736, sp_737], Original ATen: [aten.add, aten.convolution]
        buf516 = extern_kernels.convolution(buf514, buf515, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf514
        buf517 = reinterpret_tensor(buf523, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_738, sp_739], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf516, arg732_1, arg733_1, arg734_1, arg735_1, buf517, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg732_1
        del arg733_1
        del arg734_1
        del arg735_1
        buf518 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [sp_740], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf517, buf510, buf518, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf519 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [sp_740, sp_741], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg736_1, buf519, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg736_1
        # Topologically Sorted Source Nodes: [sp_740, sp_741], Original ATen: [aten.add, aten.convolution]
        buf520 = extern_kernels.convolution(buf518, buf519, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf518
        buf521 = reinterpret_tensor(buf523, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_742, sp_743], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf520, arg737_1, arg738_1, arg739_1, arg740_1, buf521, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg737_1
        del arg738_1
        del arg739_1
        del arg740_1
        del buf520
        buf522 = reinterpret_tensor(buf523, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_491], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf510, buf522, 832, 196, grid=grid(832, 196), stream=stream0)
        buf524 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [out_492], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf523, buf524, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf513
        del buf517
        del buf521
        del buf522
        del buf523
        # Topologically Sorted Source Nodes: [out_492], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf524, arg741_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg741_1
        buf526 = buf508; del buf508  # reuse
        # Topologically Sorted Source Nodes: [out_493, out_494, out_495], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf526, buf525, arg742_1, arg743_1, arg744_1, arg745_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg742_1
        del arg743_1
        del arg744_1
        del arg745_1
        del buf525
        # Topologically Sorted Source Nodes: [out_496], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, arg746_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 416, 14, 14), (81536, 1, 5824, 416))
        del arg746_1
        buf528 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [out_497, out_498], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf528, arg747_1, arg748_1, arg749_1, arg750_1, 652288, grid=grid(652288), stream=stream0)
        del arg747_1
        del arg748_1
        del arg749_1
        del arg750_1
        buf529 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [sp_745], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(arg751_1, buf529, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg751_1
        # Topologically Sorted Source Nodes: [sp_745], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(reinterpret_tensor(buf528, (8, 104, 14, 14), (81536, 1, 5824, 416), 0), buf529, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (8, 104, 14, 14), (20384, 1, 1456, 104))
        buf541 = reinterpret_tensor(buf524, (8, 416, 14, 14), (81536, 196, 14, 1), 0); del buf524  # reuse
        buf531 = reinterpret_tensor(buf541, (8, 104, 14, 14), (81536, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_746, sp_747], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf530, arg752_1, arg753_1, arg754_1, arg755_1, buf531, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg752_1
        del arg753_1
        del arg754_1
        del arg755_1
        buf532 = buf530; del buf530  # reuse
        # Topologically Sorted Source Nodes: [sp_748], Original ATen: [aten.add]
        triton_poi_fused_add_33.run(buf531, buf528, buf532, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf533 = buf529; del buf529  # reuse
        # Topologically Sorted Source Nodes: [sp_748, sp_749], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg756_1, buf533, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg756_1
        # Topologically Sorted Source Nodes: [sp_748, sp_749], Original ATen: [aten.add, aten.convolution]
        buf534 = extern_kernels.convolution(buf532, buf533, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf532
        buf535 = reinterpret_tensor(buf541, (8, 104, 14, 14), (81536, 196, 14, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_750, sp_751], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf534, arg757_1, arg758_1, arg759_1, arg760_1, buf535, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg757_1
        del arg758_1
        del arg759_1
        del arg760_1
        buf536 = buf534; del buf534  # reuse
        # Topologically Sorted Source Nodes: [sp_752], Original ATen: [aten.add]
        triton_poi_fused_add_34.run(buf535, buf528, buf536, 1568, 104, grid=grid(1568, 104), stream=stream0)
        buf537 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [sp_752, sp_753], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_27.run(arg761_1, buf537, 10816, 9, grid=grid(10816, 9), stream=stream0)
        del arg761_1
        # Topologically Sorted Source Nodes: [sp_752, sp_753], Original ATen: [aten.add, aten.convolution]
        buf538 = extern_kernels.convolution(buf536, buf537, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (8, 104, 14, 14), (20384, 1, 1456, 104))
        del buf536
        del buf537
        buf539 = reinterpret_tensor(buf541, (8, 104, 14, 14), (81536, 196, 14, 1), 40768)  # alias
        # Topologically Sorted Source Nodes: [sp_754, sp_755], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf538, arg762_1, arg763_1, arg764_1, arg765_1, buf539, 832, 196, grid=grid(832, 196), stream=stream0)
        del arg762_1
        del arg763_1
        del arg764_1
        del arg765_1
        del buf538
        buf540 = reinterpret_tensor(buf541, (8, 104, 14, 14), (81536, 196, 14, 1), 61152)  # alias
        # Topologically Sorted Source Nodes: [out_499], Original ATen: [aten.cat]
        triton_poi_fused_cat_35.run(buf528, buf540, 832, 196, grid=grid(832, 196), stream=stream0)
        buf542 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [out_500], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_30.run(buf541, buf542, 3328, 196, grid=grid(3328, 196), stream=stream0)
        del buf531
        del buf535
        del buf539
        del buf540
        del buf541
        # Topologically Sorted Source Nodes: [out_500], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf542, arg766_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg766_1
        del buf542
        buf544 = buf526; del buf526  # reuse
        # Topologically Sorted Source Nodes: [out_501, out_502, out_503], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_36.run(buf544, buf543, arg767_1, arg768_1, arg769_1, arg770_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg767_1
        del arg768_1
        del arg769_1
        del arg770_1
        del buf543
        # Topologically Sorted Source Nodes: [out_504], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf544, arg771_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (8, 832, 14, 14), (163072, 1, 11648, 832))
        del arg771_1
        buf546 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [out_505, out_506], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf546, arg772_1, arg773_1, arg774_1, arg775_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg772_1
        del arg773_1
        del arg774_1
        del arg775_1
        buf547 = empty_strided_cuda((208, 208, 3, 3), (1872, 1, 624, 208), torch.float32)
        # Topologically Sorted Source Nodes: [sp_757], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg776_1, buf547, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg776_1
        # Topologically Sorted Source Nodes: [sp_757], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(reinterpret_tensor(buf546, (8, 208, 14, 14), (163072, 1, 11648, 832), 0), buf547, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (8, 208, 7, 7), (10192, 1, 1456, 208))
        buf549 = buf547; del buf547  # reuse
        # Topologically Sorted Source Nodes: [sp_761], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg781_1, buf549, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg781_1
        # Topologically Sorted Source Nodes: [sp_761], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(reinterpret_tensor(buf546, (8, 208, 14, 14), (163072, 1, 11648, 832), 208), buf549, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (8, 208, 7, 7), (10192, 1, 1456, 208))
        buf551 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [sp_765], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg786_1, buf551, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg786_1
        # Topologically Sorted Source Nodes: [sp_765], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(reinterpret_tensor(buf546, (8, 208, 14, 14), (163072, 1, 11648, 832), 416), buf551, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (8, 208, 7, 7), (10192, 1, 1456, 208))
        buf557 = reinterpret_tensor(buf124, (8, 832, 7, 7), (40768, 49, 7, 1), 0); del buf124  # reuse
        buf553 = reinterpret_tensor(buf557, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_7], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_39.run(buf546, buf553, 392, 208, grid=grid(392, 208), stream=stream0)
        del buf546
        buf554 = reinterpret_tensor(buf557, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_758, sp_759], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf548, arg777_1, arg778_1, arg779_1, arg780_1, buf554, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg777_1
        del arg778_1
        del arg779_1
        del arg780_1
        del buf548
        buf555 = reinterpret_tensor(buf557, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        # Topologically Sorted Source Nodes: [sp_762, sp_763], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf550, arg782_1, arg783_1, arg784_1, arg785_1, buf555, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg782_1
        del arg783_1
        del arg784_1
        del arg785_1
        del buf550
        buf556 = reinterpret_tensor(buf557, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_766, sp_767], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf552, arg787_1, arg788_1, arg789_1, arg790_1, buf556, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg787_1
        del arg788_1
        del arg789_1
        del arg790_1
        del buf552
        buf558 = reinterpret_tensor(buf122, (8, 832, 7, 7), (40768, 1, 5824, 832), 0); del buf122  # reuse
        # Topologically Sorted Source Nodes: [out_508], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf557, buf558, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf553
        del buf554
        del buf555
        del buf556
        del buf557
        # Topologically Sorted Source Nodes: [out_508], Original ATen: [aten.convolution]
        buf559 = extern_kernels.convolution(buf558, arg791_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf559, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg791_1
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf544, arg796_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg796_1
        del buf544
        buf561 = buf559; del buf559  # reuse
        buf562 = empty_strided_cuda((8, 2048, 7, 7), (100352, 1, 14336, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [out_509, input_16, out_510, out_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_42.run(buf561, arg792_1, arg793_1, arg794_1, arg795_1, buf560, arg797_1, arg798_1, arg799_1, arg800_1, buf562, 802816, grid=grid(802816), stream=stream0)
        del arg792_1
        del arg793_1
        del arg794_1
        del arg795_1
        del arg797_1
        del arg798_1
        del arg799_1
        del arg800_1
        del buf560
        del buf561
        # Topologically Sorted Source Nodes: [out_511, out_512], Original ATen: [aten.relu, aten.convolution]
        buf563 = extern_kernels.convolution(buf562, arg801_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (8, 832, 7, 7), (40768, 1, 5824, 832))
        del arg801_1
        buf564 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [out_513, out_514], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf564, arg802_1, arg803_1, arg804_1, arg805_1, 326144, grid=grid(326144), stream=stream0)
        del arg802_1
        del arg803_1
        del arg804_1
        del arg805_1
        buf565 = buf551; del buf551  # reuse
        # Topologically Sorted Source Nodes: [sp_769], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg806_1, buf565, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg806_1
        # Topologically Sorted Source Nodes: [sp_769], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(reinterpret_tensor(buf564, (8, 208, 7, 7), (40768, 1, 5824, 832), 0), buf565, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (8, 208, 7, 7), (10192, 1, 1456, 208))
        buf577 = reinterpret_tensor(buf558, (8, 832, 7, 7), (40768, 49, 7, 1), 0); del buf558  # reuse
        buf567 = reinterpret_tensor(buf577, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_770, sp_771], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf566, arg807_1, arg808_1, arg809_1, arg810_1, buf567, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg807_1
        del arg808_1
        del arg809_1
        del arg810_1
        buf568 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [sp_772], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf567, buf564, buf568, 392, 208, grid=grid(392, 208), stream=stream0)
        buf569 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [sp_772, sp_773], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg811_1, buf569, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg811_1
        # Topologically Sorted Source Nodes: [sp_772, sp_773], Original ATen: [aten.add, aten.convolution]
        buf570 = extern_kernels.convolution(buf568, buf569, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (8, 208, 7, 7), (10192, 1, 1456, 208))
        del buf568
        buf571 = reinterpret_tensor(buf577, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        # Topologically Sorted Source Nodes: [sp_774, sp_775], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf570, arg812_1, arg813_1, arg814_1, arg815_1, buf571, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg812_1
        del arg813_1
        del arg814_1
        del arg815_1
        buf572 = buf570; del buf570  # reuse
        # Topologically Sorted Source Nodes: [sp_776], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf571, buf564, buf572, 392, 208, grid=grid(392, 208), stream=stream0)
        buf573 = buf569; del buf569  # reuse
        # Topologically Sorted Source Nodes: [sp_776, sp_777], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg816_1, buf573, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg816_1
        # Topologically Sorted Source Nodes: [sp_776, sp_777], Original ATen: [aten.add, aten.convolution]
        buf574 = extern_kernels.convolution(buf572, buf573, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 208, 7, 7), (10192, 1, 1456, 208))
        del buf572
        buf575 = reinterpret_tensor(buf577, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_778, sp_779], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf574, arg817_1, arg818_1, arg819_1, arg820_1, buf575, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg817_1
        del arg818_1
        del arg819_1
        del arg820_1
        del buf574
        buf576 = reinterpret_tensor(buf577, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Topologically Sorted Source Nodes: [out_515], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf564, buf576, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf578 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [out_516], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf577, buf578, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf567
        del buf571
        del buf575
        del buf576
        del buf577
        # Topologically Sorted Source Nodes: [out_516], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, arg821_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg821_1
        buf580 = buf562; del buf562  # reuse
        # Topologically Sorted Source Nodes: [out_517, out_518, out_519], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_47.run(buf580, buf579, arg822_1, arg823_1, arg824_1, arg825_1, 802816, grid=grid(802816), stream=stream0)
        del arg822_1
        del arg823_1
        del arg824_1
        del arg825_1
        del buf579
        # Topologically Sorted Source Nodes: [out_520], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, arg826_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (8, 832, 7, 7), (40768, 1, 5824, 832))
        del arg826_1
        buf582 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [out_521, out_522], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_43.run(buf582, arg827_1, arg828_1, arg829_1, arg830_1, 326144, grid=grid(326144), stream=stream0)
        del arg827_1
        del arg828_1
        del arg829_1
        del arg830_1
        buf583 = buf573; del buf573  # reuse
        # Topologically Sorted Source Nodes: [sp_781], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(arg831_1, buf583, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg831_1
        # Topologically Sorted Source Nodes: [sp_781], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(reinterpret_tensor(buf582, (8, 208, 7, 7), (40768, 1, 5824, 832), 0), buf583, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (8, 208, 7, 7), (10192, 1, 1456, 208))
        buf595 = reinterpret_tensor(buf578, (8, 832, 7, 7), (40768, 49, 7, 1), 0); del buf578  # reuse
        buf585 = reinterpret_tensor(buf595, (8, 208, 7, 7), (40768, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_782, sp_783], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf584, arg832_1, arg833_1, arg834_1, arg835_1, buf585, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg832_1
        del arg833_1
        del arg834_1
        del arg835_1
        buf586 = buf584; del buf584  # reuse
        # Topologically Sorted Source Nodes: [sp_784], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf585, buf582, buf586, 392, 208, grid=grid(392, 208), stream=stream0)
        buf587 = buf583; del buf583  # reuse
        # Topologically Sorted Source Nodes: [sp_784, sp_785], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg836_1, buf587, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg836_1
        # Topologically Sorted Source Nodes: [sp_784, sp_785], Original ATen: [aten.add, aten.convolution]
        buf588 = extern_kernels.convolution(buf586, buf587, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf588, (8, 208, 7, 7), (10192, 1, 1456, 208))
        del buf586
        buf589 = reinterpret_tensor(buf595, (8, 208, 7, 7), (40768, 49, 7, 1), 10192)  # alias
        # Topologically Sorted Source Nodes: [sp_786, sp_787], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf588, arg837_1, arg838_1, arg839_1, arg840_1, buf589, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg837_1
        del arg838_1
        del arg839_1
        del arg840_1
        buf590 = buf588; del buf588  # reuse
        # Topologically Sorted Source Nodes: [sp_788], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf589, buf582, buf590, 392, 208, grid=grid(392, 208), stream=stream0)
        buf591 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [sp_788, sp_789], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_38.run(arg841_1, buf591, 43264, 9, grid=grid(43264, 9), stream=stream0)
        del arg841_1
        # Topologically Sorted Source Nodes: [sp_788, sp_789], Original ATen: [aten.add, aten.convolution]
        buf592 = extern_kernels.convolution(buf590, buf591, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf592, (8, 208, 7, 7), (10192, 1, 1456, 208))
        del buf590
        del buf591
        buf593 = reinterpret_tensor(buf595, (8, 208, 7, 7), (40768, 49, 7, 1), 20384)  # alias
        # Topologically Sorted Source Nodes: [sp_790, sp_791], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf592, arg842_1, arg843_1, arg844_1, arg845_1, buf593, 1664, 49, grid=grid(1664, 49), stream=stream0)
        del arg842_1
        del arg843_1
        del arg844_1
        del arg845_1
        del buf592
        buf594 = reinterpret_tensor(buf595, (8, 208, 7, 7), (40768, 49, 7, 1), 30576)  # alias
        # Topologically Sorted Source Nodes: [out_523], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf582, buf594, 1664, 49, grid=grid(1664, 49), stream=stream0)
        buf596 = buf582; del buf582  # reuse
        # Topologically Sorted Source Nodes: [out_524], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf595, buf596, 6656, 49, grid=grid(6656, 49), stream=stream0)
        del buf585
        del buf589
        del buf593
        del buf594
        del buf595
        # Topologically Sorted Source Nodes: [out_524], Original ATen: [aten.convolution]
        buf597 = extern_kernels.convolution(buf596, arg846_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf597, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg846_1
        del buf596
        buf599 = empty_strided_cuda((8, 2048, 1, 1), (2048, 1, 16384, 16384), torch.float32)
        # Topologically Sorted Source Nodes: [out_525, out_526, out_527, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_48.run(buf597, arg847_1, arg848_1, arg849_1, arg850_1, buf580, buf599, 16384, 49, grid=grid(16384), stream=stream0)
        del arg847_1
        del arg848_1
        del arg849_1
        del arg850_1
        del buf580
        del buf597
        buf600 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg852_1, reinterpret_tensor(buf599, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg851_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf600)
        del arg851_1
        del arg852_1
        del buf599
    return (buf600, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((104, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((104, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((26, 26, 3, 3), (234, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((256, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((208, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((208, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((52, 52, 3, 3), (468, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, 208, 1, 1), (208, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((416, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((416, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg748_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg749_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg750_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg751_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg752_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg753_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg754_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg755_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg756_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg757_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg758_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg759_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg760_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg761_1 = rand_strided((104, 104, 3, 3), (936, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg762_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg763_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg764_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg765_1 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg766_1 = rand_strided((1024, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg767_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg768_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg769_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg770_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg771_1 = rand_strided((832, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg772_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg773_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg774_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg775_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg776_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg777_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg778_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg779_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg780_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg781_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg782_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg783_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg784_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg785_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg786_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg787_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg788_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg789_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg790_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg791_1 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg792_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg793_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg794_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg795_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg796_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg797_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg798_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg799_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg800_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg801_1 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg802_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg803_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg804_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg805_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg806_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg807_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg808_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg809_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg810_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg811_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg812_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg813_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg814_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg815_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg816_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg817_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg818_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg819_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg820_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg821_1 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg822_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg823_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg824_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg825_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg826_1 = rand_strided((832, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg827_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg828_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg829_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg830_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg831_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg832_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg833_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg834_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg835_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg836_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg837_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg838_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg839_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg840_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg841_1 = rand_strided((208, 208, 3, 3), (1872, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg842_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg843_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg844_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg845_1 = rand_strided((208, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg846_1 = rand_strided((2048, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg847_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg848_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg849_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg850_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg851_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg852_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2net101_26w_4s', benchmark_compiled_module)
