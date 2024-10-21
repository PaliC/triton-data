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
#   x_7 => convolution_149
# Graph fragment:
#   %convolution_149 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
#   x_7 => convolution_149
# Graph fragment:
#   %convolution_149 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
#   x_8 => add_387, mul_448, mul_449, sub_149
#   x_9 => relu_145
# Graph fragment:
#   %sub_149 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_149, %unsqueeze_1193), kwargs = {})
#   %mul_448 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_149, %unsqueeze_1195), kwargs = {})
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_448, %unsqueeze_1197), kwargs = {})
#   %add_387 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_449, %unsqueeze_1199), kwargs = {})
#   %relu_145 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_387,), kwargs = {})
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
#   x_8 => add_387, mul_448, mul_449, sub_149
#   x_9 => relu_145
# Graph fragment:
#   %sub_149 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_149, %unsqueeze_1193), kwargs = {})
#   %mul_448 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_149, %unsqueeze_1195), kwargs = {})
#   %mul_449 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_448, %unsqueeze_1197), kwargs = {})
#   %add_387 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_449, %unsqueeze_1199), kwargs = {})
#   %relu_145 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_387,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_145, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/mf/cmfwtaomgr2sejqbkjpry4epsmyq3cyyzxxwa2eokugwz43nntc2.py
# Topologically Sorted Source Nodes: [out_129, out_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_129 => add_389, mul_451, mul_452, sub_150
#   out_130 => relu_146
# Graph fragment:
#   %sub_150 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_150, %unsqueeze_1201), kwargs = {})
#   %mul_451 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_150, %unsqueeze_1203), kwargs = {})
#   %mul_452 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_451, %unsqueeze_1205), kwargs = {})
#   %add_389 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_452, %unsqueeze_1207), kwargs = {})
#   %relu_146 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%add_389,), kwargs = {})
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
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 112
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


# kernel path: /tmp/torchinductor_sahanp/bt/cbtkkawhjpudpsfs2lauckojswpcewhlude7oxdx6lfd34ceknt6.py
# Topologically Sorted Source Nodes: [sp_449], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_449 => convolution_151
# Graph fragment:
#   %convolution_151 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_1164, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (14*x2) + (126*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/jk/cjkqs3pj4ip66mpbyl4rymokr5uukuizejxppuql77k4cgb4m42i.py
# Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_4 => avg_pool2d_4
# Graph fragment:
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_1227, [3, 3], [1, 1], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_6 = async_compile.triton('triton_poi_fused_avg_pool2d_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp11 = tl.load(in_ptr0 + ((-6286) + x3 + (112*y5)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-6174) + x3 + (112*y5)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + y0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-6062) + x3 + (112*y5)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-14) + x3 + (112*y5)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (98 + x3 + (112*y5)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (210 + x3 + (112*y5)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + y1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (6258 + x3 + (112*y5)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (6370 + x3 + (112*y5)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (6482 + x3 + (112*y5)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-1)*y0) + ((-1)*y1) + (y0*y1) + (((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57)))*((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))) + ((-1)*y0*((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))) + ((-1)*y1*((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57)))) + ((57) * ((57) <= (2 + y0)) + (2 + y0) * ((2 + y0) < (57))) + ((57) * ((57) <= (2 + y1)) + (2 + y1) * ((2 + y1) < (57)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y6 + (3136*x3) + (351232*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3n/c3nrt5sodvd2ee6wm2efic54jbtigsuybaebukn4ca6obcrpkcpo.py
# Topologically Sorted Source Nodes: [sp_450, sp_451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_450 => add_391, mul_454, mul_455, sub_151
#   sp_451 => relu_147
# Graph fragment:
#   %sub_151 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_151, %unsqueeze_1209), kwargs = {})
#   %mul_454 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_151, %unsqueeze_1211), kwargs = {})
#   %mul_455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_454, %unsqueeze_1213), kwargs = {})
#   %add_391 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_455, %unsqueeze_1215), kwargs = {})
#   %relu_147 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_391,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (y0 + (14*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i3/ci363svoihq2otct3zucxrxukrpajanex2uvf56s4pbkj34jzbfr.py
# Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_132 => convolution_158
# Graph fragment:
#   %convolution_158 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_16, %arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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
    ynumel = 896
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (351232*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xx/cxx4x6yndyw5nnibgriubhzn46v6fabjfvtib4f32ouv5srfthxx.py
# Topologically Sorted Source Nodes: [out_133, input_10, out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_407, mul_478, mul_479, sub_159
#   out_133 => add_405, mul_475, mul_476, sub_158
#   out_134 => add_408
#   out_135 => relu_154
# Graph fragment:
#   %sub_158 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_158, %unsqueeze_1265), kwargs = {})
#   %mul_475 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_158, %unsqueeze_1267), kwargs = {})
#   %mul_476 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_475, %unsqueeze_1269), kwargs = {})
#   %add_405 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_476, %unsqueeze_1271), kwargs = {})
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_159, %unsqueeze_1273), kwargs = {})
#   %mul_478 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_479 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_478, %unsqueeze_1277), kwargs = {})
#   %add_407 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_479, %unsqueeze_1279), kwargs = {})
#   %add_408 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_405, %add_407), kwargs = {})
#   %relu_154 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_408,), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/cg/ccggtpasqwbgkxwrlyckivhkcryxdos2rvpcxoamsdoal6az33mx.py
# Topologically Sorted Source Nodes: [sp_480], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_480 => add_413
# Graph fragment:
#   %add_413 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_156, %getitem_1245), kwargs = {})
triton_poi_fused_add_10 = async_compile.triton('triton_poi_fused_add_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_10(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (14 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vd/cvdfjmo3w4cpkhomv3iiy5ogwyhojk76emzacizlhp5dfakqjpsb.py
# Topologically Sorted Source Nodes: [sp_484], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_484 => add_416
# Graph fragment:
#   %add_416 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_157, %getitem_1254), kwargs = {})
triton_poi_fused_add_11 = async_compile.triton('triton_poi_fused_add_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_11(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (28 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mb/cmbk5kksw4puewwk67cor7ydiltywd726cg25klm6x6nu4pwmw3d.py
# Topologically Sorted Source Nodes: [sp_488], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_488 => add_419
# Graph fragment:
#   %add_419 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_158, %getitem_1263), kwargs = {})
triton_poi_fused_add_12 = async_compile.triton('triton_poi_fused_add_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_12(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (42 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bo/cbofkusjs2us4c3lb2grikjhzhrdedsjn63jnglxibgdbii55rcb.py
# Topologically Sorted Source Nodes: [sp_492], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_492 => add_422
# Graph fragment:
#   %add_422 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_159, %getitem_1272), kwargs = {})
triton_poi_fused_add_13 = async_compile.triton('triton_poi_fused_add_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_13(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (56 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/66/c663fngx2s4hl57ic6bq7moy4r67pqstr5747vib72zgpltxomg6.py
# Topologically Sorted Source Nodes: [sp_496], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_496 => add_425
# Graph fragment:
#   %add_425 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_160, %getitem_1281), kwargs = {})
triton_poi_fused_add_14 = async_compile.triton('triton_poi_fused_add_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (70 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/lt/cltgswiunem77pud2zvoafptrmhesq772b2r4xdvkkudv4vb7fec.py
# Topologically Sorted Source Nodes: [sp_500], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_500 => add_428
# Graph fragment:
#   %add_428 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_161, %getitem_1290), kwargs = {})
triton_poi_fused_add_15 = async_compile.triton('triton_poi_fused_add_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_15(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
    xnumel = 14
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
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (84 + x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (14*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ik/ciklhgfingva7tvdpwom7z4obfsglqpauip2fcwn5e5xnrnbgccb.py
# Topologically Sorted Source Nodes: [out_139], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_139 => cat_17
# Graph fragment:
#   %cat_17 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_156, %relu_157, %relu_158, %relu_159, %relu_160, %relu_161, %relu_162, %getitem_1299], 1), kwargs = {})
triton_poi_fused_cat_16 = async_compile.triton('triton_poi_fused_cat_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_16(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 112
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 14
    y1 = (yindex // 14)
    tmp0 = tl.load(in_ptr0 + (98 + y0 + (112*x2) + (351232*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (3136*y0) + (351232*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3g/c3g4brsliuprz23nwqhx6ja2yknlo5oamgf4h7gfpiyr3w5safzb.py
# Topologically Sorted Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_141 => add_432, mul_505, mul_506, sub_168
#   out_142 => add_433
#   out_143 => relu_163
# Graph fragment:
#   %sub_168 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_168, %unsqueeze_1345), kwargs = {})
#   %mul_505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_168, %unsqueeze_1347), kwargs = {})
#   %mul_506 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_505, %unsqueeze_1349), kwargs = {})
#   %add_432 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_506, %unsqueeze_1351), kwargs = {})
#   %add_433 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_432, %relu_154), kwargs = {})
#   %relu_163 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_433,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ro/crorul6nrdq53v5qhpjljp5ykvesepltwx76b6pbammtxgqsaajx.py
# Topologically Sorted Source Nodes: [out_149, out_150, out_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_149 => add_457, mul_532, mul_533, sub_177
#   out_150 => add_458
#   out_151 => relu_172
# Graph fragment:
#   %sub_177 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_177, %unsqueeze_1417), kwargs = {})
#   %mul_532 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_177, %unsqueeze_1419), kwargs = {})
#   %mul_533 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_532, %unsqueeze_1421), kwargs = {})
#   %add_457 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_533, %unsqueeze_1423), kwargs = {})
#   %add_458 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_457, %relu_163), kwargs = {})
#   %relu_172 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_458,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/gs/cgswjfndx3cexqld2dzba34l4ibb2t3cnt6dhdq5mv3javbk6x3p.py
# Topologically Sorted Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_153 => add_460, mul_535, mul_536, sub_178
#   out_154 => relu_173
# Graph fragment:
#   %sub_178 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_178, %unsqueeze_1425), kwargs = {})
#   %mul_535 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_178, %unsqueeze_1427), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_535, %unsqueeze_1429), kwargs = {})
#   %add_460 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_536, %unsqueeze_1431), kwargs = {})
#   %relu_173 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%add_460,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 224
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


# kernel path: /tmp/torchinductor_sahanp/rf/crf6pu5bpprt5ktdnh7tpk453uarkvuhv63lc64tvxxnhvqhfct5.py
# Topologically Sorted Source Nodes: [sp_533], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_533 => convolution_179
# Graph fragment:
#   %convolution_179 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_1380, %arg151_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (28*x2) + (252*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zv/czvfwxqaok3a7ttwtssiiu7hzuxrhxszijrl6hmrg4vgxv2qfxck.py
# Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_5 => avg_pool2d_5
# Graph fragment:
#   %avg_pool2d_5 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_1443, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_21 = async_compile.triton('triton_poi_fused_avg_pool2d_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_21(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp11 = tl.load(in_ptr0 + ((-12572) + x3 + (448*y0) + (25088*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-12348) + x3 + (448*y0) + (25088*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-12124) + x3 + (448*y0) + (25088*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-28) + x3 + (448*y0) + (25088*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (196 + x3 + (448*y0) + (25088*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (420 + x3 + (448*y0) + (25088*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (12516 + x3 + (448*y0) + (25088*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (12740 + x3 + (448*y0) + (25088*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (12964 + x3 + (448*y0) + (25088*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57)))*((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))) + ((-2)*y0*((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))) + ((-2)*y1*((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57)))) + (4*y0*y1) + ((57) * ((57) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (57))) + ((57) * ((57) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (57)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (784*x3) + (175616*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qg/cqgmjrlk5oa4ohm63fzfq3xurz5rlsv5jqnvbsup5ofuzirf4u4k.py
# Topologically Sorted Source Nodes: [sp_534, sp_535], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_534 => add_462, mul_538, mul_539, sub_179
#   sp_535 => relu_174
# Graph fragment:
#   %sub_179 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_179, %unsqueeze_1433), kwargs = {})
#   %mul_538 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_179, %unsqueeze_1435), kwargs = {})
#   %mul_539 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_538, %unsqueeze_1437), kwargs = {})
#   %add_462 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_539, %unsqueeze_1439), kwargs = {})
#   %relu_174 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_462,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (y0 + (28*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xk/cxkrq3bapkpzgnoe7qdyyqbt4ybmd6fwe2gtdmf4rquefl6qp3zk.py
# Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_156 => convolution_186
# Graph fragment:
#   %convolution_186 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_19, %arg186_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_23 = async_compile.triton('triton_poi_fused_convolution_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_23(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1792
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 224
    y1 = (yindex // 224)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (224*x2) + (175616*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vn/cvngl66ihhujigcgkvv4hdflapmdizwt7wlvxsqmn2hks2n4frjf.py
# Topologically Sorted Source Nodes: [out_157, input_12, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_12 => add_478, mul_562, mul_563, sub_187
#   out_157 => add_476, mul_559, mul_560, sub_186
#   out_158 => add_479
#   out_159 => relu_181
# Graph fragment:
#   %sub_186 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_186, %unsqueeze_1489), kwargs = {})
#   %mul_559 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_186, %unsqueeze_1491), kwargs = {})
#   %mul_560 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_559, %unsqueeze_1493), kwargs = {})
#   %add_476 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_560, %unsqueeze_1495), kwargs = {})
#   %sub_187 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_187, %unsqueeze_1497), kwargs = {})
#   %mul_562 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_187, %unsqueeze_1499), kwargs = {})
#   %mul_563 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_562, %unsqueeze_1501), kwargs = {})
#   %add_478 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_563, %unsqueeze_1503), kwargs = {})
#   %add_479 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_476, %add_478), kwargs = {})
#   %relu_181 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_479,), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/kb/ckbvg7wgs3fdo2plk6b2cpq77ylp5yomjxs677fwfkl42ghxcdci.py
# Topologically Sorted Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_161 => add_481, mul_565, mul_566, sub_188
#   out_162 => relu_182
# Graph fragment:
#   %sub_188 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_188, %unsqueeze_1505), kwargs = {})
#   %mul_565 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_188, %unsqueeze_1507), kwargs = {})
#   %mul_566 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_565, %unsqueeze_1509), kwargs = {})
#   %add_481 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_566, %unsqueeze_1511), kwargs = {})
#   %relu_182 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%add_481,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 224
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


# kernel path: /tmp/torchinductor_sahanp/eu/ceup7ertwrgkmaj2fudi6463rik4xtflrovjpbohtqqbv2gqdrbm.py
# Topologically Sorted Source Nodes: [sp_564], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_564 => add_484
# Graph fragment:
#   %add_484 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_183, %getitem_1461), kwargs = {})
triton_poi_fused_add_26 = async_compile.triton('triton_poi_fused_add_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_26(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (28 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/rb/crb6cvucqx34llbiq6lxhrrlfhfi2rel2ujra5aagmtsnjd7m4dq.py
# Topologically Sorted Source Nodes: [sp_568], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_568 => add_487
# Graph fragment:
#   %add_487 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_184, %getitem_1470), kwargs = {})
triton_poi_fused_add_27 = async_compile.triton('triton_poi_fused_add_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_27(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (56 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fo/cfo4wlh7ndvykhx5qqtx5tw27oauhp4fmf5tf2ygioblc63pms34.py
# Topologically Sorted Source Nodes: [sp_572], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_572 => add_490
# Graph fragment:
#   %add_490 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_185, %getitem_1479), kwargs = {})
triton_poi_fused_add_28 = async_compile.triton('triton_poi_fused_add_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_28(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (84 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/o7/co7wihi5woizjw4eoj25b7wfl2ru2bcjfdjpveoufnswazcwt3pu.py
# Topologically Sorted Source Nodes: [sp_576], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_576 => add_493
# Graph fragment:
#   %add_493 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_186, %getitem_1488), kwargs = {})
triton_poi_fused_add_29 = async_compile.triton('triton_poi_fused_add_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_29(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/fc/cfcvz2bgwhscx7qzsoft2d4mvfhy3jwopucxmrjtmc7453jqv5z2.py
# Topologically Sorted Source Nodes: [sp_580], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_580 => add_496
# Graph fragment:
#   %add_496 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_187, %getitem_1497), kwargs = {})
triton_poi_fused_add_30 = async_compile.triton('triton_poi_fused_add_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_30(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (140 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h4/ch4wimdyrma6jqxwjhi5gpu2zo2pyayixgutemhp52zh3hjmldjn.py
# Topologically Sorted Source Nodes: [sp_584], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_584 => add_499
# Graph fragment:
#   %add_499 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_188, %getitem_1506), kwargs = {})
triton_poi_fused_add_31 = async_compile.triton('triton_poi_fused_add_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[8192, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_31(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
    xnumel = 28
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
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (168 + x2 + (224*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (28*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mn/cmnpd77t2ttwpvcx3eylnq3j62grsy4vks2gwq3ehjhsahswidih.py
# Topologically Sorted Source Nodes: [out_163], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_163 => cat_20
# Graph fragment:
#   %cat_20 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_183, %relu_184, %relu_185, %relu_186, %relu_187, %relu_188, %relu_189, %getitem_1515], 1), kwargs = {})
triton_poi_fused_cat_32 = async_compile.triton('triton_poi_fused_cat_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_32(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 28
    y1 = (yindex // 28)
    tmp0 = tl.load(in_ptr0 + (196 + y0 + (224*x2) + (175616*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (784*y0) + (175616*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/y6/cy6lm4totgvgrkqo34ook32m23vu6mtvormzqaynvagnoxpndwrz.py
# Topologically Sorted Source Nodes: [out_165, out_166, out_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_165 => add_503, mul_589, mul_590, sub_196
#   out_166 => add_504
#   out_167 => relu_190
# Graph fragment:
#   %sub_196 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_196, %unsqueeze_1569), kwargs = {})
#   %mul_589 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_196, %unsqueeze_1571), kwargs = {})
#   %mul_590 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_589, %unsqueeze_1573), kwargs = {})
#   %add_503 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_590, %unsqueeze_1575), kwargs = {})
#   %add_504 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_503, %relu_181), kwargs = {})
#   %relu_190 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_504,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/iz/cizyxzxwlyx2wwrtu7rn44grprwktg2odnd2o7gulnpxqf6633yw.py
# Topologically Sorted Source Nodes: [out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_185 => add_556, mul_646, mul_647, sub_215
#   out_186 => relu_209
# Graph fragment:
#   %sub_215 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_215, %unsqueeze_1721), kwargs = {})
#   %mul_646 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_215, %unsqueeze_1723), kwargs = {})
#   %mul_647 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_646, %unsqueeze_1725), kwargs = {})
#   %add_556 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_647, %unsqueeze_1727), kwargs = {})
#   %relu_209 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%add_556,), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 448
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


# kernel path: /tmp/torchinductor_sahanp/6j/c6jsh3yyvpwezwlzzb5zkqcpd6defu3i7327jnhomp2wuqaztod3.py
# Topologically Sorted Source Nodes: [sp_645], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_645 => convolution_216
# Graph fragment:
#   %convolution_216 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_1668, %arg336_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_35 = async_compile.triton('triton_poi_fused_convolution_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (56*x2) + (504*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ej/cejt6ps4dapniw6frzypxjqtztphgmnc7ke5wrsnog424q3mbzjo.py
# Topologically Sorted Source Nodes: [avg_pool2d_6], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_6 => avg_pool2d_6
# Graph fragment:
#   %avg_pool2d_6 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_1731, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_36 = async_compile.triton('triton_poi_fused_avg_pool2d_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_36(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp11 = tl.load(in_ptr0 + ((-12600) + x3 + (896*y0) + (25088*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-12152) + x3 + (896*y0) + (25088*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-11704) + x3 + (896*y0) + (25088*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-56) + x3 + (896*y0) + (25088*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (392 + x3 + (896*y0) + (25088*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (840 + x3 + (896*y0) + (25088*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (12488 + x3 + (896*y0) + (25088*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (12936 + x3 + (896*y0) + (25088*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (13384 + x3 + (896*y0) + (25088*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29)))*((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))) + ((-2)*y0*((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))) + ((-2)*y1*((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29)))) + (4*y0*y1) + ((29) * ((29) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (29))) + ((29) * ((29) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (29)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (196*x3) + (87808*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/h5/ch56pxlfqpa5iidicfugv7irymej4li3vvwtb2jho57egrprqbi7.py
# Topologically Sorted Source Nodes: [sp_646, sp_647], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_646 => add_558, mul_649, mul_650, sub_216
#   sp_647 => relu_210
# Graph fragment:
#   %sub_216 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_216, %unsqueeze_1729), kwargs = {})
#   %mul_649 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_216, %unsqueeze_1731), kwargs = {})
#   %mul_650 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_649, %unsqueeze_1733), kwargs = {})
#   %add_558 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_650, %unsqueeze_1735), kwargs = {})
#   %relu_210 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_558,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (y0 + (56*x2) + (10976*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wg/cwgkjlgjzwwsptptmfkce64uppuzingje3dmnmi4w72dsn6wtej2.py
# Topologically Sorted Source Nodes: [out_188], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_188 => convolution_223
# Graph fragment:
#   %convolution_223 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_23, %arg371_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3584
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 448
    y1 = (yindex // 448)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (448*x2) + (87808*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bx/cbxydezqjyjw372u6pdeo2eaad6l3eg6rf3bnkxr44zghukffl6g.py
# Topologically Sorted Source Nodes: [out_189, input_14, out_190, out_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_14 => add_574, mul_673, mul_674, sub_224
#   out_189 => add_572, mul_670, mul_671, sub_223
#   out_190 => add_575
#   out_191 => relu_217
# Graph fragment:
#   %sub_223 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_223, %unsqueeze_1785), kwargs = {})
#   %mul_670 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_223, %unsqueeze_1787), kwargs = {})
#   %mul_671 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_670, %unsqueeze_1789), kwargs = {})
#   %add_572 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_671, %unsqueeze_1791), kwargs = {})
#   %sub_224 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_224, %unsqueeze_1793), kwargs = {})
#   %mul_673 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_224, %unsqueeze_1795), kwargs = {})
#   %mul_674 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_673, %unsqueeze_1797), kwargs = {})
#   %add_574 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_674, %unsqueeze_1799), kwargs = {})
#   %add_575 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_572, %add_574), kwargs = {})
#   %relu_217 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_575,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/zl/czldus63prqn66vi2slt74bilo2u6wzskhitf3cwztwexoylsamv.py
# Topologically Sorted Source Nodes: [out_193, out_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_193 => add_577, mul_676, mul_677, sub_225
#   out_194 => relu_218
# Graph fragment:
#   %sub_225 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_225, %unsqueeze_1801), kwargs = {})
#   %mul_676 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_225, %unsqueeze_1803), kwargs = {})
#   %mul_677 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_676, %unsqueeze_1805), kwargs = {})
#   %add_577 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_677, %unsqueeze_1807), kwargs = {})
#   %relu_218 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%add_577,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 448
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


# kernel path: /tmp/torchinductor_sahanp/en/cenoeabvqwc7w4hjor5dkumqr7mtx2gsnyriwssfxbyas245tfnh.py
# Topologically Sorted Source Nodes: [sp_676], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_676 => add_580
# Graph fragment:
#   %add_580 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_219, %getitem_1749), kwargs = {})
triton_poi_fused_add_41 = async_compile.triton('triton_poi_fused_add_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_41(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (56 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/i3/ci3yawdxkswknljwo6d32o6oej4jshxttt6booxwdzing7ytolmk.py
# Topologically Sorted Source Nodes: [sp_680], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_680 => add_583
# Graph fragment:
#   %add_583 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_220, %getitem_1758), kwargs = {})
triton_poi_fused_add_42 = async_compile.triton('triton_poi_fused_add_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_42(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/zs/czsgvfpwgpfq523mds7s2dhcmyob2eub7zhdpyjqt5ajphldoqz2.py
# Topologically Sorted Source Nodes: [sp_684], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_684 => add_586
# Graph fragment:
#   %add_586 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_221, %getitem_1767), kwargs = {})
triton_poi_fused_add_43 = async_compile.triton('triton_poi_fused_add_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_43(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (168 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/4a/c4aapf72gnlgbd3qspavbmfsrb55yysp6wdif2yjq4ticev5dijk.py
# Topologically Sorted Source Nodes: [sp_688], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_688 => add_589
# Graph fragment:
#   %add_589 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_222, %getitem_1776), kwargs = {})
triton_poi_fused_add_44 = async_compile.triton('triton_poi_fused_add_44', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_44(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (224 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/46/c46mu4heq3a3akw574czunskpqz6gzzqplpldsfis4eypastdbvf.py
# Topologically Sorted Source Nodes: [sp_692], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_692 => add_592
# Graph fragment:
#   %add_592 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_223, %getitem_1785), kwargs = {})
triton_poi_fused_add_45 = async_compile.triton('triton_poi_fused_add_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_45(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (280 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kb/ckbbvpwcw3fmwgj6zdmxefwrc4qgfxqkc5xrxekviacjkolpguld.py
# Topologically Sorted Source Nodes: [sp_696], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_696 => add_595
# Graph fragment:
#   %add_595 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_224, %getitem_1794), kwargs = {})
triton_poi_fused_add_46 = async_compile.triton('triton_poi_fused_add_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_46(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 56
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (336 + x2 + (448*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (56*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3g/c3gw3purj2ckk3ea2dqiftdfitsy6hxlxnccqtnycse5ciu4djei.py
# Topologically Sorted Source Nodes: [out_195], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_195 => cat_24
# Graph fragment:
#   %cat_24 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_219, %relu_220, %relu_221, %relu_222, %relu_223, %relu_224, %relu_225, %getitem_1803], 1), kwargs = {})
triton_poi_fused_cat_47 = async_compile.triton('triton_poi_fused_cat_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_47(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 56
    y1 = (yindex // 56)
    tmp0 = tl.load(in_ptr0 + (392 + y0 + (448*x2) + (87808*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (196*y0) + (87808*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/vs/cvsfgskszzdy5oiqhr7nrk6yh4pos3wgtmg6mh5qk76cxhsrdhmm.py
# Topologically Sorted Source Nodes: [out_197, out_198, out_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_197 => add_599, mul_700, mul_701, sub_233
#   out_198 => add_600
#   out_199 => relu_226
# Graph fragment:
#   %sub_233 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_233, %unsqueeze_1865), kwargs = {})
#   %mul_700 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_233, %unsqueeze_1867), kwargs = {})
#   %mul_701 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_700, %unsqueeze_1869), kwargs = {})
#   %add_599 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_701, %unsqueeze_1871), kwargs = {})
#   %add_600 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_599, %relu_217), kwargs = {})
#   %relu_226 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_600,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/qd/cqddiwkwpbeydhcchgdq36oykr4dpftkybxyvnrvjv6ehzmzdzug.py
# Topologically Sorted Source Nodes: [out_233, out_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_233 => add_702, mul_811, mul_812, sub_270
#   out_234 => relu_263
# Graph fragment:
#   %sub_270 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_270, %unsqueeze_2161), kwargs = {})
#   %mul_811 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_270, %unsqueeze_2163), kwargs = {})
#   %mul_812 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_811, %unsqueeze_2165), kwargs = {})
#   %add_702 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_812, %unsqueeze_2167), kwargs = {})
#   %relu_263 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%add_702,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 896
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


# kernel path: /tmp/torchinductor_sahanp/a5/ca54vchjsgo5mgzbupgjezpfvl4mrnvre6mdzpukr2sdtrxs2qtk.py
# Topologically Sorted Source Nodes: [sp_813], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   sp_813 => convolution_271
# Graph fragment:
#   %convolution_271 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem_2100, %arg611_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_50 = async_compile.triton('triton_poi_fused_convolution_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_50(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (112*x2) + (1008*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/mh/cmhznbtgeqdv7rdf4qsg3a3l55jqxkmgimtzgtn3x5xxz6eam3f3.py
# Topologically Sorted Source Nodes: [avg_pool2d_7], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   avg_pool2d_7 => avg_pool2d_7
# Graph fragment:
#   %avg_pool2d_7 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%getitem_2163, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_51 = async_compile.triton('triton_poi_fused_avg_pool2d_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_51(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp11 = tl.load(in_ptr0 + ((-12656) + x3 + (1792*y0) + (25088*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*y0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-11760) + x3 + (1792*y0) + (25088*y4)), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + (2*y0)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-10864) + x3 + (1792*y0) + (25088*y4)), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*y1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-112) + x3 + (1792*y0) + (25088*y4)), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (784 + x3 + (1792*y0) + (25088*y4)), tmp33 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1680 + x3 + (1792*y0) + (25088*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + (2*y1)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (12432 + x3 + (1792*y0) + (25088*y4)), tmp43 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (13328 + x3 + (1792*y0) + (25088*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (14224 + x3 + (1792*y0) + (25088*y4)), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*y0) + ((-2)*y1) + (((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15)))*((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))) + ((-2)*y0*((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))) + ((-2)*y1*((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15)))) + (4*y0*y1) + ((15) * ((15) <= (2 + (2*y0))) + (2 + (2*y0)) * ((2 + (2*y0)) < (15))) + ((15) * ((15) <= (2 + (2*y1))) + (2 + (2*y1)) * ((2 + (2*y1)) < (15)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (y5 + (49*x3) + (43904*y2)), tmp53, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/sj/csjukqvaarmwahkcycemcveqv5uuf3t6qx5v55f7zow4lndzshbd.py
# Topologically Sorted Source Nodes: [sp_814, sp_815], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   sp_814 => add_704, mul_814, mul_815, sub_271
#   sp_815 => relu_264
# Graph fragment:
#   %sub_271 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_271, %unsqueeze_2169), kwargs = {})
#   %mul_814 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_271, %unsqueeze_2171), kwargs = {})
#   %mul_815 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_814, %unsqueeze_2173), kwargs = {})
#   %add_704 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_815, %unsqueeze_2175), kwargs = {})
#   %relu_264 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_704,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (y0 + (112*x2) + (5488*y1)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/52/c52js6dxmnogqjud5rkmunxlxrtzhi4neqkcfwpdztfmazpz4nnq.py
# Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_236 => convolution_278
# Graph fragment:
#   %convolution_278 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg646_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_53 = async_compile.triton('triton_poi_fused_convolution_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_53(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7168
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 896
    y1 = (yindex // 896)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (896*x2) + (43904*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/3q/c3qbuztu522bi3kq2rkntne7ovlnzqri47rnxs32i6hhcntvzalm.py
# Topologically Sorted Source Nodes: [out_237, input_16, out_238, out_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_720, mul_838, mul_839, sub_279
#   out_237 => add_718, mul_835, mul_836, sub_278
#   out_238 => add_721
#   out_239 => relu_271
# Graph fragment:
#   %sub_278 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_278, %unsqueeze_2225), kwargs = {})
#   %mul_835 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_278, %unsqueeze_2227), kwargs = {})
#   %mul_836 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_835, %unsqueeze_2229), kwargs = {})
#   %add_718 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_836, %unsqueeze_2231), kwargs = {})
#   %sub_279 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_279, %unsqueeze_2233), kwargs = {})
#   %mul_838 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_279, %unsqueeze_2235), kwargs = {})
#   %mul_839 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_838, %unsqueeze_2237), kwargs = {})
#   %add_720 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_839, %unsqueeze_2239), kwargs = {})
#   %add_721 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_718, %add_720), kwargs = {})
#   %relu_271 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_721,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/kc/ckc6ht75k2n5hkerfxwxjlc2fixp3cnc6vqkdhjozfzxuorwjzyr.py
# Topologically Sorted Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_241 => add_723, mul_841, mul_842, sub_280
#   out_242 => relu_272
# Graph fragment:
#   %sub_280 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_280, %unsqueeze_2241), kwargs = {})
#   %mul_841 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_280, %unsqueeze_2243), kwargs = {})
#   %mul_842 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_841, %unsqueeze_2245), kwargs = {})
#   %add_723 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_842, %unsqueeze_2247), kwargs = {})
#   %relu_272 : [num_users=8] = call_function[target=torch.ops.aten.relu.default](args = (%add_723,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 896
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


# kernel path: /tmp/torchinductor_sahanp/bx/cbxswm4nffsaqpsmqiwaqkiktswshtrieuorspyxtxi5vwxcxzml.py
# Topologically Sorted Source Nodes: [sp_844], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_844 => add_726
# Graph fragment:
#   %add_726 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_273, %getitem_2181), kwargs = {})
triton_poi_fused_add_56 = async_compile.triton('triton_poi_fused_add_56', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_56(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (112 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/bh/cbhjj7vkgfvgxci4vvob43scf4bbew5aatofnxu7tlxh7mcmxsph.py
# Topologically Sorted Source Nodes: [sp_848], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_848 => add_729
# Graph fragment:
#   %add_729 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_274, %getitem_2190), kwargs = {})
triton_poi_fused_add_57 = async_compile.triton('triton_poi_fused_add_57', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_57(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (224 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/45/c45be2x6ayyczkb4dpxzacqndabm2n5haulbx3qa7xzghnlwejw3.py
# Topologically Sorted Source Nodes: [sp_852], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_852 => add_732
# Graph fragment:
#   %add_732 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_275, %getitem_2199), kwargs = {})
triton_poi_fused_add_58 = async_compile.triton('triton_poi_fused_add_58', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_58(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (336 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ka/ckamxkgc3yidsanwpmgdakwy4flndljg6ptdysuzmnfodt5mv7dn.py
# Topologically Sorted Source Nodes: [sp_856], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_856 => add_735
# Graph fragment:
#   %add_735 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_276, %getitem_2208), kwargs = {})
triton_poi_fused_add_59 = async_compile.triton('triton_poi_fused_add_59', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_59(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (448 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ht/chtzx7lercjkv3e7szzwun52fyvxnqqrtca55itqbzrdp5ntpvqs.py
# Topologically Sorted Source Nodes: [sp_860], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_860 => add_738
# Graph fragment:
#   %add_738 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_277, %getitem_2217), kwargs = {})
triton_poi_fused_add_60 = async_compile.triton('triton_poi_fused_add_60', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_60(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (560 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wx/cwx3ifkdqzrfgghqxfpye6gpmjsmivwwnjbynwztrdtzyi7hkspe.py
# Topologically Sorted Source Nodes: [sp_864], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   sp_864 => add_741
# Graph fragment:
#   %add_741 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_278, %getitem_2226), kwargs = {})
triton_poi_fused_add_61 = async_compile.triton('triton_poi_fused_add_61', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[512, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_61(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (672 + x2 + (896*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + (112*y3)), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/t5/ct5fsg3c6pvgkmucf6um25mcbvwz2eonlboy75y4bqaitctkvo3g.py
# Topologically Sorted Source Nodes: [out_243], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out_243 => cat_30
# Graph fragment:
#   %cat_30 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_273, %relu_274, %relu_275, %relu_276, %relu_277, %relu_278, %relu_279, %getitem_2235], 1), kwargs = {})
triton_poi_fused_cat_62 = async_compile.triton('triton_poi_fused_cat_62', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_62(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (784 + y0 + (896*x2) + (43904*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + (49*y0) + (43904*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/oo/cooyczr2bx2icxxxs4j5fqewyq57hqzr5hf6o4acmnqzsayulqid.py
# Topologically Sorted Source Nodes: [out_245, out_246, out_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_245 => add_745, mul_865, mul_866, sub_288
#   out_246 => add_746
#   out_247 => relu_280
# Graph fragment:
#   %sub_288 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_288, %unsqueeze_2305), kwargs = {})
#   %mul_865 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_288, %unsqueeze_2307), kwargs = {})
#   %mul_866 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_865, %unsqueeze_2309), kwargs = {})
#   %add_745 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_866, %unsqueeze_2311), kwargs = {})
#   %add_746 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_745, %relu_271), kwargs = {})
#   %relu_280 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_746,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ho/cho23h34huirl44zavp5527b64tdagoju4vtk7jsfotaki3cmz53.py
# Topologically Sorted Source Nodes: [out_253, out_254, out_255, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   out_253 => add_770, mul_892, mul_893, sub_297
#   out_254 => add_771
#   out_255 => relu_289
#   x_11 => mean_1
# Graph fragment:
#   %sub_297 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_297, %unsqueeze_2377), kwargs = {})
#   %mul_892 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_297, %unsqueeze_2379), kwargs = {})
#   %mul_893 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_892, %unsqueeze_2381), kwargs = {})
#   %add_770 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_893, %unsqueeze_2383), kwargs = {})
#   %add_771 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_770, %relu_280), kwargs = {})
#   %relu_289 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_771,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_289, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_64 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_64', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (112, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg7_1, (112, ), (1, ))
    assert_size_stride(arg8_1, (112, ), (1, ))
    assert_size_stride(arg9_1, (112, ), (1, ))
    assert_size_stride(arg10_1, (112, ), (1, ))
    assert_size_stride(arg11_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg12_1, (14, ), (1, ))
    assert_size_stride(arg13_1, (14, ), (1, ))
    assert_size_stride(arg14_1, (14, ), (1, ))
    assert_size_stride(arg15_1, (14, ), (1, ))
    assert_size_stride(arg16_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg17_1, (14, ), (1, ))
    assert_size_stride(arg18_1, (14, ), (1, ))
    assert_size_stride(arg19_1, (14, ), (1, ))
    assert_size_stride(arg20_1, (14, ), (1, ))
    assert_size_stride(arg21_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg22_1, (14, ), (1, ))
    assert_size_stride(arg23_1, (14, ), (1, ))
    assert_size_stride(arg24_1, (14, ), (1, ))
    assert_size_stride(arg25_1, (14, ), (1, ))
    assert_size_stride(arg26_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg27_1, (14, ), (1, ))
    assert_size_stride(arg28_1, (14, ), (1, ))
    assert_size_stride(arg29_1, (14, ), (1, ))
    assert_size_stride(arg30_1, (14, ), (1, ))
    assert_size_stride(arg31_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg32_1, (14, ), (1, ))
    assert_size_stride(arg33_1, (14, ), (1, ))
    assert_size_stride(arg34_1, (14, ), (1, ))
    assert_size_stride(arg35_1, (14, ), (1, ))
    assert_size_stride(arg36_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg37_1, (14, ), (1, ))
    assert_size_stride(arg38_1, (14, ), (1, ))
    assert_size_stride(arg39_1, (14, ), (1, ))
    assert_size_stride(arg40_1, (14, ), (1, ))
    assert_size_stride(arg41_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg42_1, (14, ), (1, ))
    assert_size_stride(arg43_1, (14, ), (1, ))
    assert_size_stride(arg44_1, (14, ), (1, ))
    assert_size_stride(arg45_1, (14, ), (1, ))
    assert_size_stride(arg46_1, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (112, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg57_1, (112, ), (1, ))
    assert_size_stride(arg58_1, (112, ), (1, ))
    assert_size_stride(arg59_1, (112, ), (1, ))
    assert_size_stride(arg60_1, (112, ), (1, ))
    assert_size_stride(arg61_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg62_1, (14, ), (1, ))
    assert_size_stride(arg63_1, (14, ), (1, ))
    assert_size_stride(arg64_1, (14, ), (1, ))
    assert_size_stride(arg65_1, (14, ), (1, ))
    assert_size_stride(arg66_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg67_1, (14, ), (1, ))
    assert_size_stride(arg68_1, (14, ), (1, ))
    assert_size_stride(arg69_1, (14, ), (1, ))
    assert_size_stride(arg70_1, (14, ), (1, ))
    assert_size_stride(arg71_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg72_1, (14, ), (1, ))
    assert_size_stride(arg73_1, (14, ), (1, ))
    assert_size_stride(arg74_1, (14, ), (1, ))
    assert_size_stride(arg75_1, (14, ), (1, ))
    assert_size_stride(arg76_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg77_1, (14, ), (1, ))
    assert_size_stride(arg78_1, (14, ), (1, ))
    assert_size_stride(arg79_1, (14, ), (1, ))
    assert_size_stride(arg80_1, (14, ), (1, ))
    assert_size_stride(arg81_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg82_1, (14, ), (1, ))
    assert_size_stride(arg83_1, (14, ), (1, ))
    assert_size_stride(arg84_1, (14, ), (1, ))
    assert_size_stride(arg85_1, (14, ), (1, ))
    assert_size_stride(arg86_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg87_1, (14, ), (1, ))
    assert_size_stride(arg88_1, (14, ), (1, ))
    assert_size_stride(arg89_1, (14, ), (1, ))
    assert_size_stride(arg90_1, (14, ), (1, ))
    assert_size_stride(arg91_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg92_1, (14, ), (1, ))
    assert_size_stride(arg93_1, (14, ), (1, ))
    assert_size_stride(arg94_1, (14, ), (1, ))
    assert_size_stride(arg95_1, (14, ), (1, ))
    assert_size_stride(arg96_1, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg97_1, (256, ), (1, ))
    assert_size_stride(arg98_1, (256, ), (1, ))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (112, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg102_1, (112, ), (1, ))
    assert_size_stride(arg103_1, (112, ), (1, ))
    assert_size_stride(arg104_1, (112, ), (1, ))
    assert_size_stride(arg105_1, (112, ), (1, ))
    assert_size_stride(arg106_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg107_1, (14, ), (1, ))
    assert_size_stride(arg108_1, (14, ), (1, ))
    assert_size_stride(arg109_1, (14, ), (1, ))
    assert_size_stride(arg110_1, (14, ), (1, ))
    assert_size_stride(arg111_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg112_1, (14, ), (1, ))
    assert_size_stride(arg113_1, (14, ), (1, ))
    assert_size_stride(arg114_1, (14, ), (1, ))
    assert_size_stride(arg115_1, (14, ), (1, ))
    assert_size_stride(arg116_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg117_1, (14, ), (1, ))
    assert_size_stride(arg118_1, (14, ), (1, ))
    assert_size_stride(arg119_1, (14, ), (1, ))
    assert_size_stride(arg120_1, (14, ), (1, ))
    assert_size_stride(arg121_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg122_1, (14, ), (1, ))
    assert_size_stride(arg123_1, (14, ), (1, ))
    assert_size_stride(arg124_1, (14, ), (1, ))
    assert_size_stride(arg125_1, (14, ), (1, ))
    assert_size_stride(arg126_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg127_1, (14, ), (1, ))
    assert_size_stride(arg128_1, (14, ), (1, ))
    assert_size_stride(arg129_1, (14, ), (1, ))
    assert_size_stride(arg130_1, (14, ), (1, ))
    assert_size_stride(arg131_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg132_1, (14, ), (1, ))
    assert_size_stride(arg133_1, (14, ), (1, ))
    assert_size_stride(arg134_1, (14, ), (1, ))
    assert_size_stride(arg135_1, (14, ), (1, ))
    assert_size_stride(arg136_1, (14, 14, 3, 3), (126, 9, 3, 1))
    assert_size_stride(arg137_1, (14, ), (1, ))
    assert_size_stride(arg138_1, (14, ), (1, ))
    assert_size_stride(arg139_1, (14, ), (1, ))
    assert_size_stride(arg140_1, (14, ), (1, ))
    assert_size_stride(arg141_1, (256, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (224, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg147_1, (224, ), (1, ))
    assert_size_stride(arg148_1, (224, ), (1, ))
    assert_size_stride(arg149_1, (224, ), (1, ))
    assert_size_stride(arg150_1, (224, ), (1, ))
    assert_size_stride(arg151_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg152_1, (28, ), (1, ))
    assert_size_stride(arg153_1, (28, ), (1, ))
    assert_size_stride(arg154_1, (28, ), (1, ))
    assert_size_stride(arg155_1, (28, ), (1, ))
    assert_size_stride(arg156_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg157_1, (28, ), (1, ))
    assert_size_stride(arg158_1, (28, ), (1, ))
    assert_size_stride(arg159_1, (28, ), (1, ))
    assert_size_stride(arg160_1, (28, ), (1, ))
    assert_size_stride(arg161_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg162_1, (28, ), (1, ))
    assert_size_stride(arg163_1, (28, ), (1, ))
    assert_size_stride(arg164_1, (28, ), (1, ))
    assert_size_stride(arg165_1, (28, ), (1, ))
    assert_size_stride(arg166_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg167_1, (28, ), (1, ))
    assert_size_stride(arg168_1, (28, ), (1, ))
    assert_size_stride(arg169_1, (28, ), (1, ))
    assert_size_stride(arg170_1, (28, ), (1, ))
    assert_size_stride(arg171_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg172_1, (28, ), (1, ))
    assert_size_stride(arg173_1, (28, ), (1, ))
    assert_size_stride(arg174_1, (28, ), (1, ))
    assert_size_stride(arg175_1, (28, ), (1, ))
    assert_size_stride(arg176_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg177_1, (28, ), (1, ))
    assert_size_stride(arg178_1, (28, ), (1, ))
    assert_size_stride(arg179_1, (28, ), (1, ))
    assert_size_stride(arg180_1, (28, ), (1, ))
    assert_size_stride(arg181_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg182_1, (28, ), (1, ))
    assert_size_stride(arg183_1, (28, ), (1, ))
    assert_size_stride(arg184_1, (28, ), (1, ))
    assert_size_stride(arg185_1, (28, ), (1, ))
    assert_size_stride(arg186_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg187_1, (512, ), (1, ))
    assert_size_stride(arg188_1, (512, ), (1, ))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg192_1, (512, ), (1, ))
    assert_size_stride(arg193_1, (512, ), (1, ))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg197_1, (224, ), (1, ))
    assert_size_stride(arg198_1, (224, ), (1, ))
    assert_size_stride(arg199_1, (224, ), (1, ))
    assert_size_stride(arg200_1, (224, ), (1, ))
    assert_size_stride(arg201_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg202_1, (28, ), (1, ))
    assert_size_stride(arg203_1, (28, ), (1, ))
    assert_size_stride(arg204_1, (28, ), (1, ))
    assert_size_stride(arg205_1, (28, ), (1, ))
    assert_size_stride(arg206_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg207_1, (28, ), (1, ))
    assert_size_stride(arg208_1, (28, ), (1, ))
    assert_size_stride(arg209_1, (28, ), (1, ))
    assert_size_stride(arg210_1, (28, ), (1, ))
    assert_size_stride(arg211_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg212_1, (28, ), (1, ))
    assert_size_stride(arg213_1, (28, ), (1, ))
    assert_size_stride(arg214_1, (28, ), (1, ))
    assert_size_stride(arg215_1, (28, ), (1, ))
    assert_size_stride(arg216_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg217_1, (28, ), (1, ))
    assert_size_stride(arg218_1, (28, ), (1, ))
    assert_size_stride(arg219_1, (28, ), (1, ))
    assert_size_stride(arg220_1, (28, ), (1, ))
    assert_size_stride(arg221_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg222_1, (28, ), (1, ))
    assert_size_stride(arg223_1, (28, ), (1, ))
    assert_size_stride(arg224_1, (28, ), (1, ))
    assert_size_stride(arg225_1, (28, ), (1, ))
    assert_size_stride(arg226_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg227_1, (28, ), (1, ))
    assert_size_stride(arg228_1, (28, ), (1, ))
    assert_size_stride(arg229_1, (28, ), (1, ))
    assert_size_stride(arg230_1, (28, ), (1, ))
    assert_size_stride(arg231_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg232_1, (28, ), (1, ))
    assert_size_stride(arg233_1, (28, ), (1, ))
    assert_size_stride(arg234_1, (28, ), (1, ))
    assert_size_stride(arg235_1, (28, ), (1, ))
    assert_size_stride(arg236_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (512, ), (1, ))
    assert_size_stride(arg241_1, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg242_1, (224, ), (1, ))
    assert_size_stride(arg243_1, (224, ), (1, ))
    assert_size_stride(arg244_1, (224, ), (1, ))
    assert_size_stride(arg245_1, (224, ), (1, ))
    assert_size_stride(arg246_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg247_1, (28, ), (1, ))
    assert_size_stride(arg248_1, (28, ), (1, ))
    assert_size_stride(arg249_1, (28, ), (1, ))
    assert_size_stride(arg250_1, (28, ), (1, ))
    assert_size_stride(arg251_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg252_1, (28, ), (1, ))
    assert_size_stride(arg253_1, (28, ), (1, ))
    assert_size_stride(arg254_1, (28, ), (1, ))
    assert_size_stride(arg255_1, (28, ), (1, ))
    assert_size_stride(arg256_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg257_1, (28, ), (1, ))
    assert_size_stride(arg258_1, (28, ), (1, ))
    assert_size_stride(arg259_1, (28, ), (1, ))
    assert_size_stride(arg260_1, (28, ), (1, ))
    assert_size_stride(arg261_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg262_1, (28, ), (1, ))
    assert_size_stride(arg263_1, (28, ), (1, ))
    assert_size_stride(arg264_1, (28, ), (1, ))
    assert_size_stride(arg265_1, (28, ), (1, ))
    assert_size_stride(arg266_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg267_1, (28, ), (1, ))
    assert_size_stride(arg268_1, (28, ), (1, ))
    assert_size_stride(arg269_1, (28, ), (1, ))
    assert_size_stride(arg270_1, (28, ), (1, ))
    assert_size_stride(arg271_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg272_1, (28, ), (1, ))
    assert_size_stride(arg273_1, (28, ), (1, ))
    assert_size_stride(arg274_1, (28, ), (1, ))
    assert_size_stride(arg275_1, (28, ), (1, ))
    assert_size_stride(arg276_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg277_1, (28, ), (1, ))
    assert_size_stride(arg278_1, (28, ), (1, ))
    assert_size_stride(arg279_1, (28, ), (1, ))
    assert_size_stride(arg280_1, (28, ), (1, ))
    assert_size_stride(arg281_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg282_1, (512, ), (1, ))
    assert_size_stride(arg283_1, (512, ), (1, ))
    assert_size_stride(arg284_1, (512, ), (1, ))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (224, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg287_1, (224, ), (1, ))
    assert_size_stride(arg288_1, (224, ), (1, ))
    assert_size_stride(arg289_1, (224, ), (1, ))
    assert_size_stride(arg290_1, (224, ), (1, ))
    assert_size_stride(arg291_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg292_1, (28, ), (1, ))
    assert_size_stride(arg293_1, (28, ), (1, ))
    assert_size_stride(arg294_1, (28, ), (1, ))
    assert_size_stride(arg295_1, (28, ), (1, ))
    assert_size_stride(arg296_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg297_1, (28, ), (1, ))
    assert_size_stride(arg298_1, (28, ), (1, ))
    assert_size_stride(arg299_1, (28, ), (1, ))
    assert_size_stride(arg300_1, (28, ), (1, ))
    assert_size_stride(arg301_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg302_1, (28, ), (1, ))
    assert_size_stride(arg303_1, (28, ), (1, ))
    assert_size_stride(arg304_1, (28, ), (1, ))
    assert_size_stride(arg305_1, (28, ), (1, ))
    assert_size_stride(arg306_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg307_1, (28, ), (1, ))
    assert_size_stride(arg308_1, (28, ), (1, ))
    assert_size_stride(arg309_1, (28, ), (1, ))
    assert_size_stride(arg310_1, (28, ), (1, ))
    assert_size_stride(arg311_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg312_1, (28, ), (1, ))
    assert_size_stride(arg313_1, (28, ), (1, ))
    assert_size_stride(arg314_1, (28, ), (1, ))
    assert_size_stride(arg315_1, (28, ), (1, ))
    assert_size_stride(arg316_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg317_1, (28, ), (1, ))
    assert_size_stride(arg318_1, (28, ), (1, ))
    assert_size_stride(arg319_1, (28, ), (1, ))
    assert_size_stride(arg320_1, (28, ), (1, ))
    assert_size_stride(arg321_1, (28, 28, 3, 3), (252, 9, 3, 1))
    assert_size_stride(arg322_1, (28, ), (1, ))
    assert_size_stride(arg323_1, (28, ), (1, ))
    assert_size_stride(arg324_1, (28, ), (1, ))
    assert_size_stride(arg325_1, (28, ), (1, ))
    assert_size_stride(arg326_1, (512, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg327_1, (512, ), (1, ))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (448, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg332_1, (448, ), (1, ))
    assert_size_stride(arg333_1, (448, ), (1, ))
    assert_size_stride(arg334_1, (448, ), (1, ))
    assert_size_stride(arg335_1, (448, ), (1, ))
    assert_size_stride(arg336_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg337_1, (56, ), (1, ))
    assert_size_stride(arg338_1, (56, ), (1, ))
    assert_size_stride(arg339_1, (56, ), (1, ))
    assert_size_stride(arg340_1, (56, ), (1, ))
    assert_size_stride(arg341_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg342_1, (56, ), (1, ))
    assert_size_stride(arg343_1, (56, ), (1, ))
    assert_size_stride(arg344_1, (56, ), (1, ))
    assert_size_stride(arg345_1, (56, ), (1, ))
    assert_size_stride(arg346_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg347_1, (56, ), (1, ))
    assert_size_stride(arg348_1, (56, ), (1, ))
    assert_size_stride(arg349_1, (56, ), (1, ))
    assert_size_stride(arg350_1, (56, ), (1, ))
    assert_size_stride(arg351_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg352_1, (56, ), (1, ))
    assert_size_stride(arg353_1, (56, ), (1, ))
    assert_size_stride(arg354_1, (56, ), (1, ))
    assert_size_stride(arg355_1, (56, ), (1, ))
    assert_size_stride(arg356_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg357_1, (56, ), (1, ))
    assert_size_stride(arg358_1, (56, ), (1, ))
    assert_size_stride(arg359_1, (56, ), (1, ))
    assert_size_stride(arg360_1, (56, ), (1, ))
    assert_size_stride(arg361_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg362_1, (56, ), (1, ))
    assert_size_stride(arg363_1, (56, ), (1, ))
    assert_size_stride(arg364_1, (56, ), (1, ))
    assert_size_stride(arg365_1, (56, ), (1, ))
    assert_size_stride(arg366_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg367_1, (56, ), (1, ))
    assert_size_stride(arg368_1, (56, ), (1, ))
    assert_size_stride(arg369_1, (56, ), (1, ))
    assert_size_stride(arg370_1, (56, ), (1, ))
    assert_size_stride(arg371_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (1024, ), (1, ))
    assert_size_stride(arg374_1, (1024, ), (1, ))
    assert_size_stride(arg375_1, (1024, ), (1, ))
    assert_size_stride(arg376_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg382_1, (448, ), (1, ))
    assert_size_stride(arg383_1, (448, ), (1, ))
    assert_size_stride(arg384_1, (448, ), (1, ))
    assert_size_stride(arg385_1, (448, ), (1, ))
    assert_size_stride(arg386_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg387_1, (56, ), (1, ))
    assert_size_stride(arg388_1, (56, ), (1, ))
    assert_size_stride(arg389_1, (56, ), (1, ))
    assert_size_stride(arg390_1, (56, ), (1, ))
    assert_size_stride(arg391_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg392_1, (56, ), (1, ))
    assert_size_stride(arg393_1, (56, ), (1, ))
    assert_size_stride(arg394_1, (56, ), (1, ))
    assert_size_stride(arg395_1, (56, ), (1, ))
    assert_size_stride(arg396_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg397_1, (56, ), (1, ))
    assert_size_stride(arg398_1, (56, ), (1, ))
    assert_size_stride(arg399_1, (56, ), (1, ))
    assert_size_stride(arg400_1, (56, ), (1, ))
    assert_size_stride(arg401_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg402_1, (56, ), (1, ))
    assert_size_stride(arg403_1, (56, ), (1, ))
    assert_size_stride(arg404_1, (56, ), (1, ))
    assert_size_stride(arg405_1, (56, ), (1, ))
    assert_size_stride(arg406_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg407_1, (56, ), (1, ))
    assert_size_stride(arg408_1, (56, ), (1, ))
    assert_size_stride(arg409_1, (56, ), (1, ))
    assert_size_stride(arg410_1, (56, ), (1, ))
    assert_size_stride(arg411_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg412_1, (56, ), (1, ))
    assert_size_stride(arg413_1, (56, ), (1, ))
    assert_size_stride(arg414_1, (56, ), (1, ))
    assert_size_stride(arg415_1, (56, ), (1, ))
    assert_size_stride(arg416_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg417_1, (56, ), (1, ))
    assert_size_stride(arg418_1, (56, ), (1, ))
    assert_size_stride(arg419_1, (56, ), (1, ))
    assert_size_stride(arg420_1, (56, ), (1, ))
    assert_size_stride(arg421_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (1024, ), (1, ))
    assert_size_stride(arg425_1, (1024, ), (1, ))
    assert_size_stride(arg426_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg427_1, (448, ), (1, ))
    assert_size_stride(arg428_1, (448, ), (1, ))
    assert_size_stride(arg429_1, (448, ), (1, ))
    assert_size_stride(arg430_1, (448, ), (1, ))
    assert_size_stride(arg431_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg432_1, (56, ), (1, ))
    assert_size_stride(arg433_1, (56, ), (1, ))
    assert_size_stride(arg434_1, (56, ), (1, ))
    assert_size_stride(arg435_1, (56, ), (1, ))
    assert_size_stride(arg436_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg437_1, (56, ), (1, ))
    assert_size_stride(arg438_1, (56, ), (1, ))
    assert_size_stride(arg439_1, (56, ), (1, ))
    assert_size_stride(arg440_1, (56, ), (1, ))
    assert_size_stride(arg441_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg442_1, (56, ), (1, ))
    assert_size_stride(arg443_1, (56, ), (1, ))
    assert_size_stride(arg444_1, (56, ), (1, ))
    assert_size_stride(arg445_1, (56, ), (1, ))
    assert_size_stride(arg446_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg447_1, (56, ), (1, ))
    assert_size_stride(arg448_1, (56, ), (1, ))
    assert_size_stride(arg449_1, (56, ), (1, ))
    assert_size_stride(arg450_1, (56, ), (1, ))
    assert_size_stride(arg451_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg452_1, (56, ), (1, ))
    assert_size_stride(arg453_1, (56, ), (1, ))
    assert_size_stride(arg454_1, (56, ), (1, ))
    assert_size_stride(arg455_1, (56, ), (1, ))
    assert_size_stride(arg456_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg457_1, (56, ), (1, ))
    assert_size_stride(arg458_1, (56, ), (1, ))
    assert_size_stride(arg459_1, (56, ), (1, ))
    assert_size_stride(arg460_1, (56, ), (1, ))
    assert_size_stride(arg461_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg462_1, (56, ), (1, ))
    assert_size_stride(arg463_1, (56, ), (1, ))
    assert_size_stride(arg464_1, (56, ), (1, ))
    assert_size_stride(arg465_1, (56, ), (1, ))
    assert_size_stride(arg466_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, ), (1, ))
    assert_size_stride(arg471_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg472_1, (448, ), (1, ))
    assert_size_stride(arg473_1, (448, ), (1, ))
    assert_size_stride(arg474_1, (448, ), (1, ))
    assert_size_stride(arg475_1, (448, ), (1, ))
    assert_size_stride(arg476_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg477_1, (56, ), (1, ))
    assert_size_stride(arg478_1, (56, ), (1, ))
    assert_size_stride(arg479_1, (56, ), (1, ))
    assert_size_stride(arg480_1, (56, ), (1, ))
    assert_size_stride(arg481_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg482_1, (56, ), (1, ))
    assert_size_stride(arg483_1, (56, ), (1, ))
    assert_size_stride(arg484_1, (56, ), (1, ))
    assert_size_stride(arg485_1, (56, ), (1, ))
    assert_size_stride(arg486_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg487_1, (56, ), (1, ))
    assert_size_stride(arg488_1, (56, ), (1, ))
    assert_size_stride(arg489_1, (56, ), (1, ))
    assert_size_stride(arg490_1, (56, ), (1, ))
    assert_size_stride(arg491_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg492_1, (56, ), (1, ))
    assert_size_stride(arg493_1, (56, ), (1, ))
    assert_size_stride(arg494_1, (56, ), (1, ))
    assert_size_stride(arg495_1, (56, ), (1, ))
    assert_size_stride(arg496_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg497_1, (56, ), (1, ))
    assert_size_stride(arg498_1, (56, ), (1, ))
    assert_size_stride(arg499_1, (56, ), (1, ))
    assert_size_stride(arg500_1, (56, ), (1, ))
    assert_size_stride(arg501_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg502_1, (56, ), (1, ))
    assert_size_stride(arg503_1, (56, ), (1, ))
    assert_size_stride(arg504_1, (56, ), (1, ))
    assert_size_stride(arg505_1, (56, ), (1, ))
    assert_size_stride(arg506_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg507_1, (56, ), (1, ))
    assert_size_stride(arg508_1, (56, ), (1, ))
    assert_size_stride(arg509_1, (56, ), (1, ))
    assert_size_stride(arg510_1, (56, ), (1, ))
    assert_size_stride(arg511_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg512_1, (1024, ), (1, ))
    assert_size_stride(arg513_1, (1024, ), (1, ))
    assert_size_stride(arg514_1, (1024, ), (1, ))
    assert_size_stride(arg515_1, (1024, ), (1, ))
    assert_size_stride(arg516_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg517_1, (448, ), (1, ))
    assert_size_stride(arg518_1, (448, ), (1, ))
    assert_size_stride(arg519_1, (448, ), (1, ))
    assert_size_stride(arg520_1, (448, ), (1, ))
    assert_size_stride(arg521_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg522_1, (56, ), (1, ))
    assert_size_stride(arg523_1, (56, ), (1, ))
    assert_size_stride(arg524_1, (56, ), (1, ))
    assert_size_stride(arg525_1, (56, ), (1, ))
    assert_size_stride(arg526_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg527_1, (56, ), (1, ))
    assert_size_stride(arg528_1, (56, ), (1, ))
    assert_size_stride(arg529_1, (56, ), (1, ))
    assert_size_stride(arg530_1, (56, ), (1, ))
    assert_size_stride(arg531_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg532_1, (56, ), (1, ))
    assert_size_stride(arg533_1, (56, ), (1, ))
    assert_size_stride(arg534_1, (56, ), (1, ))
    assert_size_stride(arg535_1, (56, ), (1, ))
    assert_size_stride(arg536_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg537_1, (56, ), (1, ))
    assert_size_stride(arg538_1, (56, ), (1, ))
    assert_size_stride(arg539_1, (56, ), (1, ))
    assert_size_stride(arg540_1, (56, ), (1, ))
    assert_size_stride(arg541_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg542_1, (56, ), (1, ))
    assert_size_stride(arg543_1, (56, ), (1, ))
    assert_size_stride(arg544_1, (56, ), (1, ))
    assert_size_stride(arg545_1, (56, ), (1, ))
    assert_size_stride(arg546_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg547_1, (56, ), (1, ))
    assert_size_stride(arg548_1, (56, ), (1, ))
    assert_size_stride(arg549_1, (56, ), (1, ))
    assert_size_stride(arg550_1, (56, ), (1, ))
    assert_size_stride(arg551_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg552_1, (56, ), (1, ))
    assert_size_stride(arg553_1, (56, ), (1, ))
    assert_size_stride(arg554_1, (56, ), (1, ))
    assert_size_stride(arg555_1, (56, ), (1, ))
    assert_size_stride(arg556_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg557_1, (1024, ), (1, ))
    assert_size_stride(arg558_1, (1024, ), (1, ))
    assert_size_stride(arg559_1, (1024, ), (1, ))
    assert_size_stride(arg560_1, (1024, ), (1, ))
    assert_size_stride(arg561_1, (448, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg562_1, (448, ), (1, ))
    assert_size_stride(arg563_1, (448, ), (1, ))
    assert_size_stride(arg564_1, (448, ), (1, ))
    assert_size_stride(arg565_1, (448, ), (1, ))
    assert_size_stride(arg566_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg567_1, (56, ), (1, ))
    assert_size_stride(arg568_1, (56, ), (1, ))
    assert_size_stride(arg569_1, (56, ), (1, ))
    assert_size_stride(arg570_1, (56, ), (1, ))
    assert_size_stride(arg571_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg572_1, (56, ), (1, ))
    assert_size_stride(arg573_1, (56, ), (1, ))
    assert_size_stride(arg574_1, (56, ), (1, ))
    assert_size_stride(arg575_1, (56, ), (1, ))
    assert_size_stride(arg576_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg577_1, (56, ), (1, ))
    assert_size_stride(arg578_1, (56, ), (1, ))
    assert_size_stride(arg579_1, (56, ), (1, ))
    assert_size_stride(arg580_1, (56, ), (1, ))
    assert_size_stride(arg581_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg582_1, (56, ), (1, ))
    assert_size_stride(arg583_1, (56, ), (1, ))
    assert_size_stride(arg584_1, (56, ), (1, ))
    assert_size_stride(arg585_1, (56, ), (1, ))
    assert_size_stride(arg586_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg587_1, (56, ), (1, ))
    assert_size_stride(arg588_1, (56, ), (1, ))
    assert_size_stride(arg589_1, (56, ), (1, ))
    assert_size_stride(arg590_1, (56, ), (1, ))
    assert_size_stride(arg591_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg592_1, (56, ), (1, ))
    assert_size_stride(arg593_1, (56, ), (1, ))
    assert_size_stride(arg594_1, (56, ), (1, ))
    assert_size_stride(arg595_1, (56, ), (1, ))
    assert_size_stride(arg596_1, (56, 56, 3, 3), (504, 9, 3, 1))
    assert_size_stride(arg597_1, (56, ), (1, ))
    assert_size_stride(arg598_1, (56, ), (1, ))
    assert_size_stride(arg599_1, (56, ), (1, ))
    assert_size_stride(arg600_1, (56, ), (1, ))
    assert_size_stride(arg601_1, (1024, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg602_1, (1024, ), (1, ))
    assert_size_stride(arg603_1, (1024, ), (1, ))
    assert_size_stride(arg604_1, (1024, ), (1, ))
    assert_size_stride(arg605_1, (1024, ), (1, ))
    assert_size_stride(arg606_1, (896, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg607_1, (896, ), (1, ))
    assert_size_stride(arg608_1, (896, ), (1, ))
    assert_size_stride(arg609_1, (896, ), (1, ))
    assert_size_stride(arg610_1, (896, ), (1, ))
    assert_size_stride(arg611_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg612_1, (112, ), (1, ))
    assert_size_stride(arg613_1, (112, ), (1, ))
    assert_size_stride(arg614_1, (112, ), (1, ))
    assert_size_stride(arg615_1, (112, ), (1, ))
    assert_size_stride(arg616_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg617_1, (112, ), (1, ))
    assert_size_stride(arg618_1, (112, ), (1, ))
    assert_size_stride(arg619_1, (112, ), (1, ))
    assert_size_stride(arg620_1, (112, ), (1, ))
    assert_size_stride(arg621_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg622_1, (112, ), (1, ))
    assert_size_stride(arg623_1, (112, ), (1, ))
    assert_size_stride(arg624_1, (112, ), (1, ))
    assert_size_stride(arg625_1, (112, ), (1, ))
    assert_size_stride(arg626_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg627_1, (112, ), (1, ))
    assert_size_stride(arg628_1, (112, ), (1, ))
    assert_size_stride(arg629_1, (112, ), (1, ))
    assert_size_stride(arg630_1, (112, ), (1, ))
    assert_size_stride(arg631_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg632_1, (112, ), (1, ))
    assert_size_stride(arg633_1, (112, ), (1, ))
    assert_size_stride(arg634_1, (112, ), (1, ))
    assert_size_stride(arg635_1, (112, ), (1, ))
    assert_size_stride(arg636_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg637_1, (112, ), (1, ))
    assert_size_stride(arg638_1, (112, ), (1, ))
    assert_size_stride(arg639_1, (112, ), (1, ))
    assert_size_stride(arg640_1, (112, ), (1, ))
    assert_size_stride(arg641_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg642_1, (112, ), (1, ))
    assert_size_stride(arg643_1, (112, ), (1, ))
    assert_size_stride(arg644_1, (112, ), (1, ))
    assert_size_stride(arg645_1, (112, ), (1, ))
    assert_size_stride(arg646_1, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg647_1, (2048, ), (1, ))
    assert_size_stride(arg648_1, (2048, ), (1, ))
    assert_size_stride(arg649_1, (2048, ), (1, ))
    assert_size_stride(arg650_1, (2048, ), (1, ))
    assert_size_stride(arg651_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg652_1, (2048, ), (1, ))
    assert_size_stride(arg653_1, (2048, ), (1, ))
    assert_size_stride(arg654_1, (2048, ), (1, ))
    assert_size_stride(arg655_1, (2048, ), (1, ))
    assert_size_stride(arg656_1, (896, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg657_1, (896, ), (1, ))
    assert_size_stride(arg658_1, (896, ), (1, ))
    assert_size_stride(arg659_1, (896, ), (1, ))
    assert_size_stride(arg660_1, (896, ), (1, ))
    assert_size_stride(arg661_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg662_1, (112, ), (1, ))
    assert_size_stride(arg663_1, (112, ), (1, ))
    assert_size_stride(arg664_1, (112, ), (1, ))
    assert_size_stride(arg665_1, (112, ), (1, ))
    assert_size_stride(arg666_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg667_1, (112, ), (1, ))
    assert_size_stride(arg668_1, (112, ), (1, ))
    assert_size_stride(arg669_1, (112, ), (1, ))
    assert_size_stride(arg670_1, (112, ), (1, ))
    assert_size_stride(arg671_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg672_1, (112, ), (1, ))
    assert_size_stride(arg673_1, (112, ), (1, ))
    assert_size_stride(arg674_1, (112, ), (1, ))
    assert_size_stride(arg675_1, (112, ), (1, ))
    assert_size_stride(arg676_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg677_1, (112, ), (1, ))
    assert_size_stride(arg678_1, (112, ), (1, ))
    assert_size_stride(arg679_1, (112, ), (1, ))
    assert_size_stride(arg680_1, (112, ), (1, ))
    assert_size_stride(arg681_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg682_1, (112, ), (1, ))
    assert_size_stride(arg683_1, (112, ), (1, ))
    assert_size_stride(arg684_1, (112, ), (1, ))
    assert_size_stride(arg685_1, (112, ), (1, ))
    assert_size_stride(arg686_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg687_1, (112, ), (1, ))
    assert_size_stride(arg688_1, (112, ), (1, ))
    assert_size_stride(arg689_1, (112, ), (1, ))
    assert_size_stride(arg690_1, (112, ), (1, ))
    assert_size_stride(arg691_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg692_1, (112, ), (1, ))
    assert_size_stride(arg693_1, (112, ), (1, ))
    assert_size_stride(arg694_1, (112, ), (1, ))
    assert_size_stride(arg695_1, (112, ), (1, ))
    assert_size_stride(arg696_1, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg697_1, (2048, ), (1, ))
    assert_size_stride(arg698_1, (2048, ), (1, ))
    assert_size_stride(arg699_1, (2048, ), (1, ))
    assert_size_stride(arg700_1, (2048, ), (1, ))
    assert_size_stride(arg701_1, (896, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg702_1, (896, ), (1, ))
    assert_size_stride(arg703_1, (896, ), (1, ))
    assert_size_stride(arg704_1, (896, ), (1, ))
    assert_size_stride(arg705_1, (896, ), (1, ))
    assert_size_stride(arg706_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg707_1, (112, ), (1, ))
    assert_size_stride(arg708_1, (112, ), (1, ))
    assert_size_stride(arg709_1, (112, ), (1, ))
    assert_size_stride(arg710_1, (112, ), (1, ))
    assert_size_stride(arg711_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg712_1, (112, ), (1, ))
    assert_size_stride(arg713_1, (112, ), (1, ))
    assert_size_stride(arg714_1, (112, ), (1, ))
    assert_size_stride(arg715_1, (112, ), (1, ))
    assert_size_stride(arg716_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg717_1, (112, ), (1, ))
    assert_size_stride(arg718_1, (112, ), (1, ))
    assert_size_stride(arg719_1, (112, ), (1, ))
    assert_size_stride(arg720_1, (112, ), (1, ))
    assert_size_stride(arg721_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg722_1, (112, ), (1, ))
    assert_size_stride(arg723_1, (112, ), (1, ))
    assert_size_stride(arg724_1, (112, ), (1, ))
    assert_size_stride(arg725_1, (112, ), (1, ))
    assert_size_stride(arg726_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg727_1, (112, ), (1, ))
    assert_size_stride(arg728_1, (112, ), (1, ))
    assert_size_stride(arg729_1, (112, ), (1, ))
    assert_size_stride(arg730_1, (112, ), (1, ))
    assert_size_stride(arg731_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg732_1, (112, ), (1, ))
    assert_size_stride(arg733_1, (112, ), (1, ))
    assert_size_stride(arg734_1, (112, ), (1, ))
    assert_size_stride(arg735_1, (112, ), (1, ))
    assert_size_stride(arg736_1, (112, 112, 3, 3), (1008, 9, 3, 1))
    assert_size_stride(arg737_1, (112, ), (1, ))
    assert_size_stride(arg738_1, (112, ), (1, ))
    assert_size_stride(arg739_1, (112, ), (1, ))
    assert_size_stride(arg740_1, (112, ), (1, ))
    assert_size_stride(arg741_1, (2048, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg742_1, (2048, ), (1, ))
    assert_size_stride(arg743_1, (2048, ), (1, ))
    assert_size_stride(arg744_1, (2048, ), (1, ))
    assert_size_stride(arg745_1, (2048, ), (1, ))
    assert_size_stride(arg746_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg747_1, (1000, ), (1, ))
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
        assert_size_stride(buf5, (8, 112, 56, 56), (351232, 1, 6272, 112))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [out_129, out_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((14, 14, 3, 3), (126, 1, 42, 14), torch.float32)
        # Topologically Sorted Source Nodes: [sp_449], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg11_1, buf7, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [sp_449], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sp_453], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg16_1, buf9, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [sp_453], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 14, 56, 56), (351232, 1, 6272, 112), 14), buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sp_457], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg21_1, buf11, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg21_1
        # Topologically Sorted Source Nodes: [sp_457], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 14, 56, 56), (351232, 1, 6272, 112), 28), buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [sp_461], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg26_1, buf13, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg26_1
        # Topologically Sorted Source Nodes: [sp_461], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 14, 56, 56), (351232, 1, 6272, 112), 42), buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf15 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [sp_465], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg31_1, buf15, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [sp_465], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 14, 56, 56), (351232, 1, 6272, 112), 56), buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [sp_469], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg36_1, buf17, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg36_1
        # Topologically Sorted Source Nodes: [sp_469], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 14, 56, 56), (351232, 1, 6272, 112), 70), buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [sp_473], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg41_1, buf19, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg41_1
        # Topologically Sorted Source Nodes: [sp_473], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(reinterpret_tensor(buf6, (8, 14, 56, 56), (351232, 1, 6272, 112), 84), buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf29 = empty_strided_cuda((8, 112, 56, 56), (351232, 3136, 56, 1), torch.float32)
        buf21 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_6.run(buf6, buf21, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf22 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_450, sp_451], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf8, arg12_1, arg13_1, arg14_1, arg15_1, buf22, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del buf8
        buf23 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_454, sp_455], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf10, arg17_1, arg18_1, arg19_1, arg20_1, buf23, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf10
        buf24 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        # Topologically Sorted Source Nodes: [sp_458, sp_459], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf12, arg22_1, arg23_1, arg24_1, arg25_1, buf24, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf12
        buf25 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [sp_462, sp_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf14, arg27_1, arg28_1, arg29_1, arg30_1, buf25, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        del buf14
        buf26 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        # Topologically Sorted Source Nodes: [sp_466, sp_467], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf16, arg32_1, arg33_1, arg34_1, arg35_1, buf26, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        del buf16
        buf27 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        # Topologically Sorted Source Nodes: [sp_470, sp_471], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf18, arg37_1, arg38_1, arg39_1, arg40_1, buf27, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del buf18
        buf28 = reinterpret_tensor(buf29, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        # Topologically Sorted Source Nodes: [sp_474, sp_475], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf20, arg42_1, arg43_1, arg44_1, arg45_1, buf28, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        del buf20
        buf30 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf29, buf30, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf21
        del buf22
        del buf23
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        # Topologically Sorted Source Nodes: [out_132], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg46_1
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf4, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg51_1
        buf33 = buf31; del buf31  # reuse
        buf34 = reinterpret_tensor(buf3, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [out_133, input_10, out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf33, arg47_1, arg48_1, arg49_1, arg50_1, buf32, arg52_1, arg53_1, arg54_1, arg55_1, buf34, 6422528, grid=grid(6422528), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf32
        del buf33
        # Topologically Sorted Source Nodes: [out_135, out_136], Original ATen: [aten.relu, aten.convolution]
        buf35 = extern_kernels.convolution(buf34, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 112, 56, 56), (351232, 1, 6272, 112))
        del arg56_1
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [out_137, out_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf36, arg57_1, arg58_1, arg59_1, arg60_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        buf37 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [sp_477], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg61_1, buf37, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg61_1
        # Topologically Sorted Source Nodes: [sp_477], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(reinterpret_tensor(buf36, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf65 = reinterpret_tensor(buf30, (8, 112, 56, 56), (351232, 3136, 56, 1), 0); del buf30  # reuse
        buf39 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_478, sp_479], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf38, arg62_1, arg63_1, arg64_1, arg65_1, buf39, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        buf40 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [sp_480], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf39, buf36, buf40, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf41 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [sp_480, sp_481], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg66_1, buf41, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg66_1
        # Topologically Sorted Source Nodes: [sp_480, sp_481], Original ATen: [aten.add, aten.convolution]
        buf42 = extern_kernels.convolution(buf40, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf40
        buf43 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_482, sp_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf42, arg67_1, arg68_1, arg69_1, arg70_1, buf43, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        buf44 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [sp_484], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf43, buf36, buf44, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf45 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [sp_484, sp_485], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg71_1, buf45, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg71_1
        # Topologically Sorted Source Nodes: [sp_484, sp_485], Original ATen: [aten.add, aten.convolution]
        buf46 = extern_kernels.convolution(buf44, buf45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf44
        buf47 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        # Topologically Sorted Source Nodes: [sp_486, sp_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf46, arg72_1, arg73_1, arg74_1, arg75_1, buf47, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        buf48 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [sp_488], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf47, buf36, buf48, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf49 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [sp_488, sp_489], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg76_1, buf49, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg76_1
        # Topologically Sorted Source Nodes: [sp_488, sp_489], Original ATen: [aten.add, aten.convolution]
        buf50 = extern_kernels.convolution(buf48, buf49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf48
        buf51 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [sp_490, sp_491], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf50, arg77_1, arg78_1, arg79_1, arg80_1, buf51, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf52 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [sp_492], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf51, buf36, buf52, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf53 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [sp_492, sp_493], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg81_1, buf53, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [sp_492, sp_493], Original ATen: [aten.add, aten.convolution]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf52
        buf55 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        # Topologically Sorted Source Nodes: [sp_494, sp_495], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf54, arg82_1, arg83_1, arg84_1, arg85_1, buf55, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        buf56 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [sp_496], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf55, buf36, buf56, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf57 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [sp_496, sp_497], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg86_1, buf57, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg86_1
        # Topologically Sorted Source Nodes: [sp_496, sp_497], Original ATen: [aten.add, aten.convolution]
        buf58 = extern_kernels.convolution(buf56, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf56
        buf59 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        # Topologically Sorted Source Nodes: [sp_498, sp_499], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf58, arg87_1, arg88_1, arg89_1, arg90_1, buf59, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        buf60 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [sp_500], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf59, buf36, buf60, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf61 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [sp_500, sp_501], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg91_1, buf61, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg91_1
        # Topologically Sorted Source Nodes: [sp_500, sp_501], Original ATen: [aten.add, aten.convolution]
        buf62 = extern_kernels.convolution(buf60, buf61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf60
        buf63 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        # Topologically Sorted Source Nodes: [sp_502, sp_503], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf62, arg92_1, arg93_1, arg94_1, arg95_1, buf63, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        del buf62
        buf64 = reinterpret_tensor(buf65, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Topologically Sorted Source Nodes: [out_139], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf36, buf64, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf66 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf65, buf66, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf39
        del buf43
        del buf47
        del buf51
        del buf55
        del buf59
        del buf63
        del buf64
        del buf65
        # Topologically Sorted Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg96_1
        buf68 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [out_141, out_142, out_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf68, buf67, arg97_1, arg98_1, arg99_1, arg100_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf67
        # Topologically Sorted Source Nodes: [out_144], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 112, 56, 56), (351232, 1, 6272, 112))
        del arg101_1
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [out_145, out_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf70, arg102_1, arg103_1, arg104_1, arg105_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        buf71 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [sp_505], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg106_1, buf71, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg106_1
        # Topologically Sorted Source Nodes: [sp_505], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(reinterpret_tensor(buf70, (8, 14, 56, 56), (351232, 1, 6272, 112), 0), buf71, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 14, 56, 56), (43904, 1, 784, 14))
        buf99 = reinterpret_tensor(buf66, (8, 112, 56, 56), (351232, 3136, 56, 1), 0); del buf66  # reuse
        buf73 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_506, sp_507], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf72, arg107_1, arg108_1, arg109_1, arg110_1, buf73, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        buf74 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [sp_508], Original ATen: [aten.add]
        triton_poi_fused_add_10.run(buf73, buf70, buf74, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf75 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [sp_508, sp_509], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg111_1, buf75, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg111_1
        # Topologically Sorted Source Nodes: [sp_508, sp_509], Original ATen: [aten.add, aten.convolution]
        buf76 = extern_kernels.convolution(buf74, buf75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf74
        buf77 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_510, sp_511], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf76, arg112_1, arg113_1, arg114_1, arg115_1, buf77, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        buf78 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [sp_512], Original ATen: [aten.add]
        triton_poi_fused_add_11.run(buf77, buf70, buf78, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf79 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [sp_512, sp_513], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg116_1, buf79, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg116_1
        # Topologically Sorted Source Nodes: [sp_512, sp_513], Original ATen: [aten.add, aten.convolution]
        buf80 = extern_kernels.convolution(buf78, buf79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf78
        buf81 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 87808)  # alias
        # Topologically Sorted Source Nodes: [sp_514, sp_515], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf80, arg117_1, arg118_1, arg119_1, arg120_1, buf81, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        buf82 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [sp_516], Original ATen: [aten.add]
        triton_poi_fused_add_12.run(buf81, buf70, buf82, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf83 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [sp_516, sp_517], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg121_1, buf83, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg121_1
        # Topologically Sorted Source Nodes: [sp_516, sp_517], Original ATen: [aten.add, aten.convolution]
        buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf82
        buf85 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [sp_518, sp_519], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf84, arg122_1, arg123_1, arg124_1, arg125_1, buf85, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf86 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [sp_520], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf85, buf70, buf86, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf87 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [sp_520, sp_521], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg126_1, buf87, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [sp_520, sp_521], Original ATen: [aten.add, aten.convolution]
        buf88 = extern_kernels.convolution(buf86, buf87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf86
        buf89 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 175616)  # alias
        # Topologically Sorted Source Nodes: [sp_522, sp_523], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf88, arg127_1, arg128_1, arg129_1, arg130_1, buf89, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        buf90 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [sp_524], Original ATen: [aten.add]
        triton_poi_fused_add_14.run(buf89, buf70, buf90, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf91 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [sp_524, sp_525], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg131_1, buf91, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg131_1
        # Topologically Sorted Source Nodes: [sp_524, sp_525], Original ATen: [aten.add, aten.convolution]
        buf92 = extern_kernels.convolution(buf90, buf91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf90
        buf93 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 219520)  # alias
        # Topologically Sorted Source Nodes: [sp_526, sp_527], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf92, arg132_1, arg133_1, arg134_1, arg135_1, buf93, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        buf94 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [sp_528], Original ATen: [aten.add]
        triton_poi_fused_add_15.run(buf93, buf70, buf94, 25088, 14, grid=grid(25088, 14), stream=stream0)
        buf95 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [sp_528, sp_529], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_5.run(arg136_1, buf95, 196, 9, grid=grid(196, 9), stream=stream0)
        del arg136_1
        # Topologically Sorted Source Nodes: [sp_528, sp_529], Original ATen: [aten.add, aten.convolution]
        buf96 = extern_kernels.convolution(buf94, buf95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 14, 56, 56), (43904, 1, 784, 14))
        del buf95
        buf97 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 263424)  # alias
        # Topologically Sorted Source Nodes: [sp_530, sp_531], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf96, arg137_1, arg138_1, arg139_1, arg140_1, buf97, 112, 3136, grid=grid(112, 3136), stream=stream0)
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        buf98 = reinterpret_tensor(buf99, (8, 14, 56, 56), (351232, 3136, 56, 1), 307328)  # alias
        # Topologically Sorted Source Nodes: [out_147], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf70, buf98, 112, 3136, grid=grid(112, 3136), stream=stream0)
        buf100 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_8.run(buf99, buf100, 896, 3136, grid=grid(896, 3136), stream=stream0)
        del buf73
        del buf77
        del buf81
        del buf85
        del buf89
        del buf93
        del buf97
        del buf98
        del buf99
        # Topologically Sorted Source Nodes: [out_148], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg141_1
        del buf100
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [out_149, out_150, out_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf102, arg142_1, arg143_1, arg144_1, arg145_1, buf68, 6422528, grid=grid(6422528), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        del buf68
        # Topologically Sorted Source Nodes: [out_152], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 224, 56, 56), (702464, 1, 12544, 224))
        del arg146_1
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [out_153, out_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf104, arg147_1, arg148_1, arg149_1, arg150_1, 5619712, grid=grid(5619712), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        buf105 = empty_strided_cuda((28, 28, 3, 3), (252, 1, 84, 28), torch.float32)
        # Topologically Sorted Source Nodes: [sp_533], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg151_1, buf105, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg151_1
        # Topologically Sorted Source Nodes: [sp_533], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 28, 56, 56), (702464, 1, 12544, 224), 0), buf105, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf107 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [sp_537], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg156_1, buf107, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg156_1
        # Topologically Sorted Source Nodes: [sp_537], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 28, 56, 56), (702464, 1, 12544, 224), 28), buf107, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf109 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [sp_541], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg161_1, buf109, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg161_1
        # Topologically Sorted Source Nodes: [sp_541], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 28, 56, 56), (702464, 1, 12544, 224), 56), buf109, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf111 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [sp_545], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg166_1, buf111, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg166_1
        # Topologically Sorted Source Nodes: [sp_545], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 28, 56, 56), (702464, 1, 12544, 224), 84), buf111, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf113 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [sp_549], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg171_1, buf113, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg171_1
        # Topologically Sorted Source Nodes: [sp_549], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 28, 56, 56), (702464, 1, 12544, 224), 112), buf113, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf115 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [sp_553], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg176_1, buf115, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [sp_553], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 28, 56, 56), (702464, 1, 12544, 224), 140), buf115, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf117 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [sp_557], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg181_1, buf117, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg181_1
        # Topologically Sorted Source Nodes: [sp_557], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(reinterpret_tensor(buf104, (8, 28, 56, 56), (702464, 1, 12544, 224), 168), buf117, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf127 = empty_strided_cuda((8, 224, 28, 28), (175616, 784, 28, 1), torch.float32)
        buf119 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_5], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_21.run(buf104, buf119, 6272, 28, grid=grid(6272, 28), stream=stream0)
        del buf104
        buf120 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_534, sp_535], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf106, arg152_1, arg153_1, arg154_1, arg155_1, buf120, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del buf106
        buf121 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_538, sp_539], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf108, arg157_1, arg158_1, arg159_1, arg160_1, buf121, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        del buf108
        buf122 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_542, sp_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf110, arg162_1, arg163_1, arg164_1, arg165_1, buf122, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        del buf110
        buf123 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_546, sp_547], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf112, arg167_1, arg168_1, arg169_1, arg170_1, buf123, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        del buf112
        buf124 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        # Topologically Sorted Source Nodes: [sp_550, sp_551], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf114, arg172_1, arg173_1, arg174_1, arg175_1, buf124, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del buf114
        buf125 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        # Topologically Sorted Source Nodes: [sp_554, sp_555], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf116, arg177_1, arg178_1, arg179_1, arg180_1, buf125, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf116
        buf126 = reinterpret_tensor(buf127, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [sp_558, sp_559], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf118, arg182_1, arg183_1, arg184_1, arg185_1, buf126, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del buf118
        buf128 = empty_strided_cuda((8, 224, 28, 28), (175616, 1, 6272, 224), torch.float32)
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf127, buf128, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf119
        del buf120
        del buf121
        del buf122
        del buf123
        del buf124
        del buf125
        del buf126
        del buf127
        # Topologically Sorted Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg186_1
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf102, arg191_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg191_1
        del buf102
        buf131 = buf129; del buf129  # reuse
        buf132 = empty_strided_cuda((8, 512, 28, 28), (401408, 1, 14336, 512), torch.float32)
        # Topologically Sorted Source Nodes: [out_157, input_12, out_158, out_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_24.run(buf131, arg187_1, arg188_1, arg189_1, arg190_1, buf130, arg192_1, arg193_1, arg194_1, arg195_1, buf132, 3211264, grid=grid(3211264), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        del buf130
        del buf131
        # Topologically Sorted Source Nodes: [out_159, out_160], Original ATen: [aten.relu, aten.convolution]
        buf133 = extern_kernels.convolution(buf132, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 224, 28, 28), (175616, 1, 6272, 224))
        del arg196_1
        buf134 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [out_161, out_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf134, arg197_1, arg198_1, arg199_1, arg200_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        buf135 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [sp_561], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg201_1, buf135, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg201_1
        # Topologically Sorted Source Nodes: [sp_561], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(reinterpret_tensor(buf134, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf163 = reinterpret_tensor(buf128, (8, 224, 28, 28), (175616, 784, 28, 1), 0); del buf128  # reuse
        buf137 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_562, sp_563], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf136, arg202_1, arg203_1, arg204_1, arg205_1, buf137, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        buf138 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [sp_564], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf137, buf134, buf138, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf139 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [sp_564, sp_565], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg206_1, buf139, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg206_1
        # Topologically Sorted Source Nodes: [sp_564, sp_565], Original ATen: [aten.add, aten.convolution]
        buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf138
        buf141 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_566, sp_567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf140, arg207_1, arg208_1, arg209_1, arg210_1, buf141, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        buf142 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [sp_568], Original ATen: [aten.add]
        triton_poi_fused_add_27.run(buf141, buf134, buf142, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf143 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [sp_568, sp_569], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg211_1, buf143, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg211_1
        # Topologically Sorted Source Nodes: [sp_568, sp_569], Original ATen: [aten.add, aten.convolution]
        buf144 = extern_kernels.convolution(buf142, buf143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf142
        buf145 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_570, sp_571], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf144, arg212_1, arg213_1, arg214_1, arg215_1, buf145, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        buf146 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [sp_572], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf145, buf134, buf146, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf147 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [sp_572, sp_573], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg216_1, buf147, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg216_1
        # Topologically Sorted Source Nodes: [sp_572, sp_573], Original ATen: [aten.add, aten.convolution]
        buf148 = extern_kernels.convolution(buf146, buf147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf146
        buf149 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_574, sp_575], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf148, arg217_1, arg218_1, arg219_1, arg220_1, buf149, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        buf150 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [sp_576], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf149, buf134, buf150, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf151 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [sp_576, sp_577], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg221_1, buf151, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [sp_576, sp_577], Original ATen: [aten.add, aten.convolution]
        buf152 = extern_kernels.convolution(buf150, buf151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf150
        buf153 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        # Topologically Sorted Source Nodes: [sp_578, sp_579], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf152, arg222_1, arg223_1, arg224_1, arg225_1, buf153, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        buf154 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [sp_580], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf153, buf134, buf154, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf155 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [sp_580, sp_581], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg226_1, buf155, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg226_1
        # Topologically Sorted Source Nodes: [sp_580, sp_581], Original ATen: [aten.add, aten.convolution]
        buf156 = extern_kernels.convolution(buf154, buf155, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf154
        buf157 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        # Topologically Sorted Source Nodes: [sp_582, sp_583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf156, arg227_1, arg228_1, arg229_1, arg230_1, buf157, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        buf158 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [sp_584], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf157, buf134, buf158, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf159 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [sp_584, sp_585], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg231_1, buf159, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg231_1
        # Topologically Sorted Source Nodes: [sp_584, sp_585], Original ATen: [aten.add, aten.convolution]
        buf160 = extern_kernels.convolution(buf158, buf159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf158
        buf161 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [sp_586, sp_587], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf160, arg232_1, arg233_1, arg234_1, arg235_1, buf161, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        del buf160
        buf162 = reinterpret_tensor(buf163, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Topologically Sorted Source Nodes: [out_163], Original ATen: [aten.cat]
        triton_poi_fused_cat_32.run(buf134, buf162, 224, 784, grid=grid(224, 784), stream=stream0)
        buf164 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [out_164], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf163, buf164, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf137
        del buf141
        del buf145
        del buf149
        del buf153
        del buf157
        del buf161
        del buf162
        del buf163
        # Topologically Sorted Source Nodes: [out_164], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg236_1
        buf166 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [out_165, out_166, out_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf166, buf165, arg237_1, arg238_1, arg239_1, arg240_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del buf165
        # Topologically Sorted Source Nodes: [out_168], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 224, 28, 28), (175616, 1, 6272, 224))
        del arg241_1
        buf168 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [out_169, out_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf168, arg242_1, arg243_1, arg244_1, arg245_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        buf169 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [sp_589], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg246_1, buf169, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg246_1
        # Topologically Sorted Source Nodes: [sp_589], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(reinterpret_tensor(buf168, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf197 = reinterpret_tensor(buf164, (8, 224, 28, 28), (175616, 784, 28, 1), 0); del buf164  # reuse
        buf171 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_590, sp_591], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf170, arg247_1, arg248_1, arg249_1, arg250_1, buf171, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        buf172 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [sp_592], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf171, buf168, buf172, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf173 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [sp_592, sp_593], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg251_1, buf173, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg251_1
        # Topologically Sorted Source Nodes: [sp_592, sp_593], Original ATen: [aten.add, aten.convolution]
        buf174 = extern_kernels.convolution(buf172, buf173, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf172
        buf175 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_594, sp_595], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf174, arg252_1, arg253_1, arg254_1, arg255_1, buf175, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        buf176 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [sp_596], Original ATen: [aten.add]
        triton_poi_fused_add_27.run(buf175, buf168, buf176, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf177 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [sp_596, sp_597], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg256_1, buf177, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg256_1
        # Topologically Sorted Source Nodes: [sp_596, sp_597], Original ATen: [aten.add, aten.convolution]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf176
        buf179 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_598, sp_599], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf178, arg257_1, arg258_1, arg259_1, arg260_1, buf179, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        buf180 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [sp_600], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf179, buf168, buf180, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf181 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [sp_600, sp_601], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg261_1, buf181, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg261_1
        # Topologically Sorted Source Nodes: [sp_600, sp_601], Original ATen: [aten.add, aten.convolution]
        buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf180
        buf183 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_602, sp_603], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf182, arg262_1, arg263_1, arg264_1, arg265_1, buf183, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        buf184 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [sp_604], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf183, buf168, buf184, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf185 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [sp_604, sp_605], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg266_1, buf185, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg266_1
        # Topologically Sorted Source Nodes: [sp_604, sp_605], Original ATen: [aten.add, aten.convolution]
        buf186 = extern_kernels.convolution(buf184, buf185, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf184
        buf187 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        # Topologically Sorted Source Nodes: [sp_606, sp_607], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf186, arg267_1, arg268_1, arg269_1, arg270_1, buf187, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        buf188 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [sp_608], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf187, buf168, buf188, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf189 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [sp_608, sp_609], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg271_1, buf189, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg271_1
        # Topologically Sorted Source Nodes: [sp_608, sp_609], Original ATen: [aten.add, aten.convolution]
        buf190 = extern_kernels.convolution(buf188, buf189, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf188
        buf191 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        # Topologically Sorted Source Nodes: [sp_610, sp_611], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf190, arg272_1, arg273_1, arg274_1, arg275_1, buf191, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        buf192 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [sp_612], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf191, buf168, buf192, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf193 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [sp_612, sp_613], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg276_1, buf193, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg276_1
        # Topologically Sorted Source Nodes: [sp_612, sp_613], Original ATen: [aten.add, aten.convolution]
        buf194 = extern_kernels.convolution(buf192, buf193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf192
        buf195 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [sp_614, sp_615], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf194, arg277_1, arg278_1, arg279_1, arg280_1, buf195, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        del buf194
        buf196 = reinterpret_tensor(buf197, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Topologically Sorted Source Nodes: [out_171], Original ATen: [aten.cat]
        triton_poi_fused_cat_32.run(buf168, buf196, 224, 784, grid=grid(224, 784), stream=stream0)
        buf198 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [out_172], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf197, buf198, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf171
        del buf175
        del buf179
        del buf183
        del buf187
        del buf191
        del buf195
        del buf196
        del buf197
        # Topologically Sorted Source Nodes: [out_172], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg281_1
        buf200 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [out_173, out_174, out_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf200, buf199, arg282_1, arg283_1, arg284_1, arg285_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        del buf199
        # Topologically Sorted Source Nodes: [out_176], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 224, 28, 28), (175616, 1, 6272, 224))
        del arg286_1
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [out_177, out_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf202, arg287_1, arg288_1, arg289_1, arg290_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        buf203 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [sp_617], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(arg291_1, buf203, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg291_1
        # Topologically Sorted Source Nodes: [sp_617], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(reinterpret_tensor(buf202, (8, 28, 28, 28), (175616, 1, 6272, 224), 0), buf203, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 28, 28, 28), (21952, 1, 784, 28))
        buf231 = reinterpret_tensor(buf198, (8, 224, 28, 28), (175616, 784, 28, 1), 0); del buf198  # reuse
        buf205 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_618, sp_619], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf204, arg292_1, arg293_1, arg294_1, arg295_1, buf205, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        buf206 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [sp_620], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf205, buf202, buf206, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf207 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [sp_620, sp_621], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg296_1, buf207, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg296_1
        # Topologically Sorted Source Nodes: [sp_620, sp_621], Original ATen: [aten.add, aten.convolution]
        buf208 = extern_kernels.convolution(buf206, buf207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf206
        buf209 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_622, sp_623], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf208, arg297_1, arg298_1, arg299_1, arg300_1, buf209, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        buf210 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [sp_624], Original ATen: [aten.add]
        triton_poi_fused_add_27.run(buf209, buf202, buf210, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf211 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [sp_624, sp_625], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg301_1, buf211, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg301_1
        # Topologically Sorted Source Nodes: [sp_624, sp_625], Original ATen: [aten.add, aten.convolution]
        buf212 = extern_kernels.convolution(buf210, buf211, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf210
        buf213 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_626, sp_627], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf212, arg302_1, arg303_1, arg304_1, arg305_1, buf213, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        buf214 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [sp_628], Original ATen: [aten.add]
        triton_poi_fused_add_28.run(buf213, buf202, buf214, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf215 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [sp_628, sp_629], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg306_1, buf215, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg306_1
        # Topologically Sorted Source Nodes: [sp_628, sp_629], Original ATen: [aten.add, aten.convolution]
        buf216 = extern_kernels.convolution(buf214, buf215, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf214
        buf217 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_630, sp_631], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf216, arg307_1, arg308_1, arg309_1, arg310_1, buf217, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        buf218 = buf216; del buf216  # reuse
        # Topologically Sorted Source Nodes: [sp_632], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf217, buf202, buf218, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf219 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [sp_632, sp_633], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg311_1, buf219, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [sp_632, sp_633], Original ATen: [aten.add, aten.convolution]
        buf220 = extern_kernels.convolution(buf218, buf219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf218
        buf221 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 87808)  # alias
        # Topologically Sorted Source Nodes: [sp_634, sp_635], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf220, arg312_1, arg313_1, arg314_1, arg315_1, buf221, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        buf222 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [sp_636], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf221, buf202, buf222, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf223 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [sp_636, sp_637], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg316_1, buf223, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg316_1
        # Topologically Sorted Source Nodes: [sp_636, sp_637], Original ATen: [aten.add, aten.convolution]
        buf224 = extern_kernels.convolution(buf222, buf223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf222
        buf225 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 109760)  # alias
        # Topologically Sorted Source Nodes: [sp_638, sp_639], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf224, arg317_1, arg318_1, arg319_1, arg320_1, buf225, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        buf226 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [sp_640], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf225, buf202, buf226, 6272, 28, grid=grid(6272, 28), stream=stream0)
        buf227 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [sp_640, sp_641], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_20.run(arg321_1, buf227, 784, 9, grid=grid(784, 9), stream=stream0)
        del arg321_1
        # Topologically Sorted Source Nodes: [sp_640, sp_641], Original ATen: [aten.add, aten.convolution]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 28, 28, 28), (21952, 1, 784, 28))
        del buf226
        del buf227
        buf229 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 131712)  # alias
        # Topologically Sorted Source Nodes: [sp_642, sp_643], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf228, arg322_1, arg323_1, arg324_1, arg325_1, buf229, 224, 784, grid=grid(224, 784), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        del buf228
        buf230 = reinterpret_tensor(buf231, (8, 28, 28, 28), (175616, 784, 28, 1), 153664)  # alias
        # Topologically Sorted Source Nodes: [out_179], Original ATen: [aten.cat]
        triton_poi_fused_cat_32.run(buf202, buf230, 224, 784, grid=grid(224, 784), stream=stream0)
        buf232 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_23.run(buf231, buf232, 1792, 784, grid=grid(1792, 784), stream=stream0)
        del buf205
        del buf209
        del buf213
        del buf217
        del buf221
        del buf225
        del buf229
        del buf230
        del buf231
        # Topologically Sorted Source Nodes: [out_180], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf233, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg326_1
        del buf232
        buf234 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [out_181, out_182, out_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_33.run(buf234, buf233, arg327_1, arg328_1, arg329_1, arg330_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        del buf233
        # Topologically Sorted Source Nodes: [out_184], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, arg331_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 448, 28, 28), (351232, 1, 12544, 448))
        del arg331_1
        buf236 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [out_185, out_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_34.run(buf236, arg332_1, arg333_1, arg334_1, arg335_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        buf237 = empty_strided_cuda((56, 56, 3, 3), (504, 1, 168, 56), torch.float32)
        # Topologically Sorted Source Nodes: [sp_645], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg336_1, buf237, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg336_1
        # Topologically Sorted Source Nodes: [sp_645], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(reinterpret_tensor(buf236, (8, 56, 28, 28), (351232, 1, 12544, 448), 0), buf237, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf239 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [sp_649], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg341_1, buf239, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg341_1
        # Topologically Sorted Source Nodes: [sp_649], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(reinterpret_tensor(buf236, (8, 56, 28, 28), (351232, 1, 12544, 448), 56), buf239, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf241 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [sp_653], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg346_1, buf241, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg346_1
        # Topologically Sorted Source Nodes: [sp_653], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(reinterpret_tensor(buf236, (8, 56, 28, 28), (351232, 1, 12544, 448), 112), buf241, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf243 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [sp_657], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg351_1, buf243, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg351_1
        # Topologically Sorted Source Nodes: [sp_657], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(reinterpret_tensor(buf236, (8, 56, 28, 28), (351232, 1, 12544, 448), 168), buf243, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf245 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [sp_661], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg356_1, buf245, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg356_1
        # Topologically Sorted Source Nodes: [sp_661], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(reinterpret_tensor(buf236, (8, 56, 28, 28), (351232, 1, 12544, 448), 224), buf245, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf247 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [sp_665], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg361_1, buf247, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg361_1
        # Topologically Sorted Source Nodes: [sp_665], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(reinterpret_tensor(buf236, (8, 56, 28, 28), (351232, 1, 12544, 448), 280), buf247, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf249 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [sp_669], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg366_1, buf249, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg366_1
        # Topologically Sorted Source Nodes: [sp_669], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(reinterpret_tensor(buf236, (8, 56, 28, 28), (351232, 1, 12544, 448), 336), buf249, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf259 = empty_strided_cuda((8, 448, 14, 14), (87808, 196, 14, 1), torch.float32)
        buf251 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_6], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_36.run(buf236, buf251, 1568, 56, grid=grid(1568, 56), stream=stream0)
        del buf236
        buf252 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_646, sp_647], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf238, arg337_1, arg338_1, arg339_1, arg340_1, buf252, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        del buf238
        buf253 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_650, sp_651], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf240, arg342_1, arg343_1, arg344_1, arg345_1, buf253, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        del buf240
        buf254 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_654, sp_655], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf242, arg347_1, arg348_1, arg349_1, arg350_1, buf254, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        del buf242
        buf255 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_658, sp_659], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf244, arg352_1, arg353_1, arg354_1, arg355_1, buf255, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        del buf244
        buf256 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_662, sp_663], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf246, arg357_1, arg358_1, arg359_1, arg360_1, buf256, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        del buf246
        buf257 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        # Topologically Sorted Source Nodes: [sp_666, sp_667], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf248, arg362_1, arg363_1, arg364_1, arg365_1, buf257, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del buf248
        buf258 = reinterpret_tensor(buf259, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_670, sp_671], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf250, arg367_1, arg368_1, arg369_1, arg370_1, buf258, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        del arg370_1
        del buf250
        buf260 = empty_strided_cuda((8, 448, 14, 14), (87808, 1, 6272, 448), torch.float32)
        # Topologically Sorted Source Nodes: [out_188], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf259, buf260, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf251
        del buf252
        del buf253
        del buf254
        del buf255
        del buf256
        del buf257
        del buf258
        del buf259
        # Topologically Sorted Source Nodes: [out_188], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, arg371_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg371_1
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf234, arg376_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg376_1
        del buf234
        buf263 = buf261; del buf261  # reuse
        buf264 = reinterpret_tensor(buf4, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [out_189, input_14, out_190, out_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_39.run(buf263, arg372_1, arg373_1, arg374_1, arg375_1, buf262, arg377_1, arg378_1, arg379_1, arg380_1, buf264, 1605632, grid=grid(1605632), stream=stream0)
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        del buf262
        del buf263
        # Topologically Sorted Source Nodes: [out_191, out_192], Original ATen: [aten.relu, aten.convolution]
        buf265 = extern_kernels.convolution(buf264, arg381_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 448, 14, 14), (87808, 1, 6272, 448))
        del arg381_1
        buf266 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [out_193, out_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf266, arg382_1, arg383_1, arg384_1, arg385_1, 702464, grid=grid(702464), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        buf267 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [sp_673], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg386_1, buf267, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg386_1
        # Topologically Sorted Source Nodes: [sp_673], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(reinterpret_tensor(buf266, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf295 = reinterpret_tensor(buf260, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf260  # reuse
        buf269 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_674, sp_675], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf268, arg387_1, arg388_1, arg389_1, arg390_1, buf269, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        buf270 = buf268; del buf268  # reuse
        # Topologically Sorted Source Nodes: [sp_676], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf269, buf266, buf270, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf271 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [sp_676, sp_677], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg391_1, buf271, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg391_1
        # Topologically Sorted Source Nodes: [sp_676, sp_677], Original ATen: [aten.add, aten.convolution]
        buf272 = extern_kernels.convolution(buf270, buf271, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf270
        buf273 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_678, sp_679], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf272, arg392_1, arg393_1, arg394_1, arg395_1, buf273, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        buf274 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [sp_680], Original ATen: [aten.add]
        triton_poi_fused_add_42.run(buf273, buf266, buf274, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf275 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [sp_680, sp_681], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg396_1, buf275, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg396_1
        # Topologically Sorted Source Nodes: [sp_680, sp_681], Original ATen: [aten.add, aten.convolution]
        buf276 = extern_kernels.convolution(buf274, buf275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf274
        buf277 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_682, sp_683], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf276, arg397_1, arg398_1, arg399_1, arg400_1, buf277, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg397_1
        del arg398_1
        del arg399_1
        del arg400_1
        buf278 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [sp_684], Original ATen: [aten.add]
        triton_poi_fused_add_43.run(buf277, buf266, buf278, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf279 = buf275; del buf275  # reuse
        # Topologically Sorted Source Nodes: [sp_684, sp_685], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg401_1, buf279, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg401_1
        # Topologically Sorted Source Nodes: [sp_684, sp_685], Original ATen: [aten.add, aten.convolution]
        buf280 = extern_kernels.convolution(buf278, buf279, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf278
        buf281 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_686, sp_687], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf280, arg402_1, arg403_1, arg404_1, arg405_1, buf281, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        buf282 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [sp_688], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf281, buf266, buf282, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf283 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [sp_688, sp_689], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg406_1, buf283, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg406_1
        # Topologically Sorted Source Nodes: [sp_688, sp_689], Original ATen: [aten.add, aten.convolution]
        buf284 = extern_kernels.convolution(buf282, buf283, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf282
        buf285 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_690, sp_691], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf284, arg407_1, arg408_1, arg409_1, arg410_1, buf285, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        del arg410_1
        buf286 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [sp_692], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf285, buf266, buf286, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf287 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [sp_692, sp_693], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg411_1, buf287, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg411_1
        # Topologically Sorted Source Nodes: [sp_692, sp_693], Original ATen: [aten.add, aten.convolution]
        buf288 = extern_kernels.convolution(buf286, buf287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf286
        buf289 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        # Topologically Sorted Source Nodes: [sp_694, sp_695], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf288, arg412_1, arg413_1, arg414_1, arg415_1, buf289, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        buf290 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [sp_696], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(buf289, buf266, buf290, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf291 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [sp_696, sp_697], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg416_1, buf291, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg416_1
        # Topologically Sorted Source Nodes: [sp_696, sp_697], Original ATen: [aten.add, aten.convolution]
        buf292 = extern_kernels.convolution(buf290, buf291, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf290
        buf293 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_698, sp_699], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf292, arg417_1, arg418_1, arg419_1, arg420_1, buf293, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg417_1
        del arg418_1
        del arg419_1
        del arg420_1
        del buf292
        buf294 = reinterpret_tensor(buf295, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Topologically Sorted Source Nodes: [out_195], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf266, buf294, 448, 196, grid=grid(448, 196), stream=stream0)
        buf296 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [out_196], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf295, buf296, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf269
        del buf273
        del buf277
        del buf281
        del buf285
        del buf289
        del buf293
        del buf294
        del buf295
        # Topologically Sorted Source Nodes: [out_196], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, arg421_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg421_1
        buf298 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [out_197, out_198, out_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf298, buf297, arg422_1, arg423_1, arg424_1, arg425_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        del buf297
        # Topologically Sorted Source Nodes: [out_200], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, arg426_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 448, 14, 14), (87808, 1, 6272, 448))
        del arg426_1
        buf300 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [out_201, out_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf300, arg427_1, arg428_1, arg429_1, arg430_1, 702464, grid=grid(702464), stream=stream0)
        del arg427_1
        del arg428_1
        del arg429_1
        del arg430_1
        buf301 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [sp_701], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg431_1, buf301, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg431_1
        # Topologically Sorted Source Nodes: [sp_701], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(reinterpret_tensor(buf300, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf301, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf329 = reinterpret_tensor(buf296, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf296  # reuse
        buf303 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_702, sp_703], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf302, arg432_1, arg433_1, arg434_1, arg435_1, buf303, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg432_1
        del arg433_1
        del arg434_1
        del arg435_1
        buf304 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [sp_704], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf303, buf300, buf304, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf305 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [sp_704, sp_705], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg436_1, buf305, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg436_1
        # Topologically Sorted Source Nodes: [sp_704, sp_705], Original ATen: [aten.add, aten.convolution]
        buf306 = extern_kernels.convolution(buf304, buf305, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf304
        buf307 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_706, sp_707], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf306, arg437_1, arg438_1, arg439_1, arg440_1, buf307, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg437_1
        del arg438_1
        del arg439_1
        del arg440_1
        buf308 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [sp_708], Original ATen: [aten.add]
        triton_poi_fused_add_42.run(buf307, buf300, buf308, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf309 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [sp_708, sp_709], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg441_1, buf309, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg441_1
        # Topologically Sorted Source Nodes: [sp_708, sp_709], Original ATen: [aten.add, aten.convolution]
        buf310 = extern_kernels.convolution(buf308, buf309, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf308
        buf311 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_710, sp_711], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf310, arg442_1, arg443_1, arg444_1, arg445_1, buf311, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        buf312 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [sp_712], Original ATen: [aten.add]
        triton_poi_fused_add_43.run(buf311, buf300, buf312, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf313 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [sp_712, sp_713], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg446_1, buf313, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg446_1
        # Topologically Sorted Source Nodes: [sp_712, sp_713], Original ATen: [aten.add, aten.convolution]
        buf314 = extern_kernels.convolution(buf312, buf313, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf312
        buf315 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_714, sp_715], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf314, arg447_1, arg448_1, arg449_1, arg450_1, buf315, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        buf316 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [sp_716], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf315, buf300, buf316, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf317 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [sp_716, sp_717], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg451_1, buf317, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg451_1
        # Topologically Sorted Source Nodes: [sp_716, sp_717], Original ATen: [aten.add, aten.convolution]
        buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf316
        buf319 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_718, sp_719], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf318, arg452_1, arg453_1, arg454_1, arg455_1, buf319, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        buf320 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [sp_720], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf319, buf300, buf320, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf321 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [sp_720, sp_721], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg456_1, buf321, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg456_1
        # Topologically Sorted Source Nodes: [sp_720, sp_721], Original ATen: [aten.add, aten.convolution]
        buf322 = extern_kernels.convolution(buf320, buf321, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf320
        buf323 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        # Topologically Sorted Source Nodes: [sp_722, sp_723], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf322, arg457_1, arg458_1, arg459_1, arg460_1, buf323, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg457_1
        del arg458_1
        del arg459_1
        del arg460_1
        buf324 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [sp_724], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(buf323, buf300, buf324, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf325 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [sp_724, sp_725], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg461_1, buf325, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg461_1
        # Topologically Sorted Source Nodes: [sp_724, sp_725], Original ATen: [aten.add, aten.convolution]
        buf326 = extern_kernels.convolution(buf324, buf325, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf324
        buf327 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_726, sp_727], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf326, arg462_1, arg463_1, arg464_1, arg465_1, buf327, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg462_1
        del arg463_1
        del arg464_1
        del arg465_1
        del buf326
        buf328 = reinterpret_tensor(buf329, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Topologically Sorted Source Nodes: [out_203], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf300, buf328, 448, 196, grid=grid(448, 196), stream=stream0)
        buf330 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [out_204], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf329, buf330, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf303
        del buf307
        del buf311
        del buf315
        del buf319
        del buf323
        del buf327
        del buf328
        del buf329
        # Topologically Sorted Source Nodes: [out_204], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, arg466_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg466_1
        buf332 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [out_205, out_206, out_207], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf332, buf331, arg467_1, arg468_1, arg469_1, arg470_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg467_1
        del arg468_1
        del arg469_1
        del arg470_1
        del buf331
        # Topologically Sorted Source Nodes: [out_208], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, arg471_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 448, 14, 14), (87808, 1, 6272, 448))
        del arg471_1
        buf334 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [out_209, out_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf334, arg472_1, arg473_1, arg474_1, arg475_1, 702464, grid=grid(702464), stream=stream0)
        del arg472_1
        del arg473_1
        del arg474_1
        del arg475_1
        buf335 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [sp_729], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg476_1, buf335, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg476_1
        # Topologically Sorted Source Nodes: [sp_729], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(reinterpret_tensor(buf334, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf335, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf363 = reinterpret_tensor(buf330, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf330  # reuse
        buf337 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_730, sp_731], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf336, arg477_1, arg478_1, arg479_1, arg480_1, buf337, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg477_1
        del arg478_1
        del arg479_1
        del arg480_1
        buf338 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [sp_732], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf337, buf334, buf338, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf339 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [sp_732, sp_733], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg481_1, buf339, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg481_1
        # Topologically Sorted Source Nodes: [sp_732, sp_733], Original ATen: [aten.add, aten.convolution]
        buf340 = extern_kernels.convolution(buf338, buf339, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf338
        buf341 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_734, sp_735], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf340, arg482_1, arg483_1, arg484_1, arg485_1, buf341, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg482_1
        del arg483_1
        del arg484_1
        del arg485_1
        buf342 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [sp_736], Original ATen: [aten.add]
        triton_poi_fused_add_42.run(buf341, buf334, buf342, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf343 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [sp_736, sp_737], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg486_1, buf343, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg486_1
        # Topologically Sorted Source Nodes: [sp_736, sp_737], Original ATen: [aten.add, aten.convolution]
        buf344 = extern_kernels.convolution(buf342, buf343, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf342
        buf345 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_738, sp_739], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf344, arg487_1, arg488_1, arg489_1, arg490_1, buf345, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg487_1
        del arg488_1
        del arg489_1
        del arg490_1
        buf346 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [sp_740], Original ATen: [aten.add]
        triton_poi_fused_add_43.run(buf345, buf334, buf346, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf347 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [sp_740, sp_741], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg491_1, buf347, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg491_1
        # Topologically Sorted Source Nodes: [sp_740, sp_741], Original ATen: [aten.add, aten.convolution]
        buf348 = extern_kernels.convolution(buf346, buf347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf346
        buf349 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_742, sp_743], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf348, arg492_1, arg493_1, arg494_1, arg495_1, buf349, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg492_1
        del arg493_1
        del arg494_1
        del arg495_1
        buf350 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [sp_744], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf349, buf334, buf350, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf351 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [sp_744, sp_745], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg496_1, buf351, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg496_1
        # Topologically Sorted Source Nodes: [sp_744, sp_745], Original ATen: [aten.add, aten.convolution]
        buf352 = extern_kernels.convolution(buf350, buf351, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf350
        buf353 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_746, sp_747], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf352, arg497_1, arg498_1, arg499_1, arg500_1, buf353, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg497_1
        del arg498_1
        del arg499_1
        del arg500_1
        buf354 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [sp_748], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf353, buf334, buf354, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf355 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [sp_748, sp_749], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg501_1, buf355, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg501_1
        # Topologically Sorted Source Nodes: [sp_748, sp_749], Original ATen: [aten.add, aten.convolution]
        buf356 = extern_kernels.convolution(buf354, buf355, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf354
        buf357 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        # Topologically Sorted Source Nodes: [sp_750, sp_751], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf356, arg502_1, arg503_1, arg504_1, arg505_1, buf357, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg502_1
        del arg503_1
        del arg504_1
        del arg505_1
        buf358 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [sp_752], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(buf357, buf334, buf358, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf359 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [sp_752, sp_753], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg506_1, buf359, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg506_1
        # Topologically Sorted Source Nodes: [sp_752, sp_753], Original ATen: [aten.add, aten.convolution]
        buf360 = extern_kernels.convolution(buf358, buf359, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf358
        buf361 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_754, sp_755], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf360, arg507_1, arg508_1, arg509_1, arg510_1, buf361, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg507_1
        del arg508_1
        del arg509_1
        del arg510_1
        del buf360
        buf362 = reinterpret_tensor(buf363, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Topologically Sorted Source Nodes: [out_211], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf334, buf362, 448, 196, grid=grid(448, 196), stream=stream0)
        buf364 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [out_212], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf363, buf364, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf337
        del buf341
        del buf345
        del buf349
        del buf353
        del buf357
        del buf361
        del buf362
        del buf363
        # Topologically Sorted Source Nodes: [out_212], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, arg511_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg511_1
        buf366 = buf332; del buf332  # reuse
        # Topologically Sorted Source Nodes: [out_213, out_214, out_215], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf366, buf365, arg512_1, arg513_1, arg514_1, arg515_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg512_1
        del arg513_1
        del arg514_1
        del arg515_1
        del buf365
        # Topologically Sorted Source Nodes: [out_216], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, arg516_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 448, 14, 14), (87808, 1, 6272, 448))
        del arg516_1
        buf368 = buf367; del buf367  # reuse
        # Topologically Sorted Source Nodes: [out_217, out_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf368, arg517_1, arg518_1, arg519_1, arg520_1, 702464, grid=grid(702464), stream=stream0)
        del arg517_1
        del arg518_1
        del arg519_1
        del arg520_1
        buf369 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [sp_757], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg521_1, buf369, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg521_1
        # Topologically Sorted Source Nodes: [sp_757], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(reinterpret_tensor(buf368, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf369, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf397 = reinterpret_tensor(buf364, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf364  # reuse
        buf371 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_758, sp_759], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf370, arg522_1, arg523_1, arg524_1, arg525_1, buf371, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg522_1
        del arg523_1
        del arg524_1
        del arg525_1
        buf372 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [sp_760], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf371, buf368, buf372, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf373 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [sp_760, sp_761], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg526_1, buf373, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg526_1
        # Topologically Sorted Source Nodes: [sp_760, sp_761], Original ATen: [aten.add, aten.convolution]
        buf374 = extern_kernels.convolution(buf372, buf373, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf372
        buf375 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_762, sp_763], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf374, arg527_1, arg528_1, arg529_1, arg530_1, buf375, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg527_1
        del arg528_1
        del arg529_1
        del arg530_1
        buf376 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [sp_764], Original ATen: [aten.add]
        triton_poi_fused_add_42.run(buf375, buf368, buf376, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf377 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [sp_764, sp_765], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg531_1, buf377, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg531_1
        # Topologically Sorted Source Nodes: [sp_764, sp_765], Original ATen: [aten.add, aten.convolution]
        buf378 = extern_kernels.convolution(buf376, buf377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf376
        buf379 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_766, sp_767], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf378, arg532_1, arg533_1, arg534_1, arg535_1, buf379, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg532_1
        del arg533_1
        del arg534_1
        del arg535_1
        buf380 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [sp_768], Original ATen: [aten.add]
        triton_poi_fused_add_43.run(buf379, buf368, buf380, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf381 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [sp_768, sp_769], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg536_1, buf381, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg536_1
        # Topologically Sorted Source Nodes: [sp_768, sp_769], Original ATen: [aten.add, aten.convolution]
        buf382 = extern_kernels.convolution(buf380, buf381, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf380
        buf383 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_770, sp_771], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf382, arg537_1, arg538_1, arg539_1, arg540_1, buf383, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg537_1
        del arg538_1
        del arg539_1
        del arg540_1
        buf384 = buf382; del buf382  # reuse
        # Topologically Sorted Source Nodes: [sp_772], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf383, buf368, buf384, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf385 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [sp_772, sp_773], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg541_1, buf385, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg541_1
        # Topologically Sorted Source Nodes: [sp_772, sp_773], Original ATen: [aten.add, aten.convolution]
        buf386 = extern_kernels.convolution(buf384, buf385, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf384
        buf387 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_774, sp_775], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf386, arg542_1, arg543_1, arg544_1, arg545_1, buf387, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg542_1
        del arg543_1
        del arg544_1
        del arg545_1
        buf388 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [sp_776], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf387, buf368, buf388, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf389 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [sp_776, sp_777], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg546_1, buf389, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg546_1
        # Topologically Sorted Source Nodes: [sp_776, sp_777], Original ATen: [aten.add, aten.convolution]
        buf390 = extern_kernels.convolution(buf388, buf389, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf388
        buf391 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        # Topologically Sorted Source Nodes: [sp_778, sp_779], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf390, arg547_1, arg548_1, arg549_1, arg550_1, buf391, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg547_1
        del arg548_1
        del arg549_1
        del arg550_1
        buf392 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [sp_780], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(buf391, buf368, buf392, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf393 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [sp_780, sp_781], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg551_1, buf393, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg551_1
        # Topologically Sorted Source Nodes: [sp_780, sp_781], Original ATen: [aten.add, aten.convolution]
        buf394 = extern_kernels.convolution(buf392, buf393, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf392
        buf395 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_782, sp_783], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf394, arg552_1, arg553_1, arg554_1, arg555_1, buf395, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg552_1
        del arg553_1
        del arg554_1
        del arg555_1
        del buf394
        buf396 = reinterpret_tensor(buf397, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Topologically Sorted Source Nodes: [out_219], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf368, buf396, 448, 196, grid=grid(448, 196), stream=stream0)
        buf398 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [out_220], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf397, buf398, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf371
        del buf375
        del buf379
        del buf383
        del buf387
        del buf391
        del buf395
        del buf396
        del buf397
        # Topologically Sorted Source Nodes: [out_220], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, arg556_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg556_1
        buf400 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [out_221, out_222, out_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf400, buf399, arg557_1, arg558_1, arg559_1, arg560_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg557_1
        del arg558_1
        del arg559_1
        del arg560_1
        del buf399
        # Topologically Sorted Source Nodes: [out_224], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, arg561_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 448, 14, 14), (87808, 1, 6272, 448))
        del arg561_1
        buf402 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [out_225, out_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_40.run(buf402, arg562_1, arg563_1, arg564_1, arg565_1, 702464, grid=grid(702464), stream=stream0)
        del arg562_1
        del arg563_1
        del arg564_1
        del arg565_1
        buf403 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [sp_785], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_35.run(arg566_1, buf403, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg566_1
        # Topologically Sorted Source Nodes: [sp_785], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(reinterpret_tensor(buf402, (8, 56, 14, 14), (87808, 1, 6272, 448), 0), buf403, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 56, 14, 14), (10976, 1, 784, 56))
        buf431 = reinterpret_tensor(buf398, (8, 448, 14, 14), (87808, 196, 14, 1), 0); del buf398  # reuse
        buf405 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_786, sp_787], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf404, arg567_1, arg568_1, arg569_1, arg570_1, buf405, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg567_1
        del arg568_1
        del arg569_1
        del arg570_1
        buf406 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [sp_788], Original ATen: [aten.add]
        triton_poi_fused_add_41.run(buf405, buf402, buf406, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf407 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [sp_788, sp_789], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg571_1, buf407, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg571_1
        # Topologically Sorted Source Nodes: [sp_788, sp_789], Original ATen: [aten.add, aten.convolution]
        buf408 = extern_kernels.convolution(buf406, buf407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf406
        buf409 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_790, sp_791], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf408, arg572_1, arg573_1, arg574_1, arg575_1, buf409, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg572_1
        del arg573_1
        del arg574_1
        del arg575_1
        buf410 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [sp_792], Original ATen: [aten.add]
        triton_poi_fused_add_42.run(buf409, buf402, buf410, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf411 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [sp_792, sp_793], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg576_1, buf411, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg576_1
        # Topologically Sorted Source Nodes: [sp_792, sp_793], Original ATen: [aten.add, aten.convolution]
        buf412 = extern_kernels.convolution(buf410, buf411, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf410
        buf413 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_794, sp_795], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf412, arg577_1, arg578_1, arg579_1, arg580_1, buf413, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg577_1
        del arg578_1
        del arg579_1
        del arg580_1
        buf414 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [sp_796], Original ATen: [aten.add]
        triton_poi_fused_add_43.run(buf413, buf402, buf414, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf415 = buf411; del buf411  # reuse
        # Topologically Sorted Source Nodes: [sp_796, sp_797], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg581_1, buf415, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg581_1
        # Topologically Sorted Source Nodes: [sp_796, sp_797], Original ATen: [aten.add, aten.convolution]
        buf416 = extern_kernels.convolution(buf414, buf415, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf414
        buf417 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_798, sp_799], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf416, arg582_1, arg583_1, arg584_1, arg585_1, buf417, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg582_1
        del arg583_1
        del arg584_1
        del arg585_1
        buf418 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [sp_800], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf417, buf402, buf418, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf419 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [sp_800, sp_801], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg586_1, buf419, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg586_1
        # Topologically Sorted Source Nodes: [sp_800, sp_801], Original ATen: [aten.add, aten.convolution]
        buf420 = extern_kernels.convolution(buf418, buf419, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf420, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf418
        buf421 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 43904)  # alias
        # Topologically Sorted Source Nodes: [sp_802, sp_803], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf420, arg587_1, arg588_1, arg589_1, arg590_1, buf421, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg587_1
        del arg588_1
        del arg589_1
        del arg590_1
        buf422 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [sp_804], Original ATen: [aten.add]
        triton_poi_fused_add_45.run(buf421, buf402, buf422, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf423 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [sp_804, sp_805], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg591_1, buf423, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg591_1
        # Topologically Sorted Source Nodes: [sp_804, sp_805], Original ATen: [aten.add, aten.convolution]
        buf424 = extern_kernels.convolution(buf422, buf423, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf422
        buf425 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 54880)  # alias
        # Topologically Sorted Source Nodes: [sp_806, sp_807], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf424, arg592_1, arg593_1, arg594_1, arg595_1, buf425, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg592_1
        del arg593_1
        del arg594_1
        del arg595_1
        buf426 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [sp_808], Original ATen: [aten.add]
        triton_poi_fused_add_46.run(buf425, buf402, buf426, 1568, 56, grid=grid(1568, 56), stream=stream0)
        buf427 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [sp_808, sp_809], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_35.run(arg596_1, buf427, 3136, 9, grid=grid(3136, 9), stream=stream0)
        del arg596_1
        # Topologically Sorted Source Nodes: [sp_808, sp_809], Original ATen: [aten.add, aten.convolution]
        buf428 = extern_kernels.convolution(buf426, buf427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (8, 56, 14, 14), (10976, 1, 784, 56))
        del buf426
        del buf427
        buf429 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 65856)  # alias
        # Topologically Sorted Source Nodes: [sp_810, sp_811], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf428, arg597_1, arg598_1, arg599_1, arg600_1, buf429, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg597_1
        del arg598_1
        del arg599_1
        del arg600_1
        del buf428
        buf430 = reinterpret_tensor(buf431, (8, 56, 14, 14), (87808, 196, 14, 1), 76832)  # alias
        # Topologically Sorted Source Nodes: [out_227], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf402, buf430, 448, 196, grid=grid(448, 196), stream=stream0)
        buf432 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [out_228], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf431, buf432, 3584, 196, grid=grid(3584, 196), stream=stream0)
        del buf405
        del buf409
        del buf413
        del buf417
        del buf421
        del buf425
        del buf429
        del buf430
        del buf431
        # Topologically Sorted Source Nodes: [out_228], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, arg601_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg601_1
        del buf432
        buf434 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [out_229, out_230, out_231], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_48.run(buf434, buf433, arg602_1, arg603_1, arg604_1, arg605_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg602_1
        del arg603_1
        del arg604_1
        del arg605_1
        del buf433
        # Topologically Sorted Source Nodes: [out_232], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, arg606_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (8, 896, 14, 14), (175616, 1, 12544, 896))
        del arg606_1
        buf436 = buf435; del buf435  # reuse
        # Topologically Sorted Source Nodes: [out_233, out_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf436, arg607_1, arg608_1, arg609_1, arg610_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg607_1
        del arg608_1
        del arg609_1
        del arg610_1
        buf437 = empty_strided_cuda((112, 112, 3, 3), (1008, 1, 336, 112), torch.float32)
        # Topologically Sorted Source Nodes: [sp_813], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg611_1, buf437, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg611_1
        # Topologically Sorted Source Nodes: [sp_813], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(reinterpret_tensor(buf436, (8, 112, 14, 14), (175616, 1, 12544, 896), 0), buf437, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf439 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [sp_817], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg616_1, buf439, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg616_1
        # Topologically Sorted Source Nodes: [sp_817], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(reinterpret_tensor(buf436, (8, 112, 14, 14), (175616, 1, 12544, 896), 112), buf439, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf441 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [sp_821], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg621_1, buf441, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg621_1
        # Topologically Sorted Source Nodes: [sp_821], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(reinterpret_tensor(buf436, (8, 112, 14, 14), (175616, 1, 12544, 896), 224), buf441, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf443 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [sp_825], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg626_1, buf443, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg626_1
        # Topologically Sorted Source Nodes: [sp_825], Original ATen: [aten.convolution]
        buf444 = extern_kernels.convolution(reinterpret_tensor(buf436, (8, 112, 14, 14), (175616, 1, 12544, 896), 336), buf443, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf444, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf445 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [sp_829], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg631_1, buf445, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg631_1
        # Topologically Sorted Source Nodes: [sp_829], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(reinterpret_tensor(buf436, (8, 112, 14, 14), (175616, 1, 12544, 896), 448), buf445, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf447 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [sp_833], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg636_1, buf447, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg636_1
        # Topologically Sorted Source Nodes: [sp_833], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(reinterpret_tensor(buf436, (8, 112, 14, 14), (175616, 1, 12544, 896), 560), buf447, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf449 = buf447; del buf447  # reuse
        # Topologically Sorted Source Nodes: [sp_837], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg641_1, buf449, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg641_1
        # Topologically Sorted Source Nodes: [sp_837], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(reinterpret_tensor(buf436, (8, 112, 14, 14), (175616, 1, 12544, 896), 672), buf449, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf459 = reinterpret_tensor(buf96, (8, 896, 7, 7), (43904, 49, 7, 1), 0); del buf96  # reuse
        buf451 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Topologically Sorted Source Nodes: [avg_pool2d_7], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_51.run(buf436, buf451, 392, 112, grid=grid(392, 112), stream=stream0)
        del buf436
        buf452 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_814, sp_815], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf438, arg612_1, arg613_1, arg614_1, arg615_1, buf452, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg612_1
        del arg613_1
        del arg614_1
        del arg615_1
        del buf438
        buf453 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        # Topologically Sorted Source Nodes: [sp_818, sp_819], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf440, arg617_1, arg618_1, arg619_1, arg620_1, buf453, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg617_1
        del arg618_1
        del arg619_1
        del arg620_1
        del buf440
        buf454 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_822, sp_823], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf442, arg622_1, arg623_1, arg624_1, arg625_1, buf454, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg622_1
        del arg623_1
        del arg624_1
        del arg625_1
        del buf442
        buf455 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        # Topologically Sorted Source Nodes: [sp_826, sp_827], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf444, arg627_1, arg628_1, arg629_1, arg630_1, buf455, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg627_1
        del arg628_1
        del arg629_1
        del arg630_1
        del buf444
        buf456 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_830, sp_831], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf446, arg632_1, arg633_1, arg634_1, arg635_1, buf456, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg632_1
        del arg633_1
        del arg634_1
        del arg635_1
        del buf446
        buf457 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        # Topologically Sorted Source Nodes: [sp_834, sp_835], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf448, arg637_1, arg638_1, arg639_1, arg640_1, buf457, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg637_1
        del arg638_1
        del arg639_1
        del arg640_1
        del buf448
        buf458 = reinterpret_tensor(buf459, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_838, sp_839], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf450, arg642_1, arg643_1, arg644_1, arg645_1, buf458, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg642_1
        del arg643_1
        del arg644_1
        del arg645_1
        del buf450
        buf460 = reinterpret_tensor(buf94, (8, 896, 7, 7), (43904, 1, 6272, 896), 0); del buf94  # reuse
        # Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf459, buf460, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf451
        del buf452
        del buf453
        del buf454
        del buf455
        del buf456
        del buf457
        del buf458
        del buf459
        # Topologically Sorted Source Nodes: [out_236], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, arg646_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg646_1
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf434, arg651_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg651_1
        del buf434
        buf463 = buf461; del buf461  # reuse
        buf464 = empty_strided_cuda((8, 2048, 7, 7), (100352, 1, 14336, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [out_237, input_16, out_238, out_239], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_54.run(buf463, arg647_1, arg648_1, arg649_1, arg650_1, buf462, arg652_1, arg653_1, arg654_1, arg655_1, buf464, 802816, grid=grid(802816), stream=stream0)
        del arg647_1
        del arg648_1
        del arg649_1
        del arg650_1
        del arg652_1
        del arg653_1
        del arg654_1
        del arg655_1
        del buf462
        del buf463
        # Topologically Sorted Source Nodes: [out_239, out_240], Original ATen: [aten.relu, aten.convolution]
        buf465 = extern_kernels.convolution(buf464, arg656_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (8, 896, 7, 7), (43904, 1, 6272, 896))
        del arg656_1
        buf466 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [out_241, out_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf466, arg657_1, arg658_1, arg659_1, arg660_1, 351232, grid=grid(351232), stream=stream0)
        del arg657_1
        del arg658_1
        del arg659_1
        del arg660_1
        buf467 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [sp_841], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg661_1, buf467, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg661_1
        # Topologically Sorted Source Nodes: [sp_841], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(reinterpret_tensor(buf466, (8, 112, 7, 7), (43904, 1, 6272, 896), 0), buf467, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf468, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf495 = reinterpret_tensor(buf460, (8, 896, 7, 7), (43904, 49, 7, 1), 0); del buf460  # reuse
        buf469 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_842, sp_843], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf468, arg662_1, arg663_1, arg664_1, arg665_1, buf469, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg662_1
        del arg663_1
        del arg664_1
        del arg665_1
        buf470 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [sp_844], Original ATen: [aten.add]
        triton_poi_fused_add_56.run(buf469, buf466, buf470, 392, 112, grid=grid(392, 112), stream=stream0)
        buf471 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [sp_844, sp_845], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg666_1, buf471, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg666_1
        # Topologically Sorted Source Nodes: [sp_844, sp_845], Original ATen: [aten.add, aten.convolution]
        buf472 = extern_kernels.convolution(buf470, buf471, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf470
        buf473 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        # Topologically Sorted Source Nodes: [sp_846, sp_847], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf472, arg667_1, arg668_1, arg669_1, arg670_1, buf473, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg667_1
        del arg668_1
        del arg669_1
        del arg670_1
        buf474 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [sp_848], Original ATen: [aten.add]
        triton_poi_fused_add_57.run(buf473, buf466, buf474, 392, 112, grid=grid(392, 112), stream=stream0)
        buf475 = buf471; del buf471  # reuse
        # Topologically Sorted Source Nodes: [sp_848, sp_849], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg671_1, buf475, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg671_1
        # Topologically Sorted Source Nodes: [sp_848, sp_849], Original ATen: [aten.add, aten.convolution]
        buf476 = extern_kernels.convolution(buf474, buf475, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf474
        buf477 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_850, sp_851], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf476, arg672_1, arg673_1, arg674_1, arg675_1, buf477, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg672_1
        del arg673_1
        del arg674_1
        del arg675_1
        buf478 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [sp_852], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf477, buf466, buf478, 392, 112, grid=grid(392, 112), stream=stream0)
        buf479 = buf475; del buf475  # reuse
        # Topologically Sorted Source Nodes: [sp_852, sp_853], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg676_1, buf479, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg676_1
        # Topologically Sorted Source Nodes: [sp_852, sp_853], Original ATen: [aten.add, aten.convolution]
        buf480 = extern_kernels.convolution(buf478, buf479, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf480, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf478
        buf481 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        # Topologically Sorted Source Nodes: [sp_854, sp_855], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf480, arg677_1, arg678_1, arg679_1, arg680_1, buf481, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg677_1
        del arg678_1
        del arg679_1
        del arg680_1
        buf482 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [sp_856], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf481, buf466, buf482, 392, 112, grid=grid(392, 112), stream=stream0)
        buf483 = buf479; del buf479  # reuse
        # Topologically Sorted Source Nodes: [sp_856, sp_857], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg681_1, buf483, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg681_1
        # Topologically Sorted Source Nodes: [sp_856, sp_857], Original ATen: [aten.add, aten.convolution]
        buf484 = extern_kernels.convolution(buf482, buf483, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf482
        buf485 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_858, sp_859], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf484, arg682_1, arg683_1, arg684_1, arg685_1, buf485, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg682_1
        del arg683_1
        del arg684_1
        del arg685_1
        buf486 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [sp_860], Original ATen: [aten.add]
        triton_poi_fused_add_60.run(buf485, buf466, buf486, 392, 112, grid=grid(392, 112), stream=stream0)
        buf487 = buf483; del buf483  # reuse
        # Topologically Sorted Source Nodes: [sp_860, sp_861], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg686_1, buf487, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg686_1
        # Topologically Sorted Source Nodes: [sp_860, sp_861], Original ATen: [aten.add, aten.convolution]
        buf488 = extern_kernels.convolution(buf486, buf487, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf486
        buf489 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        # Topologically Sorted Source Nodes: [sp_862, sp_863], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf488, arg687_1, arg688_1, arg689_1, arg690_1, buf489, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg687_1
        del arg688_1
        del arg689_1
        del arg690_1
        buf490 = buf488; del buf488  # reuse
        # Topologically Sorted Source Nodes: [sp_864], Original ATen: [aten.add]
        triton_poi_fused_add_61.run(buf489, buf466, buf490, 392, 112, grid=grid(392, 112), stream=stream0)
        buf491 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [sp_864, sp_865], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg691_1, buf491, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg691_1
        # Topologically Sorted Source Nodes: [sp_864, sp_865], Original ATen: [aten.add, aten.convolution]
        buf492 = extern_kernels.convolution(buf490, buf491, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf490
        buf493 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_866, sp_867], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf492, arg692_1, arg693_1, arg694_1, arg695_1, buf493, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg692_1
        del arg693_1
        del arg694_1
        del arg695_1
        del buf492
        buf494 = reinterpret_tensor(buf495, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Topologically Sorted Source Nodes: [out_243], Original ATen: [aten.cat]
        triton_poi_fused_cat_62.run(buf466, buf494, 896, 49, grid=grid(896, 49), stream=stream0)
        buf496 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [out_244], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf495, buf496, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf469
        del buf473
        del buf477
        del buf481
        del buf485
        del buf489
        del buf493
        del buf494
        del buf495
        # Topologically Sorted Source Nodes: [out_244], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, arg696_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg696_1
        buf498 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [out_245, out_246, out_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_63.run(buf498, buf497, arg697_1, arg698_1, arg699_1, arg700_1, 802816, grid=grid(802816), stream=stream0)
        del arg697_1
        del arg698_1
        del arg699_1
        del arg700_1
        del buf497
        # Topologically Sorted Source Nodes: [out_248], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, arg701_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (8, 896, 7, 7), (43904, 1, 6272, 896))
        del arg701_1
        buf500 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [out_249, out_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_55.run(buf500, arg702_1, arg703_1, arg704_1, arg705_1, 351232, grid=grid(351232), stream=stream0)
        del arg702_1
        del arg703_1
        del arg704_1
        del arg705_1
        buf501 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [sp_869], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_50.run(arg706_1, buf501, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg706_1
        # Topologically Sorted Source Nodes: [sp_869], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(reinterpret_tensor(buf500, (8, 112, 7, 7), (43904, 1, 6272, 896), 0), buf501, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (8, 112, 7, 7), (5488, 1, 784, 112))
        buf529 = reinterpret_tensor(buf496, (8, 896, 7, 7), (43904, 49, 7, 1), 0); del buf496  # reuse
        buf503 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [sp_870, sp_871], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf502, arg707_1, arg708_1, arg709_1, arg710_1, buf503, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg707_1
        del arg708_1
        del arg709_1
        del arg710_1
        buf504 = buf502; del buf502  # reuse
        # Topologically Sorted Source Nodes: [sp_872], Original ATen: [aten.add]
        triton_poi_fused_add_56.run(buf503, buf500, buf504, 392, 112, grid=grid(392, 112), stream=stream0)
        buf505 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [sp_872, sp_873], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg711_1, buf505, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg711_1
        # Topologically Sorted Source Nodes: [sp_872, sp_873], Original ATen: [aten.add, aten.convolution]
        buf506 = extern_kernels.convolution(buf504, buf505, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf504
        buf507 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 5488)  # alias
        # Topologically Sorted Source Nodes: [sp_874, sp_875], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf506, arg712_1, arg713_1, arg714_1, arg715_1, buf507, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg712_1
        del arg713_1
        del arg714_1
        del arg715_1
        buf508 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [sp_876], Original ATen: [aten.add]
        triton_poi_fused_add_57.run(buf507, buf500, buf508, 392, 112, grid=grid(392, 112), stream=stream0)
        buf509 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [sp_876, sp_877], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg716_1, buf509, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg716_1
        # Topologically Sorted Source Nodes: [sp_876, sp_877], Original ATen: [aten.add, aten.convolution]
        buf510 = extern_kernels.convolution(buf508, buf509, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf508
        buf511 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 10976)  # alias
        # Topologically Sorted Source Nodes: [sp_878, sp_879], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf510, arg717_1, arg718_1, arg719_1, arg720_1, buf511, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg717_1
        del arg718_1
        del arg719_1
        del arg720_1
        buf512 = buf510; del buf510  # reuse
        # Topologically Sorted Source Nodes: [sp_880], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf511, buf500, buf512, 392, 112, grid=grid(392, 112), stream=stream0)
        buf513 = buf509; del buf509  # reuse
        # Topologically Sorted Source Nodes: [sp_880, sp_881], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg721_1, buf513, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg721_1
        # Topologically Sorted Source Nodes: [sp_880, sp_881], Original ATen: [aten.add, aten.convolution]
        buf514 = extern_kernels.convolution(buf512, buf513, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf512
        buf515 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 16464)  # alias
        # Topologically Sorted Source Nodes: [sp_882, sp_883], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf514, arg722_1, arg723_1, arg724_1, arg725_1, buf515, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg722_1
        del arg723_1
        del arg724_1
        del arg725_1
        buf516 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [sp_884], Original ATen: [aten.add]
        triton_poi_fused_add_59.run(buf515, buf500, buf516, 392, 112, grid=grid(392, 112), stream=stream0)
        buf517 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [sp_884, sp_885], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg726_1, buf517, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg726_1
        # Topologically Sorted Source Nodes: [sp_884, sp_885], Original ATen: [aten.add, aten.convolution]
        buf518 = extern_kernels.convolution(buf516, buf517, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf518, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf516
        buf519 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 21952)  # alias
        # Topologically Sorted Source Nodes: [sp_886, sp_887], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf518, arg727_1, arg728_1, arg729_1, arg730_1, buf519, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg727_1
        del arg728_1
        del arg729_1
        del arg730_1
        buf520 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [sp_888], Original ATen: [aten.add]
        triton_poi_fused_add_60.run(buf519, buf500, buf520, 392, 112, grid=grid(392, 112), stream=stream0)
        buf521 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [sp_888, sp_889], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg731_1, buf521, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg731_1
        # Topologically Sorted Source Nodes: [sp_888, sp_889], Original ATen: [aten.add, aten.convolution]
        buf522 = extern_kernels.convolution(buf520, buf521, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf520
        buf523 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 27440)  # alias
        # Topologically Sorted Source Nodes: [sp_890, sp_891], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf522, arg732_1, arg733_1, arg734_1, arg735_1, buf523, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg732_1
        del arg733_1
        del arg734_1
        del arg735_1
        buf524 = buf522; del buf522  # reuse
        # Topologically Sorted Source Nodes: [sp_892], Original ATen: [aten.add]
        triton_poi_fused_add_61.run(buf523, buf500, buf524, 392, 112, grid=grid(392, 112), stream=stream0)
        buf525 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [sp_892, sp_893], Original ATen: [aten.add, aten.convolution]
        triton_poi_fused_convolution_50.run(arg736_1, buf525, 12544, 9, grid=grid(12544, 9), stream=stream0)
        del arg736_1
        # Topologically Sorted Source Nodes: [sp_892, sp_893], Original ATen: [aten.add, aten.convolution]
        buf526 = extern_kernels.convolution(buf524, buf525, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (8, 112, 7, 7), (5488, 1, 784, 112))
        del buf524
        del buf525
        buf527 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        # Topologically Sorted Source Nodes: [sp_894, sp_895], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf526, arg737_1, arg738_1, arg739_1, arg740_1, buf527, 896, 49, grid=grid(896, 49), stream=stream0)
        del arg737_1
        del arg738_1
        del arg739_1
        del arg740_1
        del buf526
        buf528 = reinterpret_tensor(buf529, (8, 112, 7, 7), (43904, 49, 7, 1), 38416)  # alias
        # Topologically Sorted Source Nodes: [out_251], Original ATen: [aten.cat]
        triton_poi_fused_cat_62.run(buf500, buf528, 896, 49, grid=grid(896, 49), stream=stream0)
        buf530 = buf500; del buf500  # reuse
        # Topologically Sorted Source Nodes: [out_252], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_53.run(buf529, buf530, 7168, 49, grid=grid(7168, 49), stream=stream0)
        del buf503
        del buf507
        del buf511
        del buf515
        del buf519
        del buf523
        del buf527
        del buf528
        del buf529
        # Topologically Sorted Source Nodes: [out_252], Original ATen: [aten.convolution]
        buf531 = extern_kernels.convolution(buf530, arg741_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf531, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg741_1
        del buf530
        buf533 = empty_strided_cuda((8, 2048, 1, 1), (2048, 1, 16384, 16384), torch.float32)
        # Topologically Sorted Source Nodes: [out_253, out_254, out_255, x_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_64.run(buf531, arg742_1, arg743_1, arg744_1, arg745_1, buf498, buf533, 16384, 49, grid=grid(16384), stream=stream0)
        del arg742_1
        del arg743_1
        del arg744_1
        del arg745_1
        del buf498
        del buf531
        buf534 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg747_1, reinterpret_tensor(buf533, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg746_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf534)
        del arg746_1
        del arg747_1
        del buf533
    return (buf534, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((112, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((112, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((112, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((14, 14, 3, 3), (126, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((224, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((224, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((28, 28, 3, 3), (252, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((512, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((448, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((448, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((56, 56, 3, 3), (504, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((1024, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((896, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg628_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg631_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg634_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg637_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg640_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg643_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg646_1 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg649_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg652_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg655_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((896, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg658_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg661_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg664_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg667_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg670_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg673_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg676_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg679_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg682_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg685_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg688_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg691_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg694_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg697_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg700_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((896, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg703_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg706_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg709_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg712_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg715_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg718_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg721_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg724_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg727_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg728_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg729_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg730_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg731_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg732_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg733_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg734_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg735_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg736_1 = rand_strided((112, 112, 3, 3), (1008, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg737_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg738_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg739_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg740_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg741_1 = rand_strided((2048, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg742_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg743_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg744_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg745_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg746_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg747_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2net50_14w_8s', benchmark_compiled_module)
