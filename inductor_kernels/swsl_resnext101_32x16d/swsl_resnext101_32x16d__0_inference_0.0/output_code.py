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
# Topologically Sorted Source Nodes: [x_337], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_337 => convolution_104
# Graph fragment:
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_337], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_337 => convolution_104
# Graph fragment:
#   %convolution_104 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_338, x_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_338 => add_242, mul_313, mul_314, sub_104
#   x_339 => relu_100
# Graph fragment:
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_833), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %unsqueeze_837), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_839), kwargs = {})
#   %relu_100 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_242,), kwargs = {})
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
# Topologically Sorted Source Nodes: [x_338, x_339, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_338 => add_242, mul_313, mul_314, sub_104
#   x_339 => relu_100
#   x_340 => _low_memory_max_pool2d_with_offsets_1
# Graph fragment:
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_833), kwargs = {})
#   %mul_313 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_313, %unsqueeze_837), kwargs = {})
#   %add_242 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_314, %unsqueeze_839), kwargs = {})
#   %relu_100 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_242,), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%relu_100, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/56/c56vfg6ss2wzfvvghyg5z6na4uwbhavv73wyrhvsfn7346rxepm5.py
# Topologically Sorted Source Nodes: [x_342, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_342 => add_244, mul_316, mul_317, sub_105
#   x_343 => relu_101
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_841), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_316, %unsqueeze_845), kwargs = {})
#   %add_244 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_317, %unsqueeze_847), kwargs = {})
#   %relu_101 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_244,), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_sahanp/rm/crmhp2m6d7xakgaakasxck4ygehyooteihsqouzcsca4cidskigc.py
# Topologically Sorted Source Nodes: [x_342, x_343, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_342 => add_244, mul_316, mul_317, sub_105
#   x_343 => relu_101
#   x_344 => convolution_106
# Graph fragment:
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_841), kwargs = {})
#   %mul_316 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_316, %unsqueeze_845), kwargs = {})
#   %add_244 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_317, %unsqueeze_847), kwargs = {})
#   %relu_101 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_244,), kwargs = {})
#   %convolution_106 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_101, %arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/wy/cwynv445atkqqaqzyurkdjwnwkncgooi2xz4y5mxo2s7fvw6d3ib.py
# Topologically Sorted Source Nodes: [x_348, input_10, x_349, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_250, mul_325, mul_326, sub_108
#   x_348 => add_248, mul_322, mul_323, sub_107
#   x_349 => add_251
#   x_350 => relu_103
# Graph fragment:
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_857), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_107, %unsqueeze_859), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_322, %unsqueeze_861), kwargs = {})
#   %add_248 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_323, %unsqueeze_863), kwargs = {})
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_865), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %unsqueeze_867), kwargs = {})
#   %mul_326 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_325, %unsqueeze_869), kwargs = {})
#   %add_250 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_326, %unsqueeze_871), kwargs = {})
#   %add_251 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_248, %add_250), kwargs = {})
#   %relu_103 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_251,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/2q/c2qazok5k43hzb3zdh4yum7p7qhh3vy4fokhc5mqzyzsk5zhwol3.py
# Topologically Sorted Source Nodes: [x_358, x_359, x_360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_358 => add_257, mul_334, mul_335, sub_111
#   x_359 => add_258
#   x_360 => relu_106
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_889), kwargs = {})
#   %mul_334 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_891), kwargs = {})
#   %mul_335 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_334, %unsqueeze_893), kwargs = {})
#   %add_257 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_335, %unsqueeze_895), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_257, %relu_103), kwargs = {})
#   %relu_106 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_258,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ro/crotcqfngshffysugylnjlllxf3tmud5pmyebohavresaqxhyiuf.py
# Topologically Sorted Source Nodes: [x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_372 => add_267, mul_346, mul_347, sub_115
#   x_373 => relu_110
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_921), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_346, %unsqueeze_925), kwargs = {})
#   %add_267 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_347, %unsqueeze_927), kwargs = {})
#   %relu_110 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_267,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
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


# kernel path: /tmp/torchinductor_sahanp/et/cetpez5zznmjr6a4dczceimjsd6qxgovy5hatei2gygoikeui3s4.py
# Topologically Sorted Source Nodes: [x_372, x_373, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_372 => add_267, mul_346, mul_347, sub_115
#   x_373 => relu_110
#   x_374 => convolution_116
# Graph fragment:
#   %sub_115 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_921), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_115, %unsqueeze_923), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_346, %unsqueeze_925), kwargs = {})
#   %add_267 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_347, %unsqueeze_927), kwargs = {})
#   %relu_110 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_267,), kwargs = {})
#   %convolution_116 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_110, %arg61_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
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


# kernel path: /tmp/torchinductor_sahanp/hq/chqg6jtbzztctxxxvdehrhvuwzumhyx7x4k5auvx7yoyd6opx44g.py
# Topologically Sorted Source Nodes: [x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_375 => add_269, mul_349, mul_350, sub_116
#   x_376 => relu_111
# Graph fragment:
#   %sub_116 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_929), kwargs = {})
#   %mul_349 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_116, %unsqueeze_931), kwargs = {})
#   %mul_350 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_349, %unsqueeze_933), kwargs = {})
#   %add_269 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_350, %unsqueeze_935), kwargs = {})
#   %relu_111 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_269,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_sahanp/yx/cyxrywmrat5ofuyuofbuw2jgrlrll77ftwxvxa4mvavn2ks4dzze.py
# Topologically Sorted Source Nodes: [x_378, input_12, x_379, x_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_12 => add_273, mul_355, mul_356, sub_118
#   x_378 => add_271, mul_352, mul_353, sub_117
#   x_379 => add_274
#   x_380 => relu_112
# Graph fragment:
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_937), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_939), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_941), kwargs = {})
#   %add_271 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_943), kwargs = {})
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_945), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_947), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_355, %unsqueeze_949), kwargs = {})
#   %add_273 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_356, %unsqueeze_951), kwargs = {})
#   %add_274 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_271, %add_273), kwargs = {})
#   %relu_112 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_274,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ar/carakkyxge7dholl5cqusvvtkxfswj6hl7r3lfhzaxobwnjo3eci.py
# Topologically Sorted Source Nodes: [x_388, x_389, x_390], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_388 => add_280, mul_364, mul_365, sub_121
#   x_389 => add_281
#   x_390 => relu_115
# Graph fragment:
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_121, %unsqueeze_969), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %unsqueeze_971), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_364, %unsqueeze_973), kwargs = {})
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_365, %unsqueeze_975), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_280, %relu_112), kwargs = {})
#   %relu_115 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_281,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/q3/cq3ldgsrxa2brar5xoyd5e4auhtjzn6okeap6jhyz6vtzvowncqs.py
# Topologically Sorted Source Nodes: [x_412, x_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_412 => add_297, mul_385, mul_386, sub_128
#   x_413 => relu_122
# Graph fragment:
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_1025), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %unsqueeze_1029), kwargs = {})
#   %add_297 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %unsqueeze_1031), kwargs = {})
#   %relu_122 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_297,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
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


# kernel path: /tmp/torchinductor_sahanp/gw/cgwpihgcyic3aqnclewryt6heu23osnsjhiwxgamlnv4f7vsg6o6.py
# Topologically Sorted Source Nodes: [x_412, x_413, x_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_412 => add_297, mul_385, mul_386, sub_128
#   x_413 => relu_122
#   x_414 => convolution_129
# Graph fragment:
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_1025), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_385, %unsqueeze_1029), kwargs = {})
#   %add_297 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_386, %unsqueeze_1031), kwargs = {})
#   %relu_122 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_297,), kwargs = {})
#   %convolution_129 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_122, %arg126_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/dg/cdgcpkshgre6ylop3mrztpsfviwbyzt4jqsitdl75t7kdi5twn2n.py
# Topologically Sorted Source Nodes: [x_415, x_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_415 => add_299, mul_388, mul_389, sub_129
#   x_416 => relu_123
# Graph fragment:
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_1033), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_389 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %unsqueeze_1037), kwargs = {})
#   %add_299 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_389, %unsqueeze_1039), kwargs = {})
#   %relu_123 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_299,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_sahanp/ik/cik7bc3rm7ysunrrvmo4pisrqm2gnkgxlvyz7xbwzsbvonjjpqli.py
# Topologically Sorted Source Nodes: [x_418, input_14, x_419, x_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_14 => add_303, mul_394, mul_395, sub_131
#   x_418 => add_301, mul_391, mul_392, sub_130
#   x_419 => add_304
#   x_420 => relu_124
# Graph fragment:
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_1041), kwargs = {})
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_130, %unsqueeze_1043), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_391, %unsqueeze_1045), kwargs = {})
#   %add_301 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_392, %unsqueeze_1047), kwargs = {})
#   %sub_131 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_131, %unsqueeze_1049), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_131, %unsqueeze_1051), kwargs = {})
#   %mul_395 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_394, %unsqueeze_1053), kwargs = {})
#   %add_303 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_395, %unsqueeze_1055), kwargs = {})
#   %add_304 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_301, %add_303), kwargs = {})
#   %relu_124 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_304,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/bh/cbhwqsmipxx7v4de3f2i2pillg2uzhcr6wteg2a5kg7tbu6g2ffr.py
# Topologically Sorted Source Nodes: [x_428, x_429, x_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_428 => add_310, mul_403, mul_404, sub_134
#   x_429 => add_311
#   x_430 => relu_127
# Graph fragment:
#   %sub_134 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_134, %unsqueeze_1073), kwargs = {})
#   %mul_403 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_134, %unsqueeze_1075), kwargs = {})
#   %mul_404 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_403, %unsqueeze_1077), kwargs = {})
#   %add_310 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_404, %unsqueeze_1079), kwargs = {})
#   %add_311 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_310, %relu_124), kwargs = {})
#   %relu_127 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_311,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/wm/cwmpoawru356nbub5vml7bvukbendjl24koc25bjmzmf74jry3al.py
# Topologically Sorted Source Nodes: [x_468, x_469, x_470], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_468 => add_338, mul_439, mul_440, sub_146
#   x_469 => add_339
#   x_470 => relu_139
# Graph fragment:
#   %sub_146 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_1169), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_146, %unsqueeze_1171), kwargs = {})
#   %mul_440 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_439, %unsqueeze_1173), kwargs = {})
#   %add_338 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_440, %unsqueeze_1175), kwargs = {})
#   %add_339 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_338, %relu_136), kwargs = {})
#   %relu_139 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_339,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/hd/chdz2bpfbyyefze2czfztwypsqs6n3sdqj7efm46di3vtjnueddj.py
# Topologically Sorted Source Nodes: [x_642, x_643], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_642 => add_460, mul_595, mul_596, sub_198
#   x_643 => relu_191
# Graph fragment:
#   %sub_198 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_198, %unsqueeze_1585), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_198, %unsqueeze_1587), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_595, %unsqueeze_1589), kwargs = {})
#   %add_460 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_596, %unsqueeze_1591), kwargs = {})
#   %relu_191 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_460,), kwargs = {})
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
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
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


# kernel path: /tmp/torchinductor_sahanp/4g/c4g22g2jha4ukd7muq2mibqwj2bicrgixb37wsjf6yg2mq6fjkhk.py
# Topologically Sorted Source Nodes: [x_642, x_643, x_644], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   x_642 => add_460, mul_595, mul_596, sub_198
#   x_643 => relu_191
#   x_644 => convolution_199
# Graph fragment:
#   %sub_198 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_198, %unsqueeze_1585), kwargs = {})
#   %mul_595 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_198, %unsqueeze_1587), kwargs = {})
#   %mul_596 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_595, %unsqueeze_1589), kwargs = {})
#   %add_460 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_596, %unsqueeze_1591), kwargs = {})
#   %relu_191 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_460,), kwargs = {})
#   %convolution_199 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_191, %arg476_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[524288, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 524288
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ij/cijqtpzl7sdme64bcnl5pjqlit2ykhleubtzlcqvuraxw24m37s7.py
# Topologically Sorted Source Nodes: [x_645, x_646], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_645 => add_462, mul_598, mul_599, sub_199
#   x_646 => relu_192
# Graph fragment:
#   %sub_199 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_199, %unsqueeze_1593), kwargs = {})
#   %mul_598 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_199, %unsqueeze_1595), kwargs = {})
#   %mul_599 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_598, %unsqueeze_1597), kwargs = {})
#   %add_462 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_599, %unsqueeze_1599), kwargs = {})
#   %relu_192 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_462,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 4096
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


# kernel path: /tmp/torchinductor_sahanp/6r/c6rrxblpg2224ja3eg4p2x62j6lu3tpk3ca4aqa7gj25corctxxm.py
# Topologically Sorted Source Nodes: [x_648, input_16, x_649, x_650], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_466, mul_604, mul_605, sub_201
#   x_648 => add_464, mul_601, mul_602, sub_200
#   x_649 => add_467
#   x_650 => relu_193
# Graph fragment:
#   %sub_200 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_200, %unsqueeze_1601), kwargs = {})
#   %mul_601 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_200, %unsqueeze_1603), kwargs = {})
#   %mul_602 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_601, %unsqueeze_1605), kwargs = {})
#   %add_464 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_602, %unsqueeze_1607), kwargs = {})
#   %sub_201 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_201, %unsqueeze_1609), kwargs = {})
#   %mul_604 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_201, %unsqueeze_1611), kwargs = {})
#   %mul_605 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_604, %unsqueeze_1613), kwargs = {})
#   %add_466 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_605, %unsqueeze_1615), kwargs = {})
#   %add_467 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_464, %add_466), kwargs = {})
#   %relu_193 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_467,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/c5/cc53lvile2ucuua5xyrv7hsfpmglqwehrk5lh3bypijkbea5jukg.py
# Topologically Sorted Source Nodes: [x_658, x_659, x_660], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_658 => add_473, mul_613, mul_614, sub_204
#   x_659 => add_474
#   x_660 => relu_196
# Graph fragment:
#   %sub_204 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_204, %unsqueeze_1633), kwargs = {})
#   %mul_613 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_204, %unsqueeze_1635), kwargs = {})
#   %mul_614 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_613, %unsqueeze_1637), kwargs = {})
#   %add_473 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_614, %unsqueeze_1639), kwargs = {})
#   %add_474 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_473, %relu_193), kwargs = {})
#   %relu_196 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_474,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/ac/cacnssewmpz7w6bcgkuhboksoe6ifbvrzbyfnqxadiaccp4ttf2c.py
# Topologically Sorted Source Nodes: [x_668, x_669, x_670, x_671], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_668 => add_480, mul_622, mul_623, sub_207
#   x_669 => add_481
#   x_670 => relu_199
#   x_671 => mean_1
# Graph fragment:
#   %sub_207 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_207, %unsqueeze_1657), kwargs = {})
#   %mul_622 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_207, %unsqueeze_1659), kwargs = {})
#   %mul_623 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_622, %unsqueeze_1661), kwargs = {})
#   %add_480 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_623, %unsqueeze_1663), kwargs = {})
#   %add_481 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_480, %relu_196), kwargs = {})
#   %relu_199 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_481,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_199, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_24 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (512, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (512, ), (1, ))
    assert_size_stride(arg10_1, (512, ), (1, ))
    assert_size_stride(arg11_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg12_1, (512, ), (1, ))
    assert_size_stride(arg13_1, (512, ), (1, ))
    assert_size_stride(arg14_1, (512, ), (1, ))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, ), (1, ))
    assert_size_stride(arg21_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (256, ), (1, ))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg57_1, (1024, ), (1, ))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (1024, ), (1, ))
    assert_size_stride(arg61_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg62_1, (1024, ), (1, ))
    assert_size_stride(arg63_1, (1024, ), (1, ))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (1024, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (1024, ), (1, ))
    assert_size_stride(arg85_1, (1024, ), (1, ))
    assert_size_stride(arg86_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (1024, ), (1, ))
    assert_size_stride(arg94_1, (1024, ), (1, ))
    assert_size_stride(arg95_1, (1024, ), (1, ))
    assert_size_stride(arg96_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg97_1, (1024, ), (1, ))
    assert_size_stride(arg98_1, (1024, ), (1, ))
    assert_size_stride(arg99_1, (1024, ), (1, ))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg107_1, (1024, ), (1, ))
    assert_size_stride(arg108_1, (1024, ), (1, ))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg112_1, (1024, ), (1, ))
    assert_size_stride(arg113_1, (1024, ), (1, ))
    assert_size_stride(arg114_1, (1024, ), (1, ))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg122_1, (2048, ), (1, ))
    assert_size_stride(arg123_1, (2048, ), (1, ))
    assert_size_stride(arg124_1, (2048, ), (1, ))
    assert_size_stride(arg125_1, (2048, ), (1, ))
    assert_size_stride(arg126_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg127_1, (2048, ), (1, ))
    assert_size_stride(arg128_1, (2048, ), (1, ))
    assert_size_stride(arg129_1, (2048, ), (1, ))
    assert_size_stride(arg130_1, (2048, ), (1, ))
    assert_size_stride(arg131_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (1024, ), (1, ))
    assert_size_stride(arg135_1, (1024, ), (1, ))
    assert_size_stride(arg136_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (1024, ), (1, ))
    assert_size_stride(arg141_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg142_1, (2048, ), (1, ))
    assert_size_stride(arg143_1, (2048, ), (1, ))
    assert_size_stride(arg144_1, (2048, ), (1, ))
    assert_size_stride(arg145_1, (2048, ), (1, ))
    assert_size_stride(arg146_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg147_1, (2048, ), (1, ))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (2048, ), (1, ))
    assert_size_stride(arg150_1, (2048, ), (1, ))
    assert_size_stride(arg151_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg152_1, (1024, ), (1, ))
    assert_size_stride(arg153_1, (1024, ), (1, ))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (2048, ), (1, ))
    assert_size_stride(arg160_1, (2048, ), (1, ))
    assert_size_stride(arg161_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg162_1, (2048, ), (1, ))
    assert_size_stride(arg163_1, (2048, ), (1, ))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (2048, ), (1, ))
    assert_size_stride(arg166_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1024, ), (1, ))
    assert_size_stride(arg171_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg172_1, (2048, ), (1, ))
    assert_size_stride(arg173_1, (2048, ), (1, ))
    assert_size_stride(arg174_1, (2048, ), (1, ))
    assert_size_stride(arg175_1, (2048, ), (1, ))
    assert_size_stride(arg176_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg177_1, (2048, ), (1, ))
    assert_size_stride(arg178_1, (2048, ), (1, ))
    assert_size_stride(arg179_1, (2048, ), (1, ))
    assert_size_stride(arg180_1, (2048, ), (1, ))
    assert_size_stride(arg181_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (1024, ), (1, ))
    assert_size_stride(arg186_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg187_1, (2048, ), (1, ))
    assert_size_stride(arg188_1, (2048, ), (1, ))
    assert_size_stride(arg189_1, (2048, ), (1, ))
    assert_size_stride(arg190_1, (2048, ), (1, ))
    assert_size_stride(arg191_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg192_1, (2048, ), (1, ))
    assert_size_stride(arg193_1, (2048, ), (1, ))
    assert_size_stride(arg194_1, (2048, ), (1, ))
    assert_size_stride(arg195_1, (2048, ), (1, ))
    assert_size_stride(arg196_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg197_1, (1024, ), (1, ))
    assert_size_stride(arg198_1, (1024, ), (1, ))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg202_1, (2048, ), (1, ))
    assert_size_stride(arg203_1, (2048, ), (1, ))
    assert_size_stride(arg204_1, (2048, ), (1, ))
    assert_size_stride(arg205_1, (2048, ), (1, ))
    assert_size_stride(arg206_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg207_1, (2048, ), (1, ))
    assert_size_stride(arg208_1, (2048, ), (1, ))
    assert_size_stride(arg209_1, (2048, ), (1, ))
    assert_size_stride(arg210_1, (2048, ), (1, ))
    assert_size_stride(arg211_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg212_1, (1024, ), (1, ))
    assert_size_stride(arg213_1, (1024, ), (1, ))
    assert_size_stride(arg214_1, (1024, ), (1, ))
    assert_size_stride(arg215_1, (1024, ), (1, ))
    assert_size_stride(arg216_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg217_1, (2048, ), (1, ))
    assert_size_stride(arg218_1, (2048, ), (1, ))
    assert_size_stride(arg219_1, (2048, ), (1, ))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg222_1, (2048, ), (1, ))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (2048, ), (1, ))
    assert_size_stride(arg225_1, (2048, ), (1, ))
    assert_size_stride(arg226_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (1024, ), (1, ))
    assert_size_stride(arg229_1, (1024, ), (1, ))
    assert_size_stride(arg230_1, (1024, ), (1, ))
    assert_size_stride(arg231_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg232_1, (2048, ), (1, ))
    assert_size_stride(arg233_1, (2048, ), (1, ))
    assert_size_stride(arg234_1, (2048, ), (1, ))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg237_1, (2048, ), (1, ))
    assert_size_stride(arg238_1, (2048, ), (1, ))
    assert_size_stride(arg239_1, (2048, ), (1, ))
    assert_size_stride(arg240_1, (2048, ), (1, ))
    assert_size_stride(arg241_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg242_1, (1024, ), (1, ))
    assert_size_stride(arg243_1, (1024, ), (1, ))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg247_1, (2048, ), (1, ))
    assert_size_stride(arg248_1, (2048, ), (1, ))
    assert_size_stride(arg249_1, (2048, ), (1, ))
    assert_size_stride(arg250_1, (2048, ), (1, ))
    assert_size_stride(arg251_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg252_1, (2048, ), (1, ))
    assert_size_stride(arg253_1, (2048, ), (1, ))
    assert_size_stride(arg254_1, (2048, ), (1, ))
    assert_size_stride(arg255_1, (2048, ), (1, ))
    assert_size_stride(arg256_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg257_1, (1024, ), (1, ))
    assert_size_stride(arg258_1, (1024, ), (1, ))
    assert_size_stride(arg259_1, (1024, ), (1, ))
    assert_size_stride(arg260_1, (1024, ), (1, ))
    assert_size_stride(arg261_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg262_1, (2048, ), (1, ))
    assert_size_stride(arg263_1, (2048, ), (1, ))
    assert_size_stride(arg264_1, (2048, ), (1, ))
    assert_size_stride(arg265_1, (2048, ), (1, ))
    assert_size_stride(arg266_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg267_1, (2048, ), (1, ))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (2048, ), (1, ))
    assert_size_stride(arg270_1, (2048, ), (1, ))
    assert_size_stride(arg271_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (1024, ), (1, ))
    assert_size_stride(arg274_1, (1024, ), (1, ))
    assert_size_stride(arg275_1, (1024, ), (1, ))
    assert_size_stride(arg276_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg277_1, (2048, ), (1, ))
    assert_size_stride(arg278_1, (2048, ), (1, ))
    assert_size_stride(arg279_1, (2048, ), (1, ))
    assert_size_stride(arg280_1, (2048, ), (1, ))
    assert_size_stride(arg281_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg282_1, (2048, ), (1, ))
    assert_size_stride(arg283_1, (2048, ), (1, ))
    assert_size_stride(arg284_1, (2048, ), (1, ))
    assert_size_stride(arg285_1, (2048, ), (1, ))
    assert_size_stride(arg286_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg287_1, (1024, ), (1, ))
    assert_size_stride(arg288_1, (1024, ), (1, ))
    assert_size_stride(arg289_1, (1024, ), (1, ))
    assert_size_stride(arg290_1, (1024, ), (1, ))
    assert_size_stride(arg291_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg292_1, (2048, ), (1, ))
    assert_size_stride(arg293_1, (2048, ), (1, ))
    assert_size_stride(arg294_1, (2048, ), (1, ))
    assert_size_stride(arg295_1, (2048, ), (1, ))
    assert_size_stride(arg296_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg297_1, (2048, ), (1, ))
    assert_size_stride(arg298_1, (2048, ), (1, ))
    assert_size_stride(arg299_1, (2048, ), (1, ))
    assert_size_stride(arg300_1, (2048, ), (1, ))
    assert_size_stride(arg301_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (1024, ), (1, ))
    assert_size_stride(arg304_1, (1024, ), (1, ))
    assert_size_stride(arg305_1, (1024, ), (1, ))
    assert_size_stride(arg306_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg307_1, (2048, ), (1, ))
    assert_size_stride(arg308_1, (2048, ), (1, ))
    assert_size_stride(arg309_1, (2048, ), (1, ))
    assert_size_stride(arg310_1, (2048, ), (1, ))
    assert_size_stride(arg311_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg312_1, (2048, ), (1, ))
    assert_size_stride(arg313_1, (2048, ), (1, ))
    assert_size_stride(arg314_1, (2048, ), (1, ))
    assert_size_stride(arg315_1, (2048, ), (1, ))
    assert_size_stride(arg316_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg317_1, (1024, ), (1, ))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg322_1, (2048, ), (1, ))
    assert_size_stride(arg323_1, (2048, ), (1, ))
    assert_size_stride(arg324_1, (2048, ), (1, ))
    assert_size_stride(arg325_1, (2048, ), (1, ))
    assert_size_stride(arg326_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg327_1, (2048, ), (1, ))
    assert_size_stride(arg328_1, (2048, ), (1, ))
    assert_size_stride(arg329_1, (2048, ), (1, ))
    assert_size_stride(arg330_1, (2048, ), (1, ))
    assert_size_stride(arg331_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg332_1, (1024, ), (1, ))
    assert_size_stride(arg333_1, (1024, ), (1, ))
    assert_size_stride(arg334_1, (1024, ), (1, ))
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg337_1, (2048, ), (1, ))
    assert_size_stride(arg338_1, (2048, ), (1, ))
    assert_size_stride(arg339_1, (2048, ), (1, ))
    assert_size_stride(arg340_1, (2048, ), (1, ))
    assert_size_stride(arg341_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg342_1, (2048, ), (1, ))
    assert_size_stride(arg343_1, (2048, ), (1, ))
    assert_size_stride(arg344_1, (2048, ), (1, ))
    assert_size_stride(arg345_1, (2048, ), (1, ))
    assert_size_stride(arg346_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (1024, ), (1, ))
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg352_1, (2048, ), (1, ))
    assert_size_stride(arg353_1, (2048, ), (1, ))
    assert_size_stride(arg354_1, (2048, ), (1, ))
    assert_size_stride(arg355_1, (2048, ), (1, ))
    assert_size_stride(arg356_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg357_1, (2048, ), (1, ))
    assert_size_stride(arg358_1, (2048, ), (1, ))
    assert_size_stride(arg359_1, (2048, ), (1, ))
    assert_size_stride(arg360_1, (2048, ), (1, ))
    assert_size_stride(arg361_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (1024, ), (1, ))
    assert_size_stride(arg365_1, (1024, ), (1, ))
    assert_size_stride(arg366_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg367_1, (2048, ), (1, ))
    assert_size_stride(arg368_1, (2048, ), (1, ))
    assert_size_stride(arg369_1, (2048, ), (1, ))
    assert_size_stride(arg370_1, (2048, ), (1, ))
    assert_size_stride(arg371_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg372_1, (2048, ), (1, ))
    assert_size_stride(arg373_1, (2048, ), (1, ))
    assert_size_stride(arg374_1, (2048, ), (1, ))
    assert_size_stride(arg375_1, (2048, ), (1, ))
    assert_size_stride(arg376_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (1024, ), (1, ))
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg382_1, (2048, ), (1, ))
    assert_size_stride(arg383_1, (2048, ), (1, ))
    assert_size_stride(arg384_1, (2048, ), (1, ))
    assert_size_stride(arg385_1, (2048, ), (1, ))
    assert_size_stride(arg386_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg387_1, (2048, ), (1, ))
    assert_size_stride(arg388_1, (2048, ), (1, ))
    assert_size_stride(arg389_1, (2048, ), (1, ))
    assert_size_stride(arg390_1, (2048, ), (1, ))
    assert_size_stride(arg391_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (1024, ), (1, ))
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg397_1, (2048, ), (1, ))
    assert_size_stride(arg398_1, (2048, ), (1, ))
    assert_size_stride(arg399_1, (2048, ), (1, ))
    assert_size_stride(arg400_1, (2048, ), (1, ))
    assert_size_stride(arg401_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg402_1, (2048, ), (1, ))
    assert_size_stride(arg403_1, (2048, ), (1, ))
    assert_size_stride(arg404_1, (2048, ), (1, ))
    assert_size_stride(arg405_1, (2048, ), (1, ))
    assert_size_stride(arg406_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg407_1, (1024, ), (1, ))
    assert_size_stride(arg408_1, (1024, ), (1, ))
    assert_size_stride(arg409_1, (1024, ), (1, ))
    assert_size_stride(arg410_1, (1024, ), (1, ))
    assert_size_stride(arg411_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg412_1, (2048, ), (1, ))
    assert_size_stride(arg413_1, (2048, ), (1, ))
    assert_size_stride(arg414_1, (2048, ), (1, ))
    assert_size_stride(arg415_1, (2048, ), (1, ))
    assert_size_stride(arg416_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg417_1, (2048, ), (1, ))
    assert_size_stride(arg418_1, (2048, ), (1, ))
    assert_size_stride(arg419_1, (2048, ), (1, ))
    assert_size_stride(arg420_1, (2048, ), (1, ))
    assert_size_stride(arg421_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (1024, ), (1, ))
    assert_size_stride(arg425_1, (1024, ), (1, ))
    assert_size_stride(arg426_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg427_1, (2048, ), (1, ))
    assert_size_stride(arg428_1, (2048, ), (1, ))
    assert_size_stride(arg429_1, (2048, ), (1, ))
    assert_size_stride(arg430_1, (2048, ), (1, ))
    assert_size_stride(arg431_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg432_1, (2048, ), (1, ))
    assert_size_stride(arg433_1, (2048, ), (1, ))
    assert_size_stride(arg434_1, (2048, ), (1, ))
    assert_size_stride(arg435_1, (2048, ), (1, ))
    assert_size_stride(arg436_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg437_1, (1024, ), (1, ))
    assert_size_stride(arg438_1, (1024, ), (1, ))
    assert_size_stride(arg439_1, (1024, ), (1, ))
    assert_size_stride(arg440_1, (1024, ), (1, ))
    assert_size_stride(arg441_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg442_1, (2048, ), (1, ))
    assert_size_stride(arg443_1, (2048, ), (1, ))
    assert_size_stride(arg444_1, (2048, ), (1, ))
    assert_size_stride(arg445_1, (2048, ), (1, ))
    assert_size_stride(arg446_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg447_1, (2048, ), (1, ))
    assert_size_stride(arg448_1, (2048, ), (1, ))
    assert_size_stride(arg449_1, (2048, ), (1, ))
    assert_size_stride(arg450_1, (2048, ), (1, ))
    assert_size_stride(arg451_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg452_1, (1024, ), (1, ))
    assert_size_stride(arg453_1, (1024, ), (1, ))
    assert_size_stride(arg454_1, (1024, ), (1, ))
    assert_size_stride(arg455_1, (1024, ), (1, ))
    assert_size_stride(arg456_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg457_1, (2048, ), (1, ))
    assert_size_stride(arg458_1, (2048, ), (1, ))
    assert_size_stride(arg459_1, (2048, ), (1, ))
    assert_size_stride(arg460_1, (2048, ), (1, ))
    assert_size_stride(arg461_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg462_1, (2048, ), (1, ))
    assert_size_stride(arg463_1, (2048, ), (1, ))
    assert_size_stride(arg464_1, (2048, ), (1, ))
    assert_size_stride(arg465_1, (2048, ), (1, ))
    assert_size_stride(arg466_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (1024, ), (1, ))
    assert_size_stride(arg470_1, (1024, ), (1, ))
    assert_size_stride(arg471_1, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg472_1, (4096, ), (1, ))
    assert_size_stride(arg473_1, (4096, ), (1, ))
    assert_size_stride(arg474_1, (4096, ), (1, ))
    assert_size_stride(arg475_1, (4096, ), (1, ))
    assert_size_stride(arg476_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg477_1, (4096, ), (1, ))
    assert_size_stride(arg478_1, (4096, ), (1, ))
    assert_size_stride(arg479_1, (4096, ), (1, ))
    assert_size_stride(arg480_1, (4096, ), (1, ))
    assert_size_stride(arg481_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg482_1, (2048, ), (1, ))
    assert_size_stride(arg483_1, (2048, ), (1, ))
    assert_size_stride(arg484_1, (2048, ), (1, ))
    assert_size_stride(arg485_1, (2048, ), (1, ))
    assert_size_stride(arg486_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg487_1, (2048, ), (1, ))
    assert_size_stride(arg488_1, (2048, ), (1, ))
    assert_size_stride(arg489_1, (2048, ), (1, ))
    assert_size_stride(arg490_1, (2048, ), (1, ))
    assert_size_stride(arg491_1, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg492_1, (4096, ), (1, ))
    assert_size_stride(arg493_1, (4096, ), (1, ))
    assert_size_stride(arg494_1, (4096, ), (1, ))
    assert_size_stride(arg495_1, (4096, ), (1, ))
    assert_size_stride(arg496_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg497_1, (4096, ), (1, ))
    assert_size_stride(arg498_1, (4096, ), (1, ))
    assert_size_stride(arg499_1, (4096, ), (1, ))
    assert_size_stride(arg500_1, (4096, ), (1, ))
    assert_size_stride(arg501_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg502_1, (2048, ), (1, ))
    assert_size_stride(arg503_1, (2048, ), (1, ))
    assert_size_stride(arg504_1, (2048, ), (1, ))
    assert_size_stride(arg505_1, (2048, ), (1, ))
    assert_size_stride(arg506_1, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg507_1, (4096, ), (1, ))
    assert_size_stride(arg508_1, (4096, ), (1, ))
    assert_size_stride(arg509_1, (4096, ), (1, ))
    assert_size_stride(arg510_1, (4096, ), (1, ))
    assert_size_stride(arg511_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg512_1, (4096, ), (1, ))
    assert_size_stride(arg513_1, (4096, ), (1, ))
    assert_size_stride(arg514_1, (4096, ), (1, ))
    assert_size_stride(arg515_1, (4096, ), (1, ))
    assert_size_stride(arg516_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg517_1, (2048, ), (1, ))
    assert_size_stride(arg518_1, (2048, ), (1, ))
    assert_size_stride(arg519_1, (2048, ), (1, ))
    assert_size_stride(arg520_1, (2048, ), (1, ))
    assert_size_stride(arg521_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg522_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_337], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_337], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 49, grid=grid(192, 49), stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_337], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 112, 112), (802816, 1, 7168, 64))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_338, x_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        buf4 = empty_strided_cuda((8, 64, 56, 56), (200704, 1, 3584, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_338, x_339, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 1605632, grid=grid(1605632), stream=stream0)
        # Topologically Sorted Source Nodes: [x_341], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 512, 56, 56), (1605632, 1, 28672, 512))
        del arg6_1
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_342, x_343], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf6, arg7_1, arg8_1, arg9_1, arg10_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg10_1
        del arg7_1
        del arg8_1
        del arg9_1
        buf7 = empty_strided_cuda((512, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_342, x_343, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg11_1, buf7, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg11_1
        # Topologically Sorted Source Nodes: [x_342, x_343, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf8, (8, 512, 56, 56), (1605632, 1, 28672, 512))
        del buf6
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_345, x_346], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf9, arg12_1, arg13_1, arg14_1, arg15_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        # Topologically Sorted Source Nodes: [x_345, x_346, x_347], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg16_1
        del buf9
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf4, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg21_1
        buf12 = buf10; del buf10  # reuse
        buf13 = reinterpret_tensor(buf3, (8, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_348, input_10, x_349, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf12, arg17_1, arg18_1, arg19_1, arg20_1, buf11, arg22_1, arg23_1, arg24_1, arg25_1, buf13, 6422528, grid=grid(6422528), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg25_1
        del buf11
        del buf12
        # Topologically Sorted Source Nodes: [x_350, x_351], Original ATen: [aten.relu, aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 512, 56, 56), (1605632, 1, 28672, 512))
        del arg26_1
        buf15 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf15, arg27_1, arg28_1, arg29_1, arg30_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg27_1
        del arg28_1
        del arg29_1
        del arg30_1
        buf16 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_352, x_353, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg31_1, buf16, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg31_1
        # Topologically Sorted Source Nodes: [x_352, x_353, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf17 = extern_kernels.convolution(buf15, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf17, (8, 512, 56, 56), (1605632, 1, 28672, 512))
        del buf15
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [x_355, x_356], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf18, arg32_1, arg33_1, arg34_1, arg35_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg32_1
        del arg33_1
        del arg34_1
        del arg35_1
        # Topologically Sorted Source Nodes: [x_355, x_356, x_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf19 = extern_kernels.convolution(buf18, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg36_1
        del buf18
        buf20 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_358, x_359, x_360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf20, buf19, arg37_1, arg38_1, arg39_1, arg40_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg37_1
        del arg38_1
        del arg39_1
        del arg40_1
        del buf19
        # Topologically Sorted Source Nodes: [x_361], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 512, 56, 56), (1605632, 1, 28672, 512))
        del arg41_1
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_362, x_363], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf22, arg42_1, arg43_1, arg44_1, arg45_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg42_1
        del arg43_1
        del arg44_1
        del arg45_1
        buf23 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_362, x_363, x_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg46_1, buf23, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg46_1
        # Topologically Sorted Source Nodes: [x_362, x_363, x_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf24, (8, 512, 56, 56), (1605632, 1, 28672, 512))
        del buf22
        del buf23
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_365, x_366], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf25, arg47_1, arg48_1, arg49_1, arg50_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg47_1
        del arg48_1
        del arg49_1
        del arg50_1
        # Topologically Sorted Source Nodes: [x_365, x_366, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 256, 56, 56), (802816, 1, 14336, 256))
        del arg51_1
        del buf25
        buf27 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_368, x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf27, buf26, arg52_1, arg53_1, arg54_1, arg55_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg52_1
        del arg53_1
        del arg54_1
        del arg55_1
        del buf26
        # Topologically Sorted Source Nodes: [x_371], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 1024, 56, 56), (3211264, 1, 57344, 1024))
        del arg56_1
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_372, x_373], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf29, arg57_1, arg58_1, arg59_1, arg60_1, 25690112, grid=grid(25690112), stream=stream0)
        del arg57_1
        del arg58_1
        del arg59_1
        del arg60_1
        buf30 = empty_strided_cuda((1024, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_372, x_373, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg61_1, buf30, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg61_1
        # Topologically Sorted Source Nodes: [x_372, x_373, x_374], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf31 = extern_kernels.convolution(buf29, buf30, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf31, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
        del buf29
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf32, arg62_1, arg63_1, arg64_1, arg65_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg62_1
        del arg63_1
        del arg64_1
        del arg65_1
        # Topologically Sorted Source Nodes: [x_375, x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg66_1
        del buf32
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf27, arg71_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg71_1
        del buf27
        buf35 = buf33; del buf33  # reuse
        buf36 = empty_strided_cuda((8, 512, 28, 28), (401408, 1, 14336, 512), torch.float32)
        # Topologically Sorted Source Nodes: [x_378, input_12, x_379, x_380], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf35, arg67_1, arg68_1, arg69_1, arg70_1, buf34, arg72_1, arg73_1, arg74_1, arg75_1, buf36, 3211264, grid=grid(3211264), stream=stream0)
        del arg67_1
        del arg68_1
        del arg69_1
        del arg70_1
        del arg72_1
        del arg73_1
        del arg74_1
        del arg75_1
        del buf34
        del buf35
        # Topologically Sorted Source Nodes: [x_380, x_381], Original ATen: [aten.relu, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
        del arg76_1
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_382, x_383], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf38, arg77_1, arg78_1, arg79_1, arg80_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg77_1
        del arg78_1
        del arg79_1
        del arg80_1
        buf39 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_382, x_383, x_384], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg81_1, buf39, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg81_1
        # Topologically Sorted Source Nodes: [x_382, x_383, x_384], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf40, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
        del buf38
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_385, x_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf41, arg82_1, arg83_1, arg84_1, arg85_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg82_1
        del arg83_1
        del arg84_1
        del arg85_1
        # Topologically Sorted Source Nodes: [x_385, x_386, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg86_1
        del buf41
        buf43 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [x_388, x_389, x_390], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf43, buf42, arg87_1, arg88_1, arg89_1, arg90_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg87_1
        del arg88_1
        del arg89_1
        del arg90_1
        del buf42
        # Topologically Sorted Source Nodes: [x_391], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
        del arg91_1
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [x_392, x_393], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf45, arg92_1, arg93_1, arg94_1, arg95_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg92_1
        del arg93_1
        del arg94_1
        del arg95_1
        buf46 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [x_392, x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg96_1, buf46, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg96_1
        # Topologically Sorted Source Nodes: [x_392, x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf47 = extern_kernels.convolution(buf45, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf47, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
        del buf45
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_395, x_396], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf48, arg97_1, arg98_1, arg99_1, arg100_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg100_1
        del arg97_1
        del arg98_1
        del arg99_1
        # Topologically Sorted Source Nodes: [x_395, x_396, x_397], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg101_1
        del buf48
        buf50 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_398, x_399, x_400], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf50, buf49, arg102_1, arg103_1, arg104_1, arg105_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del buf49
        # Topologically Sorted Source Nodes: [x_401], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
        del arg106_1
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_402, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf52, arg107_1, arg108_1, arg109_1, arg110_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        buf53 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_402, x_403, x_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg111_1, buf53, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg111_1
        # Topologically Sorted Source Nodes: [x_402, x_403, x_404], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf54, (8, 1024, 28, 28), (802816, 1, 28672, 1024))
        del buf52
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [x_405, x_406], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf55, arg112_1, arg113_1, arg114_1, arg115_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg112_1
        del arg113_1
        del arg114_1
        del arg115_1
        # Topologically Sorted Source Nodes: [x_405, x_406, x_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 512, 28, 28), (401408, 1, 14336, 512))
        del arg116_1
        del buf55
        buf57 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_408, x_409, x_410], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf57, buf56, arg117_1, arg118_1, arg119_1, arg120_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg117_1
        del arg118_1
        del arg119_1
        del arg120_1
        del buf56
        # Topologically Sorted Source Nodes: [x_411], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 2048, 28, 28), (1605632, 1, 57344, 2048))
        del arg121_1
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_412, x_413], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf59, arg122_1, arg123_1, arg124_1, arg125_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg122_1
        del arg123_1
        del arg124_1
        del arg125_1
        buf60 = empty_strided_cuda((2048, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_412, x_413, x_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg126_1, buf60, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg126_1
        # Topologically Sorted Source Nodes: [x_412, x_413, x_414], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf61 = extern_kernels.convolution(buf59, buf60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf61, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf59
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_415, x_416], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf62, arg127_1, arg128_1, arg129_1, arg130_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg127_1
        del arg128_1
        del arg129_1
        del arg130_1
        # Topologically Sorted Source Nodes: [x_415, x_416, x_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg131_1
        del buf62
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf57, arg136_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg136_1
        del buf57
        buf65 = buf63; del buf63  # reuse
        buf66 = reinterpret_tensor(buf4, (8, 1024, 14, 14), (200704, 1, 14336, 1024), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_418, input_14, x_419, x_420], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf65, arg132_1, arg133_1, arg134_1, arg135_1, buf64, arg137_1, arg138_1, arg139_1, arg140_1, buf66, 1605632, grid=grid(1605632), stream=stream0)
        del arg132_1
        del arg133_1
        del arg134_1
        del arg135_1
        del arg137_1
        del arg138_1
        del arg139_1
        del arg140_1
        del buf64
        del buf65
        # Topologically Sorted Source Nodes: [x_420, x_421], Original ATen: [aten.relu, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg141_1
        buf68 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [x_422, x_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf68, arg142_1, arg143_1, arg144_1, arg145_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg142_1
        del arg143_1
        del arg144_1
        del arg145_1
        buf69 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_422, x_423, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg146_1, buf69, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg146_1
        # Topologically Sorted Source Nodes: [x_422, x_423, x_424], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf70 = extern_kernels.convolution(buf68, buf69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf70, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf68
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_425, x_426], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf71, arg147_1, arg148_1, arg149_1, arg150_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg147_1
        del arg148_1
        del arg149_1
        del arg150_1
        # Topologically Sorted Source Nodes: [x_425, x_426, x_427], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg151_1
        del buf71
        buf73 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_428, x_429, x_430], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf73, buf72, arg152_1, arg153_1, arg154_1, arg155_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg152_1
        del arg153_1
        del arg154_1
        del arg155_1
        del buf72
        # Topologically Sorted Source Nodes: [x_431], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg156_1
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [x_432, x_433], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf75, arg157_1, arg158_1, arg159_1, arg160_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg157_1
        del arg158_1
        del arg159_1
        del arg160_1
        buf76 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_432, x_433, x_434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg161_1, buf76, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg161_1
        # Topologically Sorted Source Nodes: [x_432, x_433, x_434], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf77 = extern_kernels.convolution(buf75, buf76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf77, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf75
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_435, x_436], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf78, arg162_1, arg163_1, arg164_1, arg165_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg162_1
        del arg163_1
        del arg164_1
        del arg165_1
        # Topologically Sorted Source Nodes: [x_435, x_436, x_437], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf79 = extern_kernels.convolution(buf78, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg166_1
        del buf78
        buf80 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_438, x_439, x_440], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf80, buf79, arg167_1, arg168_1, arg169_1, arg170_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        del buf79
        # Topologically Sorted Source Nodes: [x_441], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg171_1
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_442, x_443], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf82, arg172_1, arg173_1, arg174_1, arg175_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        buf83 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_442, x_443, x_444], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg176_1, buf83, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [x_442, x_443, x_444], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf84, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf82
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [x_445, x_446], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf85, arg177_1, arg178_1, arg179_1, arg180_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        # Topologically Sorted Source Nodes: [x_445, x_446, x_447], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg181_1
        del buf85
        buf87 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_448, x_449, x_450], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf87, buf86, arg182_1, arg183_1, arg184_1, arg185_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del buf86
        # Topologically Sorted Source Nodes: [x_451], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg186_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg186_1
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_452, x_453], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf89, arg187_1, arg188_1, arg189_1, arg190_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg187_1
        del arg188_1
        del arg189_1
        del arg190_1
        buf90 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_452, x_453, x_454], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg191_1, buf90, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg191_1
        # Topologically Sorted Source Nodes: [x_452, x_453, x_454], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf91 = extern_kernels.convolution(buf89, buf90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf91, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf89
        buf92 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_455, x_456], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf92, arg192_1, arg193_1, arg194_1, arg195_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg192_1
        del arg193_1
        del arg194_1
        del arg195_1
        # Topologically Sorted Source Nodes: [x_455, x_456, x_457], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg196_1
        del buf92
        buf94 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_458, x_459, x_460], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf94, buf93, arg197_1, arg198_1, arg199_1, arg200_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg197_1
        del arg198_1
        del arg199_1
        del arg200_1
        del buf93
        # Topologically Sorted Source Nodes: [x_461], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg201_1
        buf96 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [x_462, x_463], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf96, arg202_1, arg203_1, arg204_1, arg205_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg202_1
        del arg203_1
        del arg204_1
        del arg205_1
        buf97 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_462, x_463, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg206_1, buf97, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg206_1
        # Topologically Sorted Source Nodes: [x_462, x_463, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf98, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf96
        buf99 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_465, x_466], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf99, arg207_1, arg208_1, arg209_1, arg210_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg207_1
        del arg208_1
        del arg209_1
        del arg210_1
        # Topologically Sorted Source Nodes: [x_465, x_466, x_467], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg211_1
        del buf99
        buf101 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_468, x_469, x_470], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf101, arg212_1, arg213_1, arg214_1, arg215_1, buf94, 1605632, grid=grid(1605632), stream=stream0)
        del arg212_1
        del arg213_1
        del arg214_1
        del arg215_1
        del buf94
        # Topologically Sorted Source Nodes: [x_471], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg216_1
        buf103 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_472, x_473], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf103, arg217_1, arg218_1, arg219_1, arg220_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg217_1
        del arg218_1
        del arg219_1
        del arg220_1
        buf104 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [x_472, x_473, x_474], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg221_1, buf104, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg221_1
        # Topologically Sorted Source Nodes: [x_472, x_473, x_474], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf105 = extern_kernels.convolution(buf103, buf104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf105, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf103
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [x_475, x_476], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf106, arg222_1, arg223_1, arg224_1, arg225_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg222_1
        del arg223_1
        del arg224_1
        del arg225_1
        # Topologically Sorted Source Nodes: [x_475, x_476, x_477], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg226_1
        del buf106
        buf108 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_478, x_479, x_480], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf108, buf107, arg227_1, arg228_1, arg229_1, arg230_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg227_1
        del arg228_1
        del arg229_1
        del arg230_1
        del buf107
        # Topologically Sorted Source Nodes: [x_481], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg231_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg231_1
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [x_482, x_483], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf110, arg232_1, arg233_1, arg234_1, arg235_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg232_1
        del arg233_1
        del arg234_1
        del arg235_1
        buf111 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [x_482, x_483, x_484], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg236_1, buf111, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg236_1
        # Topologically Sorted Source Nodes: [x_482, x_483, x_484], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf112 = extern_kernels.convolution(buf110, buf111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf112, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf110
        buf113 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_485, x_486], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf113, arg237_1, arg238_1, arg239_1, arg240_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        # Topologically Sorted Source Nodes: [x_485, x_486, x_487], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg241_1
        del buf113
        buf115 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_488, x_489, x_490], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf115, buf114, arg242_1, arg243_1, arg244_1, arg245_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del buf114
        # Topologically Sorted Source Nodes: [x_491], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg246_1
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [x_492, x_493], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf117, arg247_1, arg248_1, arg249_1, arg250_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        buf118 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [x_492, x_493, x_494], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg251_1, buf118, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg251_1
        # Topologically Sorted Source Nodes: [x_492, x_493, x_494], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf119, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf117
        buf120 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_495, x_496], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf120, arg252_1, arg253_1, arg254_1, arg255_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg252_1
        del arg253_1
        del arg254_1
        del arg255_1
        # Topologically Sorted Source Nodes: [x_495, x_496, x_497], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg256_1
        del buf120
        buf122 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [x_498, x_499, x_500], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf122, buf121, arg257_1, arg258_1, arg259_1, arg260_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg257_1
        del arg258_1
        del arg259_1
        del arg260_1
        del buf121
        # Topologically Sorted Source Nodes: [x_501], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg261_1
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_502, x_503], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf124, arg262_1, arg263_1, arg264_1, arg265_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg262_1
        del arg263_1
        del arg264_1
        del arg265_1
        buf125 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_502, x_503, x_504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg266_1, buf125, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg266_1
        # Topologically Sorted Source Nodes: [x_502, x_503, x_504], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf126 = extern_kernels.convolution(buf124, buf125, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf126, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf124
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_505, x_506], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf127, arg267_1, arg268_1, arg269_1, arg270_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg267_1
        del arg268_1
        del arg269_1
        del arg270_1
        # Topologically Sorted Source Nodes: [x_505, x_506, x_507], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg271_1
        del buf127
        buf129 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [x_508, x_509, x_510], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf129, buf128, arg272_1, arg273_1, arg274_1, arg275_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg272_1
        del arg273_1
        del arg274_1
        del arg275_1
        del buf128
        # Topologically Sorted Source Nodes: [x_511], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg276_1
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [x_512, x_513], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf131, arg277_1, arg278_1, arg279_1, arg280_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg277_1
        del arg278_1
        del arg279_1
        del arg280_1
        buf132 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [x_512, x_513, x_514], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg281_1, buf132, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg281_1
        # Topologically Sorted Source Nodes: [x_512, x_513, x_514], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf133 = extern_kernels.convolution(buf131, buf132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf133, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf131
        buf134 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_515, x_516], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf134, arg282_1, arg283_1, arg284_1, arg285_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg282_1
        del arg283_1
        del arg284_1
        del arg285_1
        # Topologically Sorted Source Nodes: [x_515, x_516, x_517], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf135 = extern_kernels.convolution(buf134, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg286_1
        del buf134
        buf136 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [x_518, x_519, x_520], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf136, buf135, arg287_1, arg288_1, arg289_1, arg290_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg287_1
        del arg288_1
        del arg289_1
        del arg290_1
        del buf135
        # Topologically Sorted Source Nodes: [x_521], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg291_1
        buf138 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [x_522, x_523], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf138, arg292_1, arg293_1, arg294_1, arg295_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg292_1
        del arg293_1
        del arg294_1
        del arg295_1
        buf139 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_522, x_523, x_524], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg296_1, buf139, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg296_1
        # Topologically Sorted Source Nodes: [x_522, x_523, x_524], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf140, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf138
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_525, x_526], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf141, arg297_1, arg298_1, arg299_1, arg300_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg297_1
        del arg298_1
        del arg299_1
        del arg300_1
        # Topologically Sorted Source Nodes: [x_525, x_526, x_527], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf142 = extern_kernels.convolution(buf141, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg301_1
        del buf141
        buf143 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [x_528, x_529, x_530], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf143, buf142, arg302_1, arg303_1, arg304_1, arg305_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg302_1
        del arg303_1
        del arg304_1
        del arg305_1
        del buf142
        # Topologically Sorted Source Nodes: [x_531], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg306_1
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_532, x_533], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf145, arg307_1, arg308_1, arg309_1, arg310_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg307_1
        del arg308_1
        del arg309_1
        del arg310_1
        buf146 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_532, x_533, x_534], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg311_1, buf146, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg311_1
        # Topologically Sorted Source Nodes: [x_532, x_533, x_534], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf147 = extern_kernels.convolution(buf145, buf146, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf147, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf145
        buf148 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_535, x_536], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf148, arg312_1, arg313_1, arg314_1, arg315_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg312_1
        del arg313_1
        del arg314_1
        del arg315_1
        # Topologically Sorted Source Nodes: [x_535, x_536, x_537], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf149 = extern_kernels.convolution(buf148, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg316_1
        del buf148
        buf150 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [x_538, x_539, x_540], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf150, buf149, arg317_1, arg318_1, arg319_1, arg320_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg317_1
        del arg318_1
        del arg319_1
        del arg320_1
        del buf149
        # Topologically Sorted Source Nodes: [x_541], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg321_1
        buf152 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_542, x_543], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf152, arg322_1, arg323_1, arg324_1, arg325_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg322_1
        del arg323_1
        del arg324_1
        del arg325_1
        buf153 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_542, x_543, x_544], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg326_1, buf153, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg326_1
        # Topologically Sorted Source Nodes: [x_542, x_543, x_544], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf154 = extern_kernels.convolution(buf152, buf153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf154, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf152
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_545, x_546], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf155, arg327_1, arg328_1, arg329_1, arg330_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg327_1
        del arg328_1
        del arg329_1
        del arg330_1
        # Topologically Sorted Source Nodes: [x_545, x_546, x_547], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg331_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg331_1
        del buf155
        buf157 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [x_548, x_549, x_550], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf157, buf156, arg332_1, arg333_1, arg334_1, arg335_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        del buf156
        # Topologically Sorted Source Nodes: [x_551], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg336_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg336_1
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_552, x_553], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf159, arg337_1, arg338_1, arg339_1, arg340_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg337_1
        del arg338_1
        del arg339_1
        del arg340_1
        buf160 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_552, x_553, x_554], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg341_1, buf160, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg341_1
        # Topologically Sorted Source Nodes: [x_552, x_553, x_554], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf161, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf159
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_555, x_556], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf162, arg342_1, arg343_1, arg344_1, arg345_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg342_1
        del arg343_1
        del arg344_1
        del arg345_1
        # Topologically Sorted Source Nodes: [x_555, x_556, x_557], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg346_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg346_1
        del buf162
        buf164 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_558, x_559, x_560], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf164, buf163, arg347_1, arg348_1, arg349_1, arg350_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg347_1
        del arg348_1
        del arg349_1
        del arg350_1
        del buf163
        # Topologically Sorted Source Nodes: [x_561], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg351_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg351_1
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_562, x_563], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf166, arg352_1, arg353_1, arg354_1, arg355_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg352_1
        del arg353_1
        del arg354_1
        del arg355_1
        buf167 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_562, x_563, x_564], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg356_1, buf167, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg356_1
        # Topologically Sorted Source Nodes: [x_562, x_563, x_564], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf168 = extern_kernels.convolution(buf166, buf167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf168, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf166
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_565, x_566], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf169, arg357_1, arg358_1, arg359_1, arg360_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg357_1
        del arg358_1
        del arg359_1
        del arg360_1
        # Topologically Sorted Source Nodes: [x_565, x_566, x_567], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf170 = extern_kernels.convolution(buf169, arg361_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg361_1
        del buf169
        buf171 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_568, x_569, x_570], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf171, buf170, arg362_1, arg363_1, arg364_1, arg365_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del buf170
        # Topologically Sorted Source Nodes: [x_571], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg366_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg366_1
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_572, x_573], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf173, arg367_1, arg368_1, arg369_1, arg370_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg367_1
        del arg368_1
        del arg369_1
        del arg370_1
        buf174 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [x_572, x_573, x_574], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg371_1, buf174, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg371_1
        # Topologically Sorted Source Nodes: [x_572, x_573, x_574], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf175 = extern_kernels.convolution(buf173, buf174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf175, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf173
        buf176 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_575, x_576], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf176, arg372_1, arg373_1, arg374_1, arg375_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg372_1
        del arg373_1
        del arg374_1
        del arg375_1
        # Topologically Sorted Source Nodes: [x_575, x_576, x_577], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf177 = extern_kernels.convolution(buf176, arg376_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg376_1
        del buf176
        buf178 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_578, x_579, x_580], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf178, buf177, arg377_1, arg378_1, arg379_1, arg380_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg377_1
        del arg378_1
        del arg379_1
        del arg380_1
        del buf177
        # Topologically Sorted Source Nodes: [x_581], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, arg381_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg381_1
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [x_582, x_583], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf180, arg382_1, arg383_1, arg384_1, arg385_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg382_1
        del arg383_1
        del arg384_1
        del arg385_1
        buf181 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_582, x_583, x_584], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg386_1, buf181, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg386_1
        # Topologically Sorted Source Nodes: [x_582, x_583, x_584], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf182, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf180
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_585, x_586], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf183, arg387_1, arg388_1, arg389_1, arg390_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg387_1
        del arg388_1
        del arg389_1
        del arg390_1
        # Topologically Sorted Source Nodes: [x_585, x_586, x_587], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf184 = extern_kernels.convolution(buf183, arg391_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg391_1
        del buf183
        buf185 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_588, x_589, x_590], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf185, buf184, arg392_1, arg393_1, arg394_1, arg395_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg392_1
        del arg393_1
        del arg394_1
        del arg395_1
        del buf184
        # Topologically Sorted Source Nodes: [x_591], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg396_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg396_1
        buf187 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [x_592, x_593], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf187, arg397_1, arg398_1, arg399_1, arg400_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg397_1
        del arg398_1
        del arg399_1
        del arg400_1
        buf188 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_592, x_593, x_594], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg401_1, buf188, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg401_1
        # Topologically Sorted Source Nodes: [x_592, x_593, x_594], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf189 = extern_kernels.convolution(buf187, buf188, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf189, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf187
        buf190 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [x_595, x_596], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf190, arg402_1, arg403_1, arg404_1, arg405_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg402_1
        del arg403_1
        del arg404_1
        del arg405_1
        # Topologically Sorted Source Nodes: [x_595, x_596, x_597], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf191 = extern_kernels.convolution(buf190, arg406_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg406_1
        del buf190
        buf192 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [x_598, x_599, x_600], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf192, buf191, arg407_1, arg408_1, arg409_1, arg410_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg407_1
        del arg408_1
        del arg409_1
        del arg410_1
        del buf191
        # Topologically Sorted Source Nodes: [x_601], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg411_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg411_1
        buf194 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_602, x_603], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf194, arg412_1, arg413_1, arg414_1, arg415_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        buf195 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [x_602, x_603, x_604], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg416_1, buf195, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg416_1
        # Topologically Sorted Source Nodes: [x_602, x_603, x_604], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf196 = extern_kernels.convolution(buf194, buf195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf196, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf194
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [x_605, x_606], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf197, arg417_1, arg418_1, arg419_1, arg420_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg417_1
        del arg418_1
        del arg419_1
        del arg420_1
        # Topologically Sorted Source Nodes: [x_605, x_606, x_607], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf198 = extern_kernels.convolution(buf197, arg421_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg421_1
        del buf197
        buf199 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [x_608, x_609, x_610], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf199, buf198, arg422_1, arg423_1, arg424_1, arg425_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg422_1
        del arg423_1
        del arg424_1
        del arg425_1
        del buf198
        # Topologically Sorted Source Nodes: [x_611], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, arg426_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg426_1
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_612, x_613], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf201, arg427_1, arg428_1, arg429_1, arg430_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg427_1
        del arg428_1
        del arg429_1
        del arg430_1
        buf202 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_612, x_613, x_614], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg431_1, buf202, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg431_1
        # Topologically Sorted Source Nodes: [x_612, x_613, x_614], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf203 = extern_kernels.convolution(buf201, buf202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf203, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf201
        buf204 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_615, x_616], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf204, arg432_1, arg433_1, arg434_1, arg435_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg432_1
        del arg433_1
        del arg434_1
        del arg435_1
        # Topologically Sorted Source Nodes: [x_615, x_616, x_617], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf205 = extern_kernels.convolution(buf204, arg436_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg436_1
        del buf204
        buf206 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_618, x_619, x_620], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf206, buf205, arg437_1, arg438_1, arg439_1, arg440_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg437_1
        del arg438_1
        del arg439_1
        del arg440_1
        del buf205
        # Topologically Sorted Source Nodes: [x_621], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg441_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg441_1
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_622, x_623], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf208, arg442_1, arg443_1, arg444_1, arg445_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg442_1
        del arg443_1
        del arg444_1
        del arg445_1
        buf209 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_622, x_623, x_624], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg446_1, buf209, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg446_1
        # Topologically Sorted Source Nodes: [x_622, x_623, x_624], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf210, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf208
        buf211 = buf210; del buf210  # reuse
        # Topologically Sorted Source Nodes: [x_625, x_626], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf211, arg447_1, arg448_1, arg449_1, arg450_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg447_1
        del arg448_1
        del arg449_1
        del arg450_1
        # Topologically Sorted Source Nodes: [x_625, x_626, x_627], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf212 = extern_kernels.convolution(buf211, arg451_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf212, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg451_1
        del buf211
        buf213 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_628, x_629, x_630], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf213, buf212, arg452_1, arg453_1, arg454_1, arg455_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg452_1
        del arg453_1
        del arg454_1
        del arg455_1
        del buf212
        # Topologically Sorted Source Nodes: [x_631], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, arg456_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del arg456_1
        buf215 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [x_632, x_633], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf215, arg457_1, arg458_1, arg459_1, arg460_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg457_1
        del arg458_1
        del arg459_1
        del arg460_1
        buf216 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_632, x_633, x_634], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg461_1, buf216, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg461_1
        # Topologically Sorted Source Nodes: [x_632, x_633, x_634], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf217 = extern_kernels.convolution(buf215, buf216, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf217, (8, 2048, 14, 14), (401408, 1, 28672, 2048))
        del buf215
        del buf216
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_635, x_636], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf218, arg462_1, arg463_1, arg464_1, arg465_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg462_1
        del arg463_1
        del arg464_1
        del arg465_1
        # Topologically Sorted Source Nodes: [x_635, x_636, x_637], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg466_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 1024, 14, 14), (200704, 1, 14336, 1024))
        del arg466_1
        del buf218
        buf220 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [x_638, x_639, x_640], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf220, buf219, arg467_1, arg468_1, arg469_1, arg470_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg467_1
        del arg468_1
        del arg469_1
        del arg470_1
        del buf219
        # Topologically Sorted Source Nodes: [x_641], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg471_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 4096, 14, 14), (802816, 1, 57344, 4096))
        del arg471_1
        buf222 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_642, x_643], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf222, arg472_1, arg473_1, arg474_1, arg475_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg472_1
        del arg473_1
        del arg474_1
        del arg475_1
        buf223 = empty_strided_cuda((4096, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_642, x_643, x_644], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(arg476_1, buf223, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del arg476_1
        # Topologically Sorted Source Nodes: [x_642, x_643, x_644], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf224 = extern_kernels.convolution(buf222, buf223, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf224, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
        del buf222
        buf225 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [x_645, x_646], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf225, arg477_1, arg478_1, arg479_1, arg480_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg477_1
        del arg478_1
        del arg479_1
        del arg480_1
        # Topologically Sorted Source Nodes: [x_645, x_646, x_647], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf226 = extern_kernels.convolution(buf225, arg481_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg481_1
        del buf225
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf220, arg486_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg486_1
        del buf220
        buf228 = buf226; del buf226  # reuse
        buf229 = empty_strided_cuda((8, 2048, 7, 7), (100352, 1, 14336, 2048), torch.float32)
        # Topologically Sorted Source Nodes: [x_648, input_16, x_649, x_650], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf228, arg482_1, arg483_1, arg484_1, arg485_1, buf227, arg487_1, arg488_1, arg489_1, arg490_1, buf229, 802816, grid=grid(802816), stream=stream0)
        del arg482_1
        del arg483_1
        del arg484_1
        del arg485_1
        del arg487_1
        del arg488_1
        del arg489_1
        del arg490_1
        del buf227
        del buf228
        # Topologically Sorted Source Nodes: [x_650, x_651], Original ATen: [aten.relu, aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg491_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
        del arg491_1
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [x_652, x_653], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf231, arg492_1, arg493_1, arg494_1, arg495_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg492_1
        del arg493_1
        del arg494_1
        del arg495_1
        buf232 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [x_652, x_653, x_654], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(arg496_1, buf232, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del arg496_1
        # Topologically Sorted Source Nodes: [x_652, x_653, x_654], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf233 = extern_kernels.convolution(buf231, buf232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf233, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
        del buf231
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [x_655, x_656], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf234, arg497_1, arg498_1, arg499_1, arg500_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg497_1
        del arg498_1
        del arg499_1
        del arg500_1
        # Topologically Sorted Source Nodes: [x_655, x_656, x_657], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf235 = extern_kernels.convolution(buf234, arg501_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg501_1
        del buf234
        buf236 = buf229; del buf229  # reuse
        # Topologically Sorted Source Nodes: [x_658, x_659, x_660], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_23.run(buf236, buf235, arg502_1, arg503_1, arg504_1, arg505_1, 802816, grid=grid(802816), stream=stream0)
        del arg502_1
        del arg503_1
        del arg504_1
        del arg505_1
        del buf235
        # Topologically Sorted Source Nodes: [x_661], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, arg506_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
        del arg506_1
        buf238 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [x_662, x_663], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf238, arg507_1, arg508_1, arg509_1, arg510_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg507_1
        del arg508_1
        del arg509_1
        del arg510_1
        buf239 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_662, x_663, x_664], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(arg511_1, buf239, 524288, 9, grid=grid(524288, 9), stream=stream0)
        del arg511_1
        # Topologically Sorted Source Nodes: [x_662, x_663, x_664], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf240 = extern_kernels.convolution(buf238, buf239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf240, (8, 4096, 7, 7), (200704, 1, 28672, 4096))
        del buf238
        del buf239
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [x_665, x_666], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf241, arg512_1, arg513_1, arg514_1, arg515_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg512_1
        del arg513_1
        del arg514_1
        del arg515_1
        # Topologically Sorted Source Nodes: [x_665, x_666, x_667], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
        buf242 = extern_kernels.convolution(buf241, arg516_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 2048, 7, 7), (100352, 1, 14336, 2048))
        del arg516_1
        del buf241
        buf244 = empty_strided_cuda((8, 2048, 1, 1), (2048, 1, 16384, 16384), torch.float32)
        # Topologically Sorted Source Nodes: [x_668, x_669, x_670, x_671], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_24.run(buf242, arg517_1, arg518_1, arg519_1, arg520_1, buf236, buf244, 16384, 49, grid=grid(16384), stream=stream0)
        del arg517_1
        del arg518_1
        del arg519_1
        del arg520_1
        del buf236
        del buf242
        buf245 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_673], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg522_1, reinterpret_tensor(buf244, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg521_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf245)
        del arg521_1
        del arg522_1
        del buf244
    return (buf245, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swsl_resnext101_32x16d', benchmark_compiled_module)
