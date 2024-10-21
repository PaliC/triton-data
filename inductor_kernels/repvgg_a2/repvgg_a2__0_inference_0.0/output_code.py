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


# kernel path: /tmp/torchinductor_sahanp/f6/cf6rq5plsuuq7d3uoyka66trri7pvs6ztcunc6377volqqihzips.py
# Topologically Sorted Source Nodes: [x_149, x_151], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_149 => convolution_44
#   x_151 => convolution_45
# Graph fragment:
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr1 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/g4/cg44vkswcejqn6yhzia3bvq33k4wssu3n2hwq3ducaq2xqudsjyz.py
# Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_151 => convolution_45
# Graph fragment:
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
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


# kernel path: /tmp/torchinductor_sahanp/p6/cp64nbtko6vm43u2eg3itmjp2kvtj7c3jjyzxagycdxup4jbn4ey.py
# Topologically Sorted Source Nodes: [x_150, x_152, x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   x_150 => add_162, mul_184, mul_185, sub_61
#   x_152 => add_164, mul_187, mul_188, sub_62
#   x_153 => add_165
#   x_154 => relu_22
# Graph fragment:
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_489), kwargs = {})
#   %mul_184 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %unsqueeze_491), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_184, %unsqueeze_493), kwargs = {})
#   %add_162 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_185, %unsqueeze_495), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_497), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %unsqueeze_501), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_503), kwargs = {})
#   %add_165 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_162, %add_164), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_165,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_sahanp/2x/c2xpefkgbtyowgllhxjhi6z5zcgibyc3fxucigcfivuqny6pca6l.py
# Topologically Sorted Source Nodes: [x_157], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_157 => convolution_47
# Graph fragment:
#   %convolution_47 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_22, %arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
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


# kernel path: /tmp/torchinductor_sahanp/oh/cohiuip4lhcredhahj7a5s6rdwlctcmejvmpwwdqh2aej4gkju2n.py
# Topologically Sorted Source Nodes: [x_156, x_158, x_159, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_22 => relu_23
#   x_156 => add_167, mul_190, mul_191, sub_63
#   x_158 => add_169, mul_193, mul_194, sub_64
#   x_159 => add_170
# Graph fragment:
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_505), kwargs = {})
#   %mul_190 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_190, %unsqueeze_509), kwargs = {})
#   %add_167 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_191, %unsqueeze_511), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_47, %unsqueeze_513), kwargs = {})
#   %mul_193 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %unsqueeze_515), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %unsqueeze_517), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %unsqueeze_519), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_167, %add_169), kwargs = {})
#   %relu_23 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_170,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 96
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


# kernel path: /tmp/torchinductor_sahanp/4n/c4njlrsjhw5wlthfnekzjb4ckxk4uffx7kgnwxk66v5yfi7ai4xs.py
# Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_163 => convolution_49
# Graph fragment:
#   %convolution_49 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_23, %arg30_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_5 = async_compile.triton('triton_poi_fused_convolution_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (864*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/kk/ckke3awrhzm5s4gx2ja36ve7i5uz3yhsdawu6n2riydiqh7utfp5.py
# Topologically Sorted Source Nodes: [x_162, x_164, x_165, x_160, x_166, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_23 => relu_24
#   x_160 => add_172, mul_196, mul_197, sub_65
#   x_162 => add_174, mul_199, mul_200, sub_66
#   x_164 => add_176, mul_202, mul_203, sub_67
#   x_165 => add_177
#   x_166 => add_178
# Graph fragment:
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %unsqueeze_529), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %unsqueeze_531), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_199, %unsqueeze_533), kwargs = {})
#   %add_174 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_200, %unsqueeze_535), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_537), kwargs = {})
#   %mul_202 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %unsqueeze_541), kwargs = {})
#   %add_176 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %unsqueeze_543), kwargs = {})
#   %add_177 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_174, %add_176), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_23, %unsqueeze_521), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %unsqueeze_523), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_196, %unsqueeze_525), kwargs = {})
#   %add_172 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_197, %unsqueeze_527), kwargs = {})
#   %add_178 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_177, %add_172), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_178,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 96
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
    tmp30 = tl.load(in_out_ptr1 + (x2), None)
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
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
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp4
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp7 / tmp35
    tmp37 = tmp36 * tmp9
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp29 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(in_out_ptr1 + (x2), tmp45, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/n6/cn6qihkge3ncptzczp43fua7pcfk625o2wjly5wjrpc2iedoknwq.py
# Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_169 => convolution_51
# Graph fragment:
#   %convolution_51 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %arg40_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18432
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (864*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/iu/ciuakrzjup6ahn2ymmubcvkxlbb5w3olkidmajuvwrk37lwj7zu2.py
# Topologically Sorted Source Nodes: [x_168, x_170, x_171, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_24 => relu_25
#   x_168 => add_180, mul_205, mul_206, sub_68
#   x_170 => add_182, mul_208, mul_209, sub_69
#   x_171 => add_183
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_545), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_205, %unsqueeze_549), kwargs = {})
#   %add_180 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_206, %unsqueeze_551), kwargs = {})
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_553), kwargs = {})
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_209 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_208, %unsqueeze_557), kwargs = {})
#   %add_182 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_209, %unsqueeze_559), kwargs = {})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_180, %add_182), kwargs = {})
#   %relu_25 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_183,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 192
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


# kernel path: /tmp/torchinductor_sahanp/6e/c6eqy2frw5saz3qwitmyjbf65f4bbpjnlrvrsk3dpsgvpji74u36.py
# Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_175 => convolution_53
# Graph fragment:
#   %convolution_53 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %arg54_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/xk/cxk2lmhkj5nuytixyp2wcwc3tdqghvoxaz2w25qzwyfrtcfzibnr.py
# Topologically Sorted Source Nodes: [x_174, x_176, x_177, x_172, x_178, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_25 => relu_26
#   x_172 => add_185, mul_211, mul_212, sub_70
#   x_174 => add_187, mul_214, mul_215, sub_71
#   x_176 => add_189, mul_217, mul_218, sub_72
#   x_177 => add_190
#   x_178 => add_191
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_569), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_573), kwargs = {})
#   %add_187 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_575), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_577), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_217, %unsqueeze_581), kwargs = {})
#   %add_189 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_218, %unsqueeze_583), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_187, %add_189), kwargs = {})
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_25, %unsqueeze_561), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_211, %unsqueeze_565), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_212, %unsqueeze_567), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_190, %add_185), kwargs = {})
#   %relu_26 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_191,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 192
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
    tmp30 = tl.load(in_out_ptr1 + (x2), None)
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
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
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp4
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp7 / tmp35
    tmp37 = tmp36 * tmp9
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp29 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(in_out_ptr1 + (x2), tmp45, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/b6/cb65gcjjaswuzxdqr7aar5ot7cf6yi5on7twrtus37ycli7nhkln.py
# Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_195 => convolution_59
# Graph fragment:
#   %convolution_59 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_28, %arg92_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_11 = async_compile.triton('triton_poi_fused_convolution_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/on/conjvhbqsbdfeushppl4p3for42ykwkmkqzweff3cbwf6fv74qbp.py
# Topologically Sorted Source Nodes: [x_194, x_196, x_197, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_28 => relu_29
#   x_194 => add_209, mul_238, mul_239, sub_79
#   x_196 => add_211, mul_241, mul_242, sub_80
#   x_197 => add_212
# Graph fragment:
#   %sub_79 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_633), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_79, %unsqueeze_635), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_238, %unsqueeze_637), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_239, %unsqueeze_639), kwargs = {})
#   %sub_80 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_59, %unsqueeze_641), kwargs = {})
#   %mul_241 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_80, %unsqueeze_643), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_241, %unsqueeze_645), kwargs = {})
#   %add_211 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_242, %unsqueeze_647), kwargs = {})
#   %add_212 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_209, %add_211), kwargs = {})
#   %relu_29 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_212,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
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


# kernel path: /tmp/torchinductor_sahanp/f2/cf2sk2lbe35ptm36t5znujndnppimzolfu7h5aqbbsqzlcnjoeji.py
# Topologically Sorted Source Nodes: [x_201], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_201 => convolution_61
# Graph fragment:
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_29, %arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_13 = async_compile.triton('triton_poi_fused_convolution_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_13(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (3456*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/ii/cii4mpwehme4xzg2z3gpytxr2ciy6ln5pi42lmiwxmq64oeoqkfi.py
# Topologically Sorted Source Nodes: [x_200, x_202, x_203, x_198, x_204, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_29 => relu_30
#   x_198 => add_214, mul_244, mul_245, sub_81
#   x_200 => add_216, mul_247, mul_248, sub_82
#   x_202 => add_218, mul_250, mul_251, sub_83
#   x_203 => add_219
#   x_204 => add_220
# Graph fragment:
#   %sub_82 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_60, %unsqueeze_657), kwargs = {})
#   %mul_247 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_82, %unsqueeze_659), kwargs = {})
#   %mul_248 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_247, %unsqueeze_661), kwargs = {})
#   %add_216 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_248, %unsqueeze_663), kwargs = {})
#   %sub_83 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_61, %unsqueeze_665), kwargs = {})
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_83, %unsqueeze_667), kwargs = {})
#   %mul_251 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_250, %unsqueeze_669), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_251, %unsqueeze_671), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_216, %add_218), kwargs = {})
#   %sub_81 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu_29, %unsqueeze_649), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_81, %unsqueeze_651), kwargs = {})
#   %mul_245 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_244, %unsqueeze_653), kwargs = {})
#   %add_214 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_245, %unsqueeze_655), kwargs = {})
#   %add_220 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_219, %add_214), kwargs = {})
#   %relu_30 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_220,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 384
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
    tmp30 = tl.load(in_out_ptr1 + (x2), None)
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x0), None, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
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
    tmp32 = tmp30 - tmp31
    tmp34 = tmp33 + tmp4
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tmp7 / tmp35
    tmp37 = tmp36 * tmp9
    tmp38 = tmp32 * tmp37
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp29 + tmp42
    tmp44 = tl.full([1], 0, tl.int32)
    tmp45 = triton_helpers.maximum(tmp44, tmp43)
    tl.store(in_out_ptr1 + (x2), tmp45, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/e7/ce7eiwtxmowjk3aci3ogx7igwdjf4g6arkbr42b2h4uyt75iwtai.py
# Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_291 => convolution_87
# Graph fragment:
#   %convolution_87 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_42, %arg284_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_15 = async_compile.triton('triton_poi_fused_convolution_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1048576, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_15(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 540672
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (3456*y1)), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/hs/chsp5q4uhclczbjolfk6iy3vixotmg5d25czzbj5r6c4czqizr3t.py
# Topologically Sorted Source Nodes: [x_290, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_290 => add_318, mul_361, mul_362, sub_120
#   x_292 => add_320, mul_364, mul_365, sub_121
#   x_293 => add_321
# Graph fragment:
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_961), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_963), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_361, %unsqueeze_965), kwargs = {})
#   %add_318 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_362, %unsqueeze_967), kwargs = {})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_969), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %unsqueeze_971), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_364, %unsqueeze_973), kwargs = {})
#   %add_320 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_365, %unsqueeze_975), kwargs = {})
#   %add_321 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_318, %add_320), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 551936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1408
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
    tl.store(in_out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_sahanp/qj/cqjztyqeycn72izuhmbatg4ylokzcizfc5ainnx7pep2w3itc7sb.py
# Topologically Sorted Source Nodes: [input_42, x_294], Original ATen: [aten.relu, aten.mean]
# Source node to ATen node mapping:
#   input_42 => relu_43
#   x_294 => mean_1
# Graph fragment:
#   %relu_43 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_321,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_43, [-1, -2], True), kwargs = {})
triton_per_fused_mean_relu_17 = async_compile.triton('triton_per_fused_mean_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_relu_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '369B60C25BDE3BDB2552679A82B7C6B5364D80AF90BAFCF99210C0F49B749E9C', 'are_deterministic_algorithms_enabled': True, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_relu_17(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 11264
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 1408
    x1 = (xindex // 1408)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1408*r2) + (68992*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(arg1_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg12_1, (96, ), (1, ))
    assert_size_stride(arg13_1, (96, ), (1, ))
    assert_size_stride(arg14_1, (96, ), (1, ))
    assert_size_stride(arg15_1, (96, ), (1, ))
    assert_size_stride(arg16_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg17_1, (96, ), (1, ))
    assert_size_stride(arg18_1, (96, ), (1, ))
    assert_size_stride(arg19_1, (96, ), (1, ))
    assert_size_stride(arg20_1, (96, ), (1, ))
    assert_size_stride(arg21_1, (96, ), (1, ))
    assert_size_stride(arg22_1, (96, ), (1, ))
    assert_size_stride(arg23_1, (96, ), (1, ))
    assert_size_stride(arg24_1, (96, ), (1, ))
    assert_size_stride(arg25_1, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg26_1, (96, ), (1, ))
    assert_size_stride(arg27_1, (96, ), (1, ))
    assert_size_stride(arg28_1, (96, ), (1, ))
    assert_size_stride(arg29_1, (96, ), (1, ))
    assert_size_stride(arg30_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg31_1, (96, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (192, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg36_1, (192, ), (1, ))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (192, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg41_1, (192, ), (1, ))
    assert_size_stride(arg42_1, (192, ), (1, ))
    assert_size_stride(arg43_1, (192, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (192, ), (1, ))
    assert_size_stride(arg49_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (192, ), (1, ))
    assert_size_stride(arg53_1, (192, ), (1, ))
    assert_size_stride(arg54_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg55_1, (192, ), (1, ))
    assert_size_stride(arg56_1, (192, ), (1, ))
    assert_size_stride(arg57_1, (192, ), (1, ))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (192, ), (1, ))
    assert_size_stride(arg61_1, (192, ), (1, ))
    assert_size_stride(arg62_1, (192, ), (1, ))
    assert_size_stride(arg63_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg64_1, (192, ), (1, ))
    assert_size_stride(arg65_1, (192, ), (1, ))
    assert_size_stride(arg66_1, (192, ), (1, ))
    assert_size_stride(arg67_1, (192, ), (1, ))
    assert_size_stride(arg68_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg69_1, (192, ), (1, ))
    assert_size_stride(arg70_1, (192, ), (1, ))
    assert_size_stride(arg71_1, (192, ), (1, ))
    assert_size_stride(arg72_1, (192, ), (1, ))
    assert_size_stride(arg73_1, (192, ), (1, ))
    assert_size_stride(arg74_1, (192, ), (1, ))
    assert_size_stride(arg75_1, (192, ), (1, ))
    assert_size_stride(arg76_1, (192, ), (1, ))
    assert_size_stride(arg77_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg78_1, (192, ), (1, ))
    assert_size_stride(arg79_1, (192, ), (1, ))
    assert_size_stride(arg80_1, (192, ), (1, ))
    assert_size_stride(arg81_1, (192, ), (1, ))
    assert_size_stride(arg82_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg83_1, (192, ), (1, ))
    assert_size_stride(arg84_1, (192, ), (1, ))
    assert_size_stride(arg85_1, (192, ), (1, ))
    assert_size_stride(arg86_1, (192, ), (1, ))
    assert_size_stride(arg87_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (384, ), (1, ))
    assert_size_stride(arg97_1, (384, ), (1, ))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (384, ), (1, ))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (384, ), (1, ))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (384, ), (1, ))
    assert_size_stride(arg119_1, (384, ), (1, ))
    assert_size_stride(arg120_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg121_1, (384, ), (1, ))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (384, ), (1, ))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg130_1, (384, ), (1, ))
    assert_size_stride(arg131_1, (384, ), (1, ))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (384, ), (1, ))
    assert_size_stride(arg134_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg135_1, (384, ), (1, ))
    assert_size_stride(arg136_1, (384, ), (1, ))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (384, ), (1, ))
    assert_size_stride(arg143_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, ), (1, ))
    assert_size_stride(arg148_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (384, ), (1, ))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (384, ), (1, ))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (384, ), (1, ))
    assert_size_stride(arg185_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg186_1, (384, ), (1, ))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg191_1, (384, ), (1, ))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (384, ), (1, ))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (384, ), (1, ))
    assert_size_stride(arg203_1, (384, ), (1, ))
    assert_size_stride(arg204_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (384, ), (1, ))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (384, ), (1, ))
    assert_size_stride(arg212_1, (384, ), (1, ))
    assert_size_stride(arg213_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (384, ), (1, ))
    assert_size_stride(arg218_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (384, ), (1, ))
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (384, ), (1, ))
    assert_size_stride(arg236_1, (384, ), (1, ))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (384, ), (1, ))
    assert_size_stride(arg244_1, (384, ), (1, ))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg247_1, (384, ), (1, ))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (384, ), (1, ))
    assert_size_stride(arg255_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (384, ), (1, ))
    assert_size_stride(arg264_1, (384, ), (1, ))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg275_1, (384, ), (1, ))
    assert_size_stride(arg276_1, (384, ), (1, ))
    assert_size_stride(arg277_1, (384, ), (1, ))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (1408, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg280_1, (1408, ), (1, ))
    assert_size_stride(arg281_1, (1408, ), (1, ))
    assert_size_stride(arg282_1, (1408, ), (1, ))
    assert_size_stride(arg283_1, (1408, ), (1, ))
    assert_size_stride(arg284_1, (1408, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg285_1, (1408, ), (1, ))
    assert_size_stride(arg286_1, (1408, ), (1, ))
    assert_size_stride(arg287_1, (1408, ), (1, ))
    assert_size_stride(arg288_1, (1408, ), (1, ))
    assert_size_stride(arg289_1, (1000, 1408), (1408, 1))
    assert_size_stride(arg290_1, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        buf2 = empty_strided_cuda((8, 3, 224, 224), (150528, 1, 672, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_149, x_151], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(arg1_1, buf0, buf2, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg1_1
        # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg0_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (8, 64, 112, 112), (802816, 1, 7168, 64))
        del arg0_1
        del buf0
        buf3 = empty_strided_cuda((64, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg6_1, buf3, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg6_1
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 64, 112, 112), (802816, 1, 7168, 64))
        del buf3
        buf5 = buf1; del buf1  # reuse
        buf6 = empty_strided_cuda((8, 64, 112, 112), (802816, 1, 7168, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_150, x_152, x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_2.run(buf5, arg2_1, arg3_1, arg4_1, arg5_1, buf4, arg7_1, arg8_1, arg9_1, arg10_1, buf6, 6422528, grid=grid(6422528), stream=stream0)
        del arg10_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg7_1
        del arg8_1
        del arg9_1
        del buf4
        del buf5
        # Topologically Sorted Source Nodes: [x_154, x_155], Original ATen: [aten.relu, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg11_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg11_1
        buf8 = empty_strided_cuda((96, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_157], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_3.run(arg16_1, buf8, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg16_1
        # Topologically Sorted Source Nodes: [x_157], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf6, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del buf6
        del buf8
        buf10 = buf7; del buf7  # reuse
        buf11 = empty_strided_cuda((8, 96, 56, 56), (301056, 1, 5376, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_156, x_158, x_159, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf10, arg12_1, arg13_1, arg14_1, arg15_1, buf9, arg17_1, arg18_1, arg19_1, arg20_1, buf11, 2408448, grid=grid(2408448), stream=stream0)
        del arg12_1
        del arg13_1
        del arg14_1
        del arg15_1
        del arg17_1
        del arg18_1
        del arg19_1
        del arg20_1
        del buf10
        del buf9
        # Topologically Sorted Source Nodes: [input_22, x_161], Original ATen: [aten.relu, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg25_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del arg25_1
        buf13 = empty_strided_cuda((96, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg30_1, buf13, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg30_1
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf11, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 96, 56, 56), (301056, 1, 5376, 96))
        del buf13
        buf15 = buf12; del buf12  # reuse
        buf16 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x_162, x_164, x_165, x_160, x_166, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf15, buf16, arg26_1, arg27_1, arg28_1, arg29_1, buf14, arg31_1, arg32_1, arg33_1, arg34_1, arg21_1, arg22_1, arg23_1, arg24_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        del arg24_1
        del arg26_1
        del arg27_1
        del arg28_1
        del arg29_1
        del arg31_1
        del arg32_1
        del arg33_1
        del arg34_1
        del buf14
        del buf15
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg35_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg35_1
        buf18 = empty_strided_cuda((192, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(arg40_1, buf18, 18432, 9, grid=grid(18432, 9), stream=stream0)
        del arg40_1
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf16, buf18, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del buf16
        del buf18
        buf20 = buf17; del buf17  # reuse
        buf21 = reinterpret_tensor(buf2, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_168, x_170, x_171, input_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf20, arg36_1, arg37_1, arg38_1, arg39_1, buf19, arg41_1, arg42_1, arg43_1, arg44_1, buf21, 1204224, grid=grid(1204224), stream=stream0)
        del arg36_1
        del arg37_1
        del arg38_1
        del arg39_1
        del arg41_1
        del arg42_1
        del arg43_1
        del arg44_1
        del buf19
        del buf20
        # Topologically Sorted Source Nodes: [input_24, x_173], Original ATen: [aten.relu, aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg49_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg49_1
        buf23 = empty_strided_cuda((192, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg54_1, buf23, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg54_1
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf21, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 192, 28, 28), (150528, 1, 5376, 192))
        buf25 = buf22; del buf22  # reuse
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_174, x_176, x_177, x_172, x_178, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf25, buf26, arg50_1, arg51_1, arg52_1, arg53_1, buf24, arg55_1, arg56_1, arg57_1, arg58_1, arg45_1, arg46_1, arg47_1, arg48_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg45_1
        del arg46_1
        del arg47_1
        del arg48_1
        del arg50_1
        del arg51_1
        del arg52_1
        del arg53_1
        del arg55_1
        del arg56_1
        del arg57_1
        del arg58_1
        del buf24
        del buf25
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg63_1
        buf28 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg68_1, buf28, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg68_1
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf26, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 192, 28, 28), (150528, 1, 5376, 192))
        buf30 = buf27; del buf27  # reuse
        buf31 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_181, x_183, x_184, x_179, x_185, input_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf30, buf31, arg64_1, arg65_1, arg66_1, arg67_1, buf29, arg69_1, arg70_1, arg71_1, arg72_1, arg59_1, arg60_1, arg61_1, arg62_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg59_1
        del arg60_1
        del arg61_1
        del arg62_1
        del arg64_1
        del arg65_1
        del arg66_1
        del arg67_1
        del arg69_1
        del arg70_1
        del arg71_1
        del arg72_1
        del buf29
        del buf30
        # Topologically Sorted Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg77_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del arg77_1
        buf33 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(arg82_1, buf33, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg82_1
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf31, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 192, 28, 28), (150528, 1, 5376, 192))
        del buf33
        buf35 = buf32; del buf32  # reuse
        buf36 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [x_188, x_190, x_191, x_186, x_192, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf35, buf36, arg78_1, arg79_1, arg80_1, arg81_1, buf34, arg83_1, arg84_1, arg85_1, arg86_1, arg73_1, arg74_1, arg75_1, arg76_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg73_1
        del arg74_1
        del arg75_1
        del arg76_1
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        del arg83_1
        del arg84_1
        del arg85_1
        del arg86_1
        del buf34
        del buf35
        # Topologically Sorted Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg87_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg87_1
        buf38 = empty_strided_cuda((384, 192, 3, 3), (1728, 1, 576, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_11.run(arg92_1, buf38, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del arg92_1
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf36, buf38, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del buf36
        del buf38
        buf40 = buf37; del buf37  # reuse
        buf41 = empty_strided_cuda((8, 384, 14, 14), (75264, 1, 5376, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_194, x_196, x_197, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf40, arg88_1, arg89_1, arg90_1, arg91_1, buf39, arg93_1, arg94_1, arg95_1, arg96_1, buf41, 602112, grid=grid(602112), stream=stream0)
        del arg88_1
        del arg89_1
        del arg90_1
        del arg91_1
        del arg93_1
        del arg94_1
        del arg95_1
        del arg96_1
        del buf39
        del buf40
        # Topologically Sorted Source Nodes: [input_28, x_199], Original ATen: [aten.relu, aten.convolution]
        buf42 = extern_kernels.convolution(buf41, arg101_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg101_1
        buf43 = empty_strided_cuda((384, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg106_1, buf43, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg106_1
        # Topologically Sorted Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf41, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf45 = buf42; del buf42  # reuse
        buf46 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [x_200, x_202, x_203, x_198, x_204, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf45, buf46, arg102_1, arg103_1, arg104_1, arg105_1, buf44, arg107_1, arg108_1, arg109_1, arg110_1, arg97_1, arg98_1, arg99_1, arg100_1, 602112, grid=grid(602112), stream=stream0)
        del arg100_1
        del arg102_1
        del arg103_1
        del arg104_1
        del arg105_1
        del arg107_1
        del arg108_1
        del arg109_1
        del arg110_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf44
        del buf45
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg115_1
        buf48 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_208], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg120_1, buf48, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg120_1
        # Topologically Sorted Source Nodes: [x_208], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf46, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf50 = buf47; del buf47  # reuse
        buf51 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [x_207, x_209, x_210, x_205, x_211, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf50, buf51, arg116_1, arg117_1, arg118_1, arg119_1, buf49, arg121_1, arg122_1, arg123_1, arg124_1, arg111_1, arg112_1, arg113_1, arg114_1, 602112, grid=grid(602112), stream=stream0)
        del arg111_1
        del arg112_1
        del arg113_1
        del arg114_1
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del arg121_1
        del arg122_1
        del arg123_1
        del arg124_1
        del buf49
        del buf50
        # Topologically Sorted Source Nodes: [x_213], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg129_1
        buf53 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg134_1, buf53, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg134_1
        # Topologically Sorted Source Nodes: [x_215], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf51, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf55 = buf52; del buf52  # reuse
        buf56 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_214, x_216, x_217, x_212, x_218, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf55, buf56, arg130_1, arg131_1, arg132_1, arg133_1, buf54, arg135_1, arg136_1, arg137_1, arg138_1, arg125_1, arg126_1, arg127_1, arg128_1, 602112, grid=grid(602112), stream=stream0)
        del arg125_1
        del arg126_1
        del arg127_1
        del arg128_1
        del arg130_1
        del arg131_1
        del arg132_1
        del arg133_1
        del arg135_1
        del arg136_1
        del arg137_1
        del arg138_1
        del buf54
        del buf55
        # Topologically Sorted Source Nodes: [x_220], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg143_1
        buf58 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg148_1, buf58, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg148_1
        # Topologically Sorted Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf56, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf60 = buf57; del buf57  # reuse
        buf61 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_221, x_223, x_224, x_219, x_225, input_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf60, buf61, arg144_1, arg145_1, arg146_1, arg147_1, buf59, arg149_1, arg150_1, arg151_1, arg152_1, arg139_1, arg140_1, arg141_1, arg142_1, 602112, grid=grid(602112), stream=stream0)
        del arg139_1
        del arg140_1
        del arg141_1
        del arg142_1
        del arg144_1
        del arg145_1
        del arg146_1
        del arg147_1
        del arg149_1
        del arg150_1
        del arg151_1
        del arg152_1
        del buf59
        del buf60
        # Topologically Sorted Source Nodes: [x_227], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg157_1
        buf63 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg162_1, buf63, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg162_1
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf61, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf65 = buf62; del buf62  # reuse
        buf66 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_228, x_230, x_231, x_226, x_232, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf65, buf66, arg158_1, arg159_1, arg160_1, arg161_1, buf64, arg163_1, arg164_1, arg165_1, arg166_1, arg153_1, arg154_1, arg155_1, arg156_1, 602112, grid=grid(602112), stream=stream0)
        del arg153_1
        del arg154_1
        del arg155_1
        del arg156_1
        del arg158_1
        del arg159_1
        del arg160_1
        del arg161_1
        del arg163_1
        del arg164_1
        del arg165_1
        del arg166_1
        del buf64
        del buf65
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg171_1
        buf68 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg176_1, buf68, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg176_1
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf66, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf70 = buf67; del buf67  # reuse
        buf71 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [x_235, x_237, x_238, x_233, x_239, input_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf70, buf71, arg172_1, arg173_1, arg174_1, arg175_1, buf69, arg177_1, arg178_1, arg179_1, arg180_1, arg167_1, arg168_1, arg169_1, arg170_1, 602112, grid=grid(602112), stream=stream0)
        del arg167_1
        del arg168_1
        del arg169_1
        del arg170_1
        del arg172_1
        del arg173_1
        del arg174_1
        del arg175_1
        del arg177_1
        del arg178_1
        del arg179_1
        del arg180_1
        del buf69
        del buf70
        # Topologically Sorted Source Nodes: [x_241], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg185_1
        buf73 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg190_1, buf73, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg190_1
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf71, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf75 = buf72; del buf72  # reuse
        buf76 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [x_242, x_244, x_245, x_240, x_246, input_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf75, buf76, arg186_1, arg187_1, arg188_1, arg189_1, buf74, arg191_1, arg192_1, arg193_1, arg194_1, arg181_1, arg182_1, arg183_1, arg184_1, 602112, grid=grid(602112), stream=stream0)
        del arg181_1
        del arg182_1
        del arg183_1
        del arg184_1
        del arg186_1
        del arg187_1
        del arg188_1
        del arg189_1
        del arg191_1
        del arg192_1
        del arg193_1
        del arg194_1
        del buf74
        del buf75
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg199_1
        buf78 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_250], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg204_1, buf78, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg204_1
        # Topologically Sorted Source Nodes: [x_250], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf76, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf80 = buf77; del buf77  # reuse
        buf81 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_249, x_251, x_252, x_247, x_253, input_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf80, buf81, arg200_1, arg201_1, arg202_1, arg203_1, buf79, arg205_1, arg206_1, arg207_1, arg208_1, arg195_1, arg196_1, arg197_1, arg198_1, 602112, grid=grid(602112), stream=stream0)
        del arg195_1
        del arg196_1
        del arg197_1
        del arg198_1
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg205_1
        del arg206_1
        del arg207_1
        del arg208_1
        del buf79
        del buf80
        # Topologically Sorted Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg213_1
        buf83 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg218_1, buf83, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg218_1
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf81, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf85 = buf82; del buf82  # reuse
        buf86 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_256, x_258, x_259, x_254, x_260, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf85, buf86, arg214_1, arg215_1, arg216_1, arg217_1, buf84, arg219_1, arg220_1, arg221_1, arg222_1, arg209_1, arg210_1, arg211_1, arg212_1, 602112, grid=grid(602112), stream=stream0)
        del arg209_1
        del arg210_1
        del arg211_1
        del arg212_1
        del arg214_1
        del arg215_1
        del arg216_1
        del arg217_1
        del arg219_1
        del arg220_1
        del arg221_1
        del arg222_1
        del buf84
        del buf85
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg227_1
        buf88 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_264], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg232_1, buf88, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg232_1
        # Topologically Sorted Source Nodes: [x_264], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf86, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf90 = buf87; del buf87  # reuse
        buf91 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [x_263, x_265, x_266, x_261, x_267, input_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf90, buf91, arg228_1, arg229_1, arg230_1, arg231_1, buf89, arg233_1, arg234_1, arg235_1, arg236_1, arg223_1, arg224_1, arg225_1, arg226_1, 602112, grid=grid(602112), stream=stream0)
        del arg223_1
        del arg224_1
        del arg225_1
        del arg226_1
        del arg228_1
        del arg229_1
        del arg230_1
        del arg231_1
        del arg233_1
        del arg234_1
        del arg235_1
        del arg236_1
        del buf89
        del buf90
        # Topologically Sorted Source Nodes: [x_269], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg241_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg241_1
        buf93 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg246_1, buf93, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg246_1
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf91, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf95 = buf92; del buf92  # reuse
        buf96 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_270, x_272, x_273, x_268, x_274, input_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf95, buf96, arg242_1, arg243_1, arg244_1, arg245_1, buf94, arg247_1, arg248_1, arg249_1, arg250_1, arg237_1, arg238_1, arg239_1, arg240_1, 602112, grid=grid(602112), stream=stream0)
        del arg237_1
        del arg238_1
        del arg239_1
        del arg240_1
        del arg242_1
        del arg243_1
        del arg244_1
        del arg245_1
        del arg247_1
        del arg248_1
        del arg249_1
        del arg250_1
        del buf94
        del buf95
        # Topologically Sorted Source Nodes: [x_276], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg255_1
        buf98 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg260_1, buf98, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg260_1
        # Topologically Sorted Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf96, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 384, 14, 14), (75264, 1, 5376, 384))
        buf100 = buf97; del buf97  # reuse
        buf101 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_277, x_279, x_280, x_275, x_281, input_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf100, buf101, arg256_1, arg257_1, arg258_1, arg259_1, buf99, arg261_1, arg262_1, arg263_1, arg264_1, arg251_1, arg252_1, arg253_1, arg254_1, 602112, grid=grid(602112), stream=stream0)
        del arg251_1
        del arg252_1
        del arg253_1
        del arg254_1
        del arg256_1
        del arg257_1
        del arg258_1
        del arg259_1
        del arg261_1
        del arg262_1
        del arg263_1
        del arg264_1
        del buf100
        del buf99
        # Topologically Sorted Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del arg269_1
        buf103 = buf98; del buf98  # reuse
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_13.run(arg274_1, buf103, 147456, 9, grid=grid(147456, 9), stream=stream0)
        del arg274_1
        # Topologically Sorted Source Nodes: [x_285], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf101, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 384, 14, 14), (75264, 1, 5376, 384))
        del buf103
        buf105 = buf102; del buf102  # reuse
        buf106 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_284, x_286, x_287, x_282, x_288, input_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf105, buf106, arg270_1, arg271_1, arg272_1, arg273_1, buf104, arg275_1, arg276_1, arg277_1, arg278_1, arg265_1, arg266_1, arg267_1, arg268_1, 602112, grid=grid(602112), stream=stream0)
        del arg265_1
        del arg266_1
        del arg267_1
        del arg268_1
        del arg270_1
        del arg271_1
        del arg272_1
        del arg273_1
        del arg275_1
        del arg276_1
        del arg277_1
        del arg278_1
        del buf104
        del buf105
        # Topologically Sorted Source Nodes: [x_289], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg279_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
        del arg279_1
        buf108 = empty_strided_cuda((1408, 384, 3, 3), (3456, 1, 1152, 384), torch.float32)
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg284_1, buf108, 540672, 9, grid=grid(540672, 9), stream=stream0)
        del arg284_1
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf106, buf108, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
        del buf106
        del buf108
        buf110 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_290, x_292, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_16.run(buf110, arg280_1, arg281_1, arg282_1, arg283_1, buf109, arg285_1, arg286_1, arg287_1, arg288_1, 551936, grid=grid(551936), stream=stream0)
        del arg280_1
        del arg281_1
        del arg282_1
        del arg283_1
        del arg285_1
        del arg286_1
        del arg287_1
        del arg288_1
        del buf109
        buf112 = empty_strided_cuda((8, 1408, 1, 1), (1408, 1, 11264, 11264), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, x_294], Original ATen: [aten.relu, aten.mean]
        triton_per_fused_mean_relu_17.run(buf110, buf112, 11264, 49, grid=grid(11264), stream=stream0)
        del buf110
        buf113 = empty_strided_cuda((8, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_297], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg290_1, reinterpret_tensor(buf112, (8, 1408), (1408, 1), 0), reinterpret_tensor(arg289_1, (1408, 1000), (1, 1408), 0), alpha=1, beta=1, out=buf113)
        del arg289_1
        del arg290_1
        del buf112
    return (buf113, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((96, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((192, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1408, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((1408, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((1000, 1408), (1408, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('repvgg_a2', benchmark_compiled_module)
